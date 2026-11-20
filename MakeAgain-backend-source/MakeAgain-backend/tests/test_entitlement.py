"""WP6 behaviour acceptance: server-authoritative entitlement + webhook replay."""
import base64
import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException

from conftest import account
from config import get_settings
from main import app
from memory import models as m
from memory.b07_models import Order
from memory.db import _RawSessionLocal
from api.entitlement import (
    FLAG, can_generate, can_view_assets, entitlement_view,
    require_generate, require_view_assets, router as entitlement_router,
)
from gateway.payments import (
    notification_is_duplicate, notification_replay_count,
    reset_notification_ledger,
)

# The router is normally wired by W0 in main.py; mount it here so the endpoint
# can be exercised without touching the hot file api/routes.py or main.py.
if not any(getattr(r, 'path', None) == '/api/entitlement' for r in app.routes):
    app.include_router(entitlement_router)


def _expire(user, days=1):
    with _RawSessionLocal() as db:
        u = db.get(m.User, user)
        u.membership = 'member'
        u.membership_expires_at = m.utcnow() - timedelta(days=days)
        u.membership_source = 'trial'
        db.commit()


def _activate(user):
    with _RawSessionLocal() as db:
        u = db.get(m.User, user)
        u.membership = 'member'
        u.membership_expires_at = m.utcnow() + timedelta(days=30)
        u.membership_source = 'payment'
        db.commit()


@pytest.fixture
def pay_keys(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = tmp_path / 'merchant.pem'
    public = tmp_path / 'platform.pem'
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    public.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    s = get_settings()
    overrides = dict(
        wechat_appid='test-app', wechat_pay_mchid='test-merchant',
        wechat_pay_serial_no='test-merchant-serial',
        wechat_pay_private_key_path=str(private),
        wechat_pay_platform_key_path=str(public),
        wechat_pay_platform_serial='test-platform-serial',
        wechat_pay_api_v3_key='01234567890123456789012345678901',
        wechat_pay_notify_url='https://test.invalid/api/payments/wechat/notify',
        payment_allowed_platforms='android',
    )
    for name, value in overrides.items():
        monkeypatch.setattr(s, name, value)
    return key


def build_notification(key, order, **overrides):
    data = dict(out_trade_no=order.id, transaction_id='tx-' + order.id,
                appid='test-app', mchid='test-merchant',
                payer={'openid': order.user_id},
                amount={'total': order.amount, 'currency': 'CNY'},
                trade_state='SUCCESS', **overrides)
    nonce, aad = 'wp6abcdef0123', 'transaction'
    encrypted = AESGCM(get_settings().wechat_pay_api_v3_key.encode()).encrypt(
        nonce.encode(), json.dumps(data).encode(), aad.encode())
    raw = json.dumps({'event_type': 'TRANSACTION.SUCCESS',
                      'resource': {'algorithm': 'AEAD_AES_256_GCM', 'nonce': nonce,
                                   'associated_data': aad,
                                   'ciphertext': base64.b64encode(encrypted).decode()}},
                     separators=(',', ':')).encode()
    ts = str(int(time.time()))
    sig = key.sign(ts.encode() + b'\nwp6-nonce\n' + raw + b'\n', padding.PKCS1v15(), hashes.SHA256())
    headers = {'Wechatpay-Timestamp': ts, 'Wechatpay-Nonce': 'wp6-nonce',
               'Wechatpay-Serial': 'test-platform-serial',
               'Wechatpay-Signature': base64.b64encode(sig).decode(),
               'Content-Type': 'application/json'}
    return raw, headers


def make_order(user):
    with _RawSessionLocal() as db:
        order = Order(user_id=user, request_id='wp6-payment', plan_id='plan',
                      snapshot={'name': 'Test only'}, amount=123, duration_days=30,
                      status='unpaid', expires_at=m.utcnow() + timedelta(minutes=30))
        db.add(order)
        db.commit()
        return order


# ---- 1. flag default off ----------------------------------------------------

def test_flag_default_off_keeps_legacy_behaviour(client, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    user, headers, _ = account(client)
    _expire(user)
    body = client.get('/api/entitlement', headers=headers)
    assert body.status_code == 200, body.json()
    data = body.json()
    assert data['gate_enabled'] is False
    assert data['state'] == 'expired'
    assert data['capabilities'] == {'can_generate': True, 'can_view_assets': True}
    assert can_generate(_user(user)) is True
    assert can_generate(_user(user), enabled=False) is True
    assert can_view_assets(_user(user), enabled=True) is True


def _user(user_id):
    with _RawSessionLocal() as db:
        db.expire_all()
        return db.get(m.User, user_id)


# ---- 2. expired trial -------------------------------------------------------

def test_expired_trial_denies_generate_but_allows_asset_view(client, monkeypatch):
    monkeypatch.setenv(FLAG, '1')
    user, headers, _ = account(client)
    _expire(user)

    body = client.get('/api/entitlement', headers=headers)
    assert body.status_code == 200, body.json()
    data = body.json()
    assert data['gate_enabled'] is True
    assert data['state'] == 'expired'
    assert data['membership'] == 'free'
    assert data['trial']['expired'] is True
    assert data['capabilities']['can_generate'] is False
    assert data['capabilities']['can_view_assets'] is True
    assert data['paywall']['blocked_capabilities'] == ['can_generate']

    # Direct trigger-point guards: generation fails closed, read path never blocks.
    with pytest.raises(HTTPException) as exc:
        require_generate(_user(user))
    assert exc.value.status_code == 403
    assert exc.value.detail['code'] == 'ENTITLEMENT_EXPIRED'
    assert require_view_assets(_user(user)) is True

    # A currently active member may keep generating.
    _activate(user)
    active = client.get('/api/entitlement', headers=headers).json()
    assert active['state'] == 'active'
    assert active['capabilities']['can_generate'] is True
    assert active['capabilities']['can_view_assets'] is True


# ---- 3. client forgery ------------------------------------------------------

def test_forged_client_headers_cannot_claim_membership(client, monkeypatch, pay_keys):
    monkeypatch.setenv(FLAG, 'true')
    free, free_headers, _ = account(client)
    rich, rich_headers, _ = account(client, 'rich')
    _activate(rich)

    forged = dict(free_headers)
    forged.update({'X-Membership': 'member', 'X-Entitlement': 'member',
                   'X-Role': 'admin', 'X-User-Id': rich,
                   'Authorization': free_headers['Authorization']})
    data = client.get('/api/entitlement', headers=forged).json()
    assert data['membership'] == 'free'
    assert data['state'] == 'free'
    assert data['capabilities']['can_generate'] is False

    # The reverse forgery (claiming free while genuinely a member) also fails.
    liar = dict(rich_headers)
    liar.update({'X-Membership': 'free', 'X-Entitlement': 'free'})
    assert client.get('/api/entitlement', headers=liar).json()['membership'] == 'member'

    # A forged webhook signature is rejected and never enters the replay ledger.
    reset_notification_ledger()
    fake_headers = {'Wechatpay-Timestamp': str(int(time.time())),
                    'Wechatpay-Nonce': 'forged', 'Wechatpay-Serial': 'test-platform-serial',
                    'Wechatpay-Signature': base64.b64encode(b'not-a-signature').decode()}
    r = client.post('/api/payments/wechat/notify', content=b'{}', headers=fake_headers)
    assert r.status_code == 401, r.json()
    assert notification_is_duplicate(fake_headers, b'{}') is False


# ---- 4. webhook replay ------------------------------------------------------

def test_webhook_replay_three_times_extends_membership_once(client, pay_keys):
    user, headers, _ = account(client)
    order = make_order(user)
    reset_notification_ledger()
    raw, wh_headers = build_notification(pay_keys, order)

    results = [client.post('/api/payments/wechat/notify', content=raw, headers=wh_headers)
               for _ in range(3)]
    assert [r.status_code for r in results] == [200, 200, 200], [r.json() for r in results]
    assert all(r.json()['code'] == 'SUCCESS' for r in results)

    # Gateway replay ledger observed all three accepted deliveries.
    assert notification_replay_count(wh_headers, raw) == 3
    assert notification_is_duplicate(wh_headers, raw) is True

    # Membership was extended exactly once: all subsequent reads are identical.
    expiry = client.get('/api/auth/me', headers=headers).json()['membership_expires_at']
    for _ in range(2):
        assert client.post('/api/payments/wechat/notify', content=raw, headers=wh_headers).status_code == 200
        assert client.get('/api/auth/me', headers=headers).json()['membership_expires_at'] == expiry
    with _RawSessionLocal() as db:
        row = db.get(Order, order.id)
        assert row.status == 'paid'
        assert row.transaction_id == 'tx-' + order.id


# ---- 5. paywall placement ---------------------------------------------------

def test_paywall_is_generation_only_not_low_mood_dialogue(client, monkeypatch):
    monkeypatch.setenv(FLAG, '1')
    user, headers, _ = account(client)
    _expire(user)
    data = client.get('/api/entitlement', headers=headers).json()
    paywall = data['paywall']
    assert paywall['surface'] == 'continue_generation_only'
    assert paywall['show_in_low_mood_dialogue'] is False
    assert paywall['show_on_asset_view'] is False
    assert data['capabilities']['can_view_assets'] is True  # assets stay reachable

    # Source audit: neither the crisis/low-mood path nor the report/asset reads
    # import or call the generation gate. No paywall trigger lives there.
    root = Path(__file__).resolve().parents[1]
    audited = ['emotion/crisis.py', 'emotion/crisis_flow.py', 'agent/treehole.py',
               'agent/echo.py', 'agent/daily.py', 'agent/item.py', 'agent/photo.py']
    forbidden = ('can_generate', 'require_generate', 'entitlement', 'paywall')
    for rel in audited:
        text = (root / rel).read_text(encoding='utf-8')
        assert not any(token in text for token in forbidden), rel

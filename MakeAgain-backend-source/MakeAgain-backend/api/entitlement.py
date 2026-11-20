"""Server-authoritative entitlement.

The client only displays this decision; it never decides membership itself.
Feature flag USE_ENTITLEMENT_GATE is off by default (use os.getenv, no config.py
change).  When off, behaviour is byte-for-byte the legacy one: no capability is
gated.  When on, only *continued generation* is gated; viewing already-produced
memorial assets is never gated, and no paywall is raised inside low-mood/
crisis dialogue.

This is an independent route file.  W0 must include this router in main.py.
"""
import os

from fastapi import APIRouter, Depends

from memory.db import get_session
from memory import models as m
from api.security import SecureRoute, fail, iso

router = APIRouter(prefix='/api', route_class=SecureRoute)

FLAG = 'USE_ENTITLEMENT_GATE'
_TRUTHY = ('1', 'true', 'yes', 'on')


def gate_enabled():
    """Read the flag at call time so tests/W0 can toggle it via the environment."""
    return os.getenv(FLAG, '0').strip().lower() in _TRUTHY


def entitlement_state(user):
    """One of: free | active | expired | perpetual."""
    if getattr(user, 'membership', None) != 'member':
        return 'free'
    if user.membership_expires_at is None:
        return 'perpetual'  # legacy perpetual entitlement
    return 'active' if user.membership_expires_at > m.utcnow() else 'expired'


def _active(user):
    return entitlement_state(user) in ('active', 'perpetual')


def can_generate(user, enabled=None):
    """Capability gate for 'continue generating / continue companion output'.

    Flag off -> always True (legacy behaviour, no gate).
    Flag on  -> True only while the server-side membership is active.
    """
    effective = gate_enabled() if enabled is None else enabled
    if effective:
        return _active(user)
    return True


def can_view_assets(user, enabled=None):
    """Viewing already-generated memorial assets is NEVER gated."""
    return True


def require_generate(user, enabled=None):
    """Call at the generation trigger point. Fails closed when the trial expired."""
    if not can_generate(user, enabled):
        fail(403, 'ENTITLEMENT_EXPIRED', '试用已到期，无法继续生成；已生成的纪念资产仍可查看')
    return True


def require_view_assets(user, enabled=None):
    """Read path: intentionally a no-op guard for an explicit contract."""
    return True


def entitlement_view(user, enabled=None):
    """Server judgment only. No client header is consulted."""
    state = entitlement_state(user)
    effective = gate_enabled() if enabled is None else bool(enabled)
    generate = can_generate(user, effective)
    expiry = iso(user.membership_expires_at)

    return {
        'ok': True,
        'membership': 'member' if state in ('active', 'perpetual') else 'free',
        'membership_expires_at': expiry,
        'membership_source': getattr(user, 'membership_source', None),
        'entitlement_source': getattr(user, 'membership_source', None),
        'entitlement_from_server': True,
        'gate_enabled': effective,
        'state': state,
        'trial': {
            'status': 'none' if state == 'free' else state,
            'expires_at': expiry,
            'expired': state == 'expired',
        },
        'capabilities': {
            'can_generate': generate,
            'can_view_assets': can_view_assets(user, effective),
        },
        'paywall': {
            'surface': 'continue_generation_only',
            'show_in_low_mood_dialogue': False,
            'show_on_asset_view': False,
            'blocked_capabilities': [] if generate else ['can_generate'],
        },
    }


@router.get('/entitlement')
def get_entitlement(db=Depends(get_session)):
    user_id = db.info.get('user_id')
    if not user_id:
        fail(401, 'UNAUTHENTICATED', '请先登录')
    user = db.get(m.User, user_id)
    if not user:
        fail(401, 'SESSION_EXPIRED', '登录已失效')
    # Deliberately ignore X-Membership / X-Entitlement / any client-supplied claim.
    return entitlement_view(user)


@router.get('/account/entitlement')
def get_entitlement_account(db=Depends(get_session)):
    """Alias under the account prefix (no companion requirement) for W0 wiring."""
    return get_entitlement(db)

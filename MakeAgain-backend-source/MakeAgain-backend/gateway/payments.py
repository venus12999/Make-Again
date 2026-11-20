"""WeChat Pay API v3: RSA request/response signatures and AES-GCM notifications.
Official merchant protocol: https://pay.wechatpay.cn/doc/v3/merchant/4012791861
"""
import base64,json,secrets,time,hashlib,threading
from collections import OrderedDict
from pathlib import Path
import httpx
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography import x509
from config import get_settings

class PaymentError(Exception):pass


# --- Webhook replay ledger (defense in depth) ---------------------------------
# The authoritative "grant at most once" guard lives in
# api/subscriptions.py::apply_paid (paid_at / transaction_id fences).  This
# gateway-level ledger makes verifiable replays observable and gives W0 a hook
# to reject them without touching subscriptions.py.  It is intentionally NOT
# enforced as an exception here: WeChat expects a SUCCESS ack for a replay, and
# turning it into a 401 would cause endless provider retries.  Only signatures
# that already passed verify() may enter the ledger, so forged headers can
# never poison it.
_NOTIFY_DEDUPE_MAX = 4096
_notify_seen = OrderedDict()
_notify_lock = threading.Lock()


def notification_fingerprint(headers, body):
    """Stable idempotency key for one provider notification (no secret bytes)."""
    getter = getattr(headers, 'get', lambda _k: None)

    def text(key):
        value = getter(key)
        if isinstance(value, bytes):
            value = value.decode('latin-1')
        return value or ''

    payload_hash = hashlib.sha256(body).hexdigest()
    return '{}|{}|{}|{}'.format(text('Wechatpay-Serial'), text('Wechatpay-Nonce'),
                                payload_hash, text('Wechatpay-Signature')[:24])


def _record_notification(headers, body):
    """Return True when this exact signed notification was already accepted."""
    key = notification_fingerprint(headers, body)
    with _notify_lock:
        duplicate = key in _notify_seen
        _notify_seen[key] = _notify_seen.get(key, 0) + 1
        _notify_seen.move_to_end(key)
        while len(_notify_seen) > _NOTIFY_DEDUPE_MAX:
            _notify_seen.popitem(last=False)
        return duplicate


def notification_is_duplicate(headers, body):
    """Read-only replay check; use after a successful notification() call."""
    with _notify_lock:
        return notification_fingerprint(headers, body) in _notify_seen


def notification_replay_count(headers, body):
    """How many times this exact signed notification has been accepted."""
    with _notify_lock:
        return _notify_seen.get(notification_fingerprint(headers, body), 0)


def reset_notification_ledger():
    """Test-only isolation helper."""
    with _notify_lock:
        _notify_seen.clear()


def configured(s=None):
    s=s or get_settings()
    return all((s.wechat_appid,s.wechat_pay_mchid,s.wechat_pay_serial_no,
                s.wechat_pay_private_key_path,s.wechat_pay_platform_key_path,s.wechat_pay_platform_serial,
                len(s.wechat_pay_api_v3_key.encode())==32,s.wechat_pay_notify_url.startswith('https://'),s.payment_allowed_platforms))

class WeChatPay:
    def __init__(self,settings=None):
        self.s=settings or get_settings()
        if not configured(self.s): raise PaymentError('PAYMENT_NOT_CONFIGURED')
    def sign(self,text):
        key=serialization.load_pem_private_key(Path(self.s.wechat_pay_private_key_path).read_bytes(),password=None)
        return base64.b64encode(key.sign(text.encode(),padding.PKCS1v15(),hashes.SHA256())).decode()
    def verify(self,headers,body):
        try:
            timestamp=headers['Wechatpay-Timestamp'];nonce=headers['Wechatpay-Nonce'];sig=headers['Wechatpay-Signature'];serial=headers['Wechatpay-Serial']
            if abs(int(timestamp)-int(time.time()))>300 or serial!=self.s.wechat_pay_platform_serial:raise ValueError()
            pem=Path(self.s.wechat_pay_platform_key_path).read_bytes()
            key=x509.load_pem_x509_certificate(pem).public_key() if b'BEGIN CERTIFICATE' in pem else serialization.load_pem_public_key(pem)
            key.verify(base64.b64decode(sig,validate=True),timestamp.encode()+b'\n'+nonce.encode()+b'\n'+body+b'\n',padding.PKCS1v15(),hashes.SHA256())
        except Exception as e:raise PaymentError('INVALID_PAYMENT_SIGNATURE') from e
    def request(self,method,path,payload=None):
        body=json.dumps(payload,separators=(',',':'),ensure_ascii=False) if payload is not None else ''
        ts=str(int(time.time()));nonce=secrets.token_hex(16)
        sig=self.sign(f'{method}\n{path}\n{ts}\n{nonce}\n{body}\n')
        auth=f'WECHATPAY2-SHA256-RSA2048 mchid="{self.s.wechat_pay_mchid}",nonce_str="{nonce}",timestamp="{ts}",serial_no="{self.s.wechat_pay_serial_no}",signature="{sig}"'
        try:
            response=httpx.request(method,'https://api.mch.weixin.qq.com'+path,content=body.encode(),headers={'Authorization':auth,'Content-Type':'application/json','Accept':'application/json','Wechatpay-Serial':self.s.wechat_pay_platform_serial},timeout=12)
            self.verify(response.headers,response.content)
            if response.status_code>=400:
                data=response.json();raise PaymentError('WECHAT_'+str(data.get('code','ERROR')))
            return response.json() if response.content else {}
        except httpx.HTTPError as e:raise PaymentError('PAYMENT_RESULT_UNKNOWN') from e
    def prepay(self,order):
        return self.request('POST','/v3/pay/transactions/jsapi',{'appid':self.s.wechat_appid,'mchid':self.s.wechat_pay_mchid,
            'description':order.snapshot['name'],'out_trade_no':order.id,'notify_url':self.s.wechat_pay_notify_url,
            'amount':{'total':order.amount,'currency':'CNY'},'payer':{'openid':order.user_id},
            'time_expire':order.expires_at.isoformat()+'+00:00'})['prepay_id']
    def query(self,order):
        return self.request('GET',f'/v3/pay/transactions/out-trade-no/{order.id}?mchid={self.s.wechat_pay_mchid}')
    def close(self,order):
        return self.request('POST',f'/v3/pay/transactions/out-trade-no/{order.id}/close',{'mchid':self.s.wechat_pay_mchid})
    def payment_params(self,prepay_id):
        ts=str(int(time.time()));nonce=secrets.token_hex(16);package='prepay_id='+prepay_id
        return {'channel':'wechat_pay','params':{'timeStamp':ts,'nonceStr':nonce,'package':package,'signType':'RSA',
            'paySign':self.sign(f'{self.s.wechat_appid}\n{ts}\n{nonce}\n{package}\n')}}
    def notification(self,headers,body):
        self.verify(headers,body)
        _record_notification(headers,body)  # post-verify replay observation; never raises
        try:
            data=json.loads(body)
            if data['event_type']!='TRANSACTION.SUCCESS':raise ValueError()
            res=data['resource']
            if res['algorithm']!='AEAD_AES_256_GCM':raise ValueError()
            clear=AESGCM(self.s.wechat_pay_api_v3_key.encode()).decrypt(res['nonce'].encode(),base64.b64decode(res['ciphertext'],validate=True),res.get('associated_data','').encode())
            return json.loads(clear)
        except Exception as e:raise PaymentError('INVALID_PAYMENT_NOTIFICATION') from e

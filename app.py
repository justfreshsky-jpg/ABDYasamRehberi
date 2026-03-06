import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
from collections import deque, OrderedDict
import requests
import threading
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, make_response
from groq import Groq

os.environ.setdefault('HTTPX_PROXIES', 'null')

# ─── LOGGING ────────────────────────────────────────────────
_log_dir = os.environ.get('LOG_DIR', '')
_log_handlers = [logging.StreamHandler()]
if _log_dir:
    os.makedirs(_log_dir, exist_ok=True)
    _log_handlers.append(
        RotatingFileHandler(
            os.path.join(_log_dir, 'app.log'), maxBytes=5 * 1024 * 1024, backupCount=3
        )
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
GROQ_KEY = os.environ.get('GROQ_KEY')
_groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# ─── RESPONSE CACHE ──────────────────────────────────────
_resp_cache: OrderedDict = OrderedDict()
_CACHE_MAX = 500
_CACHE_TTL = 3600
_resp_lock = threading.Lock()


def _cache_get(key):
    with _resp_lock:
        if key in _resp_cache:
            val, ts = _resp_cache[key]
            if time.time() - ts < _CACHE_TTL:
                _resp_cache.move_to_end(key)
                return val
            del _resp_cache[key]
    return None


def _cache_set(key, val):
    with _resp_lock:
        if key in _resp_cache:
            _resp_cache[key] = (val, time.time())
            _resp_cache.move_to_end(key)
        else:
            if len(_resp_cache) >= _CACHE_MAX:
                _resp_cache.popitem(last=False)
            _resp_cache[key] = (val, time.time())


# ─── LLM PROVIDERS ───────────────────────────────────────
def _call_groq(system, user):
    if not GROQ_KEY:
        raise ValueError("GROQ_KEY not set")
    completion = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=1200,
        temperature=0.6,
        timeout=45,
    )
    return completion.choices[0].message.content.replace('**', '').strip()


def _call_cerebras(system, user):
    key = os.environ.get('CEREBRAS_KEY')
    if not key:
        raise ValueError("CEREBRAS_KEY not set")
    r = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "llama-3.3-70b", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 1200, "temperature": 0.6},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].replace('**', '').strip()


def _call_gemini(system, user):
    key = os.environ.get('GEMINI_KEY')
    if not key:
        raise ValueError("GEMINI_KEY not set")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": system + "\n\n" + user}]}], "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.6}},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].replace('**', '').strip()


def _call_cohere(system, user):
    key = os.environ.get('COHERE_KEY')
    if not key:
        raise ValueError("COHERE_KEY not set")
    r = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "command-r-plus", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 1200, "temperature": 0.6},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["message"]["content"][0]["text"].replace('**', '').strip()


def _call_mistral(system, user):
    key = os.environ.get('MISTRAL_KEY')
    if not key:
        raise ValueError("MISTRAL_KEY not set")
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "mistral-small-latest", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 1200, "temperature": 0.6},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].replace('**', '').strip()


def _call_openrouter(system, user):
    key = os.environ.get('OPENROUTER_KEY')
    if not key:
        raise ValueError("OPENROUTER_KEY not set")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "meta-llama/llama-3.3-70b-instruct:free", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 1200, "temperature": 0.6},
        timeout=45,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].replace('**', '').strip()


def _call_huggingface(system, user):
    key = os.environ.get('HF_KEY')
    if not key:
        raise ValueError("HF_KEY not set")
    r = requests.post(
        "https://router.hugging-face.cn/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": "mistralai/Mistral-7B-Instruct-v0.3", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 1200, "temperature": 0.6},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].replace('**', '').strip()


_PROVIDERS = [
    ("groq", _call_groq),
    ("cerebras", _call_cerebras),
    ("gemini", _call_gemini),
    ("cohere", _call_cohere),
    ("mistral", _call_mistral),
    ("openrouter", _call_openrouter),
    ("huggingface", _call_huggingface),
]


# ─── ARKA PLANDA BLOG İÇERİĞİ ─────────────────────────────
_cache = {"content": "", "last": 0}
_cache_lock = threading.Lock()

FALLBACK = """
[VERGİ] Rideshare vergi formları Ocak sonu yayınlanır. 1099-K, 1099-NEC gerekli.
W-4 doldururken exempt yazma, iade kaybedersin.
[VİZE] F-1 ile komşu ülkelere gidilebilir (Automatic Visa Revalidation).
J-1 vize başvurusu: DS-2019 al, SEVIS öde, konsolosluk randevusu.
[TELEFON] Lifeline programı ile ücretsiz hat alınabilir.
Google Voice ile SSN olmadan ABD numarası.
[SAĞLIK] NJ Medicaid gelir düşükse ücretsiz. NY'de free clinic'ler mevcut.
[BANKA] Chase ve BofA pasaportla hesap açılıyor. Secured card ile kredi skoru başlatılır.
[RİDESHARE] Uber/Lyft için SSN + ehliyet + araç sigorta şart. 1099 formunu Ocak'ta bekle.
[EV] NJ Newark/Paterson 1+1 $900-1200. Craigslist, Zillow, Facebook Marketplace dene.
[WİSE] Türkiye transferi limitlerde $50k/yıl. Wise > Western Union.
[EHLİYET] NJ'de 6 Points of ID sistemi. Undocumented bile ehliyet alabilir.
[UÇAK] THY NJ-İstanbul $400-700. Bagaj fazlasını uçuştan 24 saat önce öde ucuza.
"""

def _fetch_blog():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept-Language": "tr-TR,tr;q=0.9",
            "Referer": "https://www.google.com/"
        }
        urls = [
            "https://abdyasam.blogspot.com/",
            "https://abdyasam.blogspot.com/search?max-results=20"
        ]
        combined = ""
        for url in urls:
            r = requests.get(url, headers=headers, timeout=8)
            if not r.ok:
                logger.warning("Blog fetch returned status %s for %s", r.status_code, url)
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            posts = soup.find_all("div", class_=lambda c: c and "post" in c.lower())
            for p in posts[:15]:
                text = p.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    combined += text[:800] + "\n---\n"
        if combined:
            with _cache_lock:
                _cache["content"] = combined[:6000]
                _cache["last"] = time.time()
    except Exception:
        logger.exception("Blog fetch failed; keeping existing cache content")
        with _cache_lock:
            if not _cache["content"]:
                _cache["content"] = FALLBACK

def _bg_refresh():
    while True:
        _fetch_blog()
        time.sleep(3600)  # Her 1 saatte güncelle

_bg_started = False
_bg_start_lock = threading.Lock()

def get_context():
    with _cache_lock:
        content = _cache["content"]
    return content if content else FALLBACK

# ─── AI ─────────────────────────────────────────────
def llm(system, user):
    usa_prompt = """
🇺🇸 SADECE ABD İLE İLGİLİ KONULARA CEVAP VER
✅ ABD VİZE / SSN / BANKA / EV / UBER / VERGİ / SAĞLIK
• Her adıma emoji ekle: ✅ 🚀 💰 📱 🏠 🪪 ✈️ 🏥 💳
• ÖNEMLİ KELİMELERİ BÜYÜK YAZ
• Kısa paragraflar, uzun listeler
• ÇIKTI ŞABLONU:
  1) Hızlı Özet (3 madde)
  2) Adım Adım Kontrol Listesi
  3) Sık Yapılan Hatalar / Riskler
  4) Resmi Linkler (varsa)
  5) Sonraki Adım (tek net öneri)
⚠️ SADECE ABD / NJ / NY konuları!
"""

    full_system = system + "\n\n" + usa_prompt + "\n\nReferans veri:\n" + get_context()

    cache_key = hashlib.md5((system + '||' + user).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        return cached

    for name, fn in _PROVIDERS:
        try:
            text = fn(full_system, user)
            if text:
                text = ''.join(
                    c for c in text
                    if (ord(c) >= 0x20 or c in '\n\r\t') and not (0xD800 <= ord(c) <= 0xDFFF)
                )
                result = text.strip()
                _cache_set(cache_key, result)
                return result
        except Exception as e:
            logger.warning("Provider %s failed: %s", name, e)

    return "⚠️ Şu anda tüm AI servisleri geçici olarak kullanılamıyor. Lütfen birkaç dakika sonra tekrar deneyin."


class BadRequestError(Exception):
    """İstek gövdesi beklenen formatta olmadığında fırlatılır."""


_MAX_FIELD_LENGTH = 4000


def require_json(required_fields=None):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequestError("JSON body gerekli.")

    required_fields = required_fields or []
    missing = [field for field in required_fields if not str(data.get(field, '')).strip()]
    if missing:
        raise BadRequestError(f"Eksik alan(lar): {', '.join(missing)}")

    for key, value in data.items():
        if isinstance(value, str) and len(value) > _MAX_FIELD_LENGTH:
            raise BadRequestError(f"İstek alanı maksimum uzunluğu ({_MAX_FIELD_LENGTH} karakter) aşıyor.")

    return data


def llm_json(system_prompt, user_prompt):
    return jsonify(result=llm(system_prompt, user_prompt))


@app.errorhandler(BadRequestError)
def handle_bad_request(error):
    return jsonify(error=str(error)), 400


def _internal_error():
    logger.exception("Unhandled route error")
    return jsonify(error="İşlem sırasında bir hata oluştu."), 500


@app.errorhandler(Exception)
def handle_unexpected_error(_error):
    logger.exception("Unhandled exception")
    return jsonify(error="İşlem sırasında bir hata oluştu."), 500


# ─── RATE LIMITING ───────────────────────────────────────
_rate_counters: dict = {}
_rate_lock = threading.Lock()
_RATE_LIMIT = 20
_RATE_WINDOW = 60  # seconds


def _check_rate_limit() -> bool:
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or 'unknown'
    now = time.time()
    with _rate_lock:
        q = _rate_counters.get(ip)
        if q is not None:
            while q and now - q[0] > _RATE_WINDOW:
                q.popleft()
            if not q:
                del _rate_counters[ip]
                q = None
        if q is None:
            q = deque()
            _rate_counters[ip] = q
        if len(q) >= _RATE_LIMIT:
            return False
        q.append(now)
        return True


# ─── LIFECYCLE HOOKS ─────────────────────────────────────
@app.before_request
def _startup_hooks():
    global _bg_started
    if not _bg_started:
        with _bg_start_lock:
            if not _bg_started:
                threading.Thread(target=_bg_refresh, daemon=True).start()
                _bg_started = True

    if request.method == 'POST':
        # CSRF: validate Origin/Referer for browser-originated requests
        origin = request.headers.get('Origin', '')
        referer = request.headers.get('Referer', '')
        if origin or referer:
            check = origin or referer
            try:
                incoming_host = urlparse(check).hostname or ''
            except Exception:
                incoming_host = ''
            request_host = (request.host or '').split(':')[0]
            if incoming_host and incoming_host != request_host:
                return jsonify(error='İzin verilmeyen kaynak.'), 403

        if not _check_rate_limit():
            return jsonify(error='Çok fazla istek. Lütfen bir dakika sonra tekrar deneyin.'), 429


# ─── HTML ─────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>ABD Yaşam Rehberi</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%A7%AD%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Segoe UI,Arial,sans-serif;background:#f0f4ff;color:#1e293b}
.hero{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;padding:40px 20px;text-align:center}
.hero h1{font-size:2.2em;font-weight:800;margin-bottom:10px}
.hero p{font-size:1.1em;opacity:.9;max-width:600px;margin:0 auto 16px}
.steps{display:flex;justify-content:center;flex-wrap:wrap;gap:10px;margin-top:12px}
.step{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);border-radius:20px;padding:6px 16px;font-size:.9em}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;max-width:1000px;margin:30px auto;padding:0 20px}
.feat{background:#fff;border-radius:14px;padding:20px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.08);border-top:4px solid #3b82f6;transition:transform .3s}
.feat:hover{transform:translateY(-6px)}
.feat i{font-size:2em;color:#1e40af;margin-bottom:8px}
.feat h3{font-size:1em;margin-bottom:4px}
.feat p{font-size:.82em;color:#64748b}
.container{max-width:900px;margin:0 auto;padding:20px}
.tabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(72px,1fr));gap:8px;margin:24px 0}
.tabs button{background:#fff;border:2px solid #e2e8f0;padding:12px 8px;border-radius:12px;cursor:pointer;font-size:12px;font-weight:600;color:#1e293b;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:4px;min-height:44px}
.tabs button i{font-size:1.4em;color:#3b82f6}
.tabs button.active{background:#1e3a8a;color:#fff;border-color:#3b82f6}
.tabs button.active i{color:#fff}
.tabs button:hover:not(.active){background:#f0f4ff;transform:translateY(-2px)}
.tab{display:none}
.tab.active{display:block;animation:fadeIn .4s}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card{background:#fff;border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
.card h2{color:#1e3a8a;font-size:1.5em;margin-bottom:12px;display:flex;align-items:center;gap:10px}
.hint{background:#f0f9ff;border-left:4px solid #3b82f6;padding:14px 16px;border-radius:0 10px 10px 0;margin-bottom:20px;font-size:.9em;color:#0f4c75}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:500px){.form-row{grid-template-columns:1fr}}
.field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
label{font-weight:600;font-size:.9em;color:#334155}
input,select,textarea{padding:12px 14px;border:2px solid #e2e8f0;border-radius:10px;font-size:15px;transition:border .2s;background:#fafbfc;width:100%}
input:focus,select:focus,textarea:focus{border-color:#3b82f6;outline:none;background:#fff}
textarea{resize:vertical;min-height:90px}
.btn{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;border:none;padding:14px;width:100%;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;margin:16px 0 8px;box-shadow:0 4px 15px rgba(30,64,175,.3);transition:all .2s;position:relative;z-index:1}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(30,64,175,.4)}
.btn:disabled{opacity:.65;cursor:not-allowed;transform:none}
.output-wrap{position:relative;margin-top:8px}
.output{background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:20px;min-height:100px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.copy-btn{position:absolute;top:10px;right:10px;background:#10b981;color:#fff;border:none;border-radius:8px;padding:6px 14px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.output.loading{color:#94a3b8;font-style:italic}
.output.error{color:#ef4444}
.btn i,.tabs button i{pointer-events:none}
.trust-row{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 10px}.trust-chip{background:#fff;border:1px solid #dbeafe;color:#1e3a8a;padding:8px 12px;border-radius:999px;font-size:.82em;font-weight:600}.goal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:8px 0 18px}.goal-card{background:#fff;border:2px solid #e2e8f0;border-radius:12px;padding:12px;cursor:pointer;transition:.2s}.goal-card:hover{border-color:#3b82f6;transform:translateY(-2px)}.goal-card h4{font-size:.95em;color:#1e3a8a;margin-bottom:4px}.goal-card p{font-size:.8em;color:#64748b}.hero-desc{font-size:1em;opacity:.85;max-width:620px;margin:0 auto 18px;line-height:1.6}.footer{text-align:center;padding:32px 20px;color:#64748b;font-size:.88em;line-height:2;background:#fff;margin-top:20px;border-radius:16px}
</style>
</head>
<body>
<div class="hero">
  <h1 id="siteTitle" style="cursor:pointer">🇺🇸 ABD Yaşam Rehberi</h1>
  <p class="hero-desc">Amerika’ya yeni gelenler için yapay zekâ destekli rehber — vize, SSN, vergi, ev, sağlık ve daha fazlası hakkında adım adım destek. <br>Hiçbir kişisel veri saklanmaz.</p>
  <div class="steps">
    <span class="step">1️⃣ Konu Seç</span>
    <span class="step">2️⃣ Bilgini Gir</span>
    <span class="step">3️⃣ Kontrol Listeni Al</span>
  </div>
</div>
<div class="container">
  <div class="tabs" id="topicTabs">
    <button class="active" role="tab" aria-selected="true" data-tab="vize"><i class="fas fa-passport"></i>Vize</button>
    <button role="tab" aria-selected="false" data-tab="vergi"><i class="fas fa-calculator"></i>Vergi</button>
    <button role="tab" aria-selected="false" data-tab="rideshare"><i class="fas fa-car"></i>İş (Rideshare)</button>
    <button role="tab" aria-selected="false" data-tab="ev"><i class="fas fa-home"></i>Ev Bulma</button>
    <button role="tab" aria-selected="false" data-tab="saglik"><i class="fas fa-heartbeat"></i>Sağlık</button>
    <button role="tab" aria-selected="false" data-tab="ehliyet"><i class="fas fa-id-card"></i>Ehliyet</button>
    <button role="tab" aria-selected="false" data-tab="ssn"><i class="fas fa-id-card-alt"></i>SSN</button>
    <button role="tab" aria-selected="false" data-tab="banka"><i class="fas fa-university"></i>Banka</button>
    <button role="tab" aria-selected="false" data-tab="telefon"><i class="fas fa-phone"></i>Telefon</button>
    <button role="tab" aria-selected="false" data-tab="arac"><i class="fas fa-car-side"></i>Araç Bulma</button>
    <button role="tab" aria-selected="false" data-tab="wise"><i class="fas fa-exchange-alt"></i>Para Transferi</button>
    <button role="tab" aria-selected="false" data-tab="ucak"><i class="fas fa-plane"></i>Uçak</button>
    <button role="tab" aria-selected="false" data-tab="sorgu"><i class="fas fa-question-circle"></i>Soru Sor</button>
    <button role="tab" aria-selected="false" data-tab="feedback"><i class="fas fa-comment-dots"></i>Geri Bildirim</button>
  </div>

  <div id="vize" class="tab active"><div class="card">
    <h2><i class="fas fa-passport"></i> Vize & Green Card</h2>
    <div class="hint">🎯 <strong>Yeni gelen için:</strong> J-1 ile başla, iş bulunca H1B'e geç.</div>
    <div class="form-row">
      <div class="field"><label for="v1">Vize Tipi</label><select id="v1"><option>J-1 Öğrenci</option><option>H-1B İş</option><option>E-2 Yatırım</option><option>Yeşil Kart (EB)</option><option>F-1 Öğrenci</option><option>Ziyaretçi B-2</option></select></div>
      <div class="field"><label for="v2">Eyalet</label><select id="v2"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select></div>
    </div>
    <div class="field"><label for="v3">Özel Durum</label><input id="v3" maxlength="2000" placeholder="örn. İlk başvuru, uzatma, reddedildim"></div>
    <button class="btn" id="vb" onclick="call('/vize',{tip:g('v1'),state:g('v2'),durum:g('v3')},'vo','vb','Kişisel Vize Planı Oluştur')">Kişisel Vize Planı Oluştur</button>
    <div class="output-wrap"><div id="vo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('vo')">Kopyala</button></div>
  </div></div>

  <div id="vergi" class="tab"><div class="card">
    <h2><i class="fas fa-calculator"></i> Vergi İadesi & Formlar</h2>
    <div class="hint">💰 <strong>İpucu:</strong> İlk yıl 1040NR doldur. Rideshare varsa 1099 da ekle.</div>
    <div class="form-row">
      <div class="field"><label for="t1">Form Tipi</label><select id="t1"><option>W-4 (Bordro)</option><option>1040NR (Uluslararası)</option><option>1099-K (Rideshare)</option><option>W-2 (Çalışan)</option></select></div>
      <div class="field"><label for="t2">Yıllık Kazanç ($)</label><input id="t2" type="number" min="0" placeholder="örn. 35000"></div>
    </div>
    <div class="form-row">
      <div class="field"><label for="t3">Vize Tipin</label><select id="t3"><option>F-1 / J-1</option><option>H-1B</option><option>Green Card</option><option>Vatandaş</option></select></div>
      <div class="field"><label for="t4">Eyalet</label><select id="t4"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select></div>
    </div>
    <button class="btn" id="tb" onclick="call('/vergi',{form:g('t1'),kazanc:g('t2'),vize:g('t3'),state:g('t4')},'to','tb','Vergi Kontrol Listesi Oluştur')">Vergi Kontrol Listesi Oluştur</button>
    <div class="output-wrap"><div id="to" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('to')">Kopyala</button></div>
  </div></div>

  <div id="rideshare" class="tab"><div class="card">
    <h2><i class="fas fa-car"></i> Uber / Lyft ile Para Kazan</h2>
    <div class="hint">🚗 <strong>Yeni gelen için:</strong> Ehliyet + araba + SSN yeter. Haftada $800-1500.</div>
    <div class="form-row">
      <div class="field"><label for="r1">Uygulama</label><select id="r1"><option>Uber</option><option>Lyft</option><option>Her İkisi</option></select></div>
      <div class="field"><label for="r2">Eyalet</label><select id="r2"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select></div>
    </div>
    <div class="field"><label for="r3">Konu</label><select id="r3"><option>Nasıl başlarım?</option><option>1099 formu / vergi</option><option>Haftada ne kadar kazanırım?</option><option>Masraf düşümü (deduction)</option></select></div>
    <button class="btn" id="rb" onclick="call('/rideshare',{app:g('r1'),state:g('r2'),konu:g('r3')},'ro','rb','Rideshare Rehberi')">Planı Oluştur</button>
    <div class="output-wrap"><div id="ro" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('ro')">Kopyala</button></div>
  </div></div>

  <div id="ev" class="tab"><div class="card">
    <h2><i class="fas fa-home"></i> Ev / Daire Kiralama</h2>
    <div class="hint">🏠 <strong>İpucu:</strong> NJ Newark/Paterson 1+1 $900-1200. Craigslist ve Zillow dene.</div>
    <div class="form-row">
      <div class="field"><label for="e1">Şehir / Bölge</label><input id="e1" maxlength="2000" placeholder="örn. Newark NJ, Jersey City"></div>
      <div class="field"><label for="e2">Bütçe ($/ay)</label><input id="e2" type="number" min="0" placeholder="1200"></div>
    </div>
    <div class="field"><label for="e3">Özel Durum</label><input id="e3" maxlength="2000" placeholder="örn. SSN yok, kredi skoru yok, evcil hayvan var"></div>
    <button class="btn" id="eb" onclick="call('/ev',{sehir:g('e1'),butce:g('e2'),durum:g('e3')},'eo','eb','Ev Bulma Rehberi')">Planı Oluştur</button>
    <div class="output-wrap"><div id="eo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('eo')">Kopyala</button></div>
  </div></div>

  <div id="saglik" class="tab"><div class="card">
    <h2><i class="fas fa-heartbeat"></i> Ücretsiz Sağlık Sigortası</h2>
    <div class="hint">🏥 <strong>İpucu:</strong> NJ'de düşük gelirle Medicaid ücretsiz. Bazı kliniklerde belge gerekmez.</div>
    <div class="form-row">
      <div class="field"><label for="h1">Eyalet</label><select id="h1"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select></div>
      <div class="field"><label for="h2">Durum</label><select id="h2"><option>Sigorta yok, nasıl alırım?</option><option>Medicaid nasıl başvururum?</option><option>Ücretsiz klinik nerede?</option><option>SSN olmadan sigorta olur mu?</option></select></div>
    </div>
    <button class="btn" id="hb" onclick="call('/saglik',{state:g('h1'),durum:g('h2')},'ho','hb','Sağlık Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="ho" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('ho')">Kopyala</button></div>
  </div></div>

  <div id="ehliyet" class="tab"><div class="card">
    <h2><i class="fas fa-id-card"></i> Ehliyet Alma (DMV)</h2>
    <div class="hint">🪪 <strong>İpucu:</strong> NJ'de 6 Points of ID sistemi. Undocumented bile ehliyet alabiliyor.</div>
    <div class="form-row">
      <div class="field"><label for="l1">Eyalet</label><select id="l1"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select></div>
      <div class="field"><label for="l2">Durum</label><select id="l2"><option>İlk kez alıyorum</option><option>Türk ehliyetimi çevirmek istiyorum</option><option>SSN / ITIN yok</option><option>Real ID lazım</option></select></div>
    </div>
    <button class="btn" id="lb" onclick="call('/ehliyet',{state:g('l1'),durum:g('l2')},'lo','lb','Ehliyet Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="lo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('lo')">Kopyala</button></div>
  </div></div>

<div id="ssn" class="tab">
  <div class="card">
    <h2><i class="fas fa-id-card-alt"></i> SSN Alma Rehberi</h2>
    <div class="hint">🆔 <strong>Yeni gelen için:</strong> F-1/J-1 öğrenciysen CPT/OPT ile alabilirsin. On-campus iş için SSN şart.</div>
    <div class="form-row">
      <div class="field">
        <label for="ss1">Vize Tipi</label>
        <select id="ss1">
          <option>F-1 Öğrenci (CPT/OPT)</option>
          <option>J-1 Öğrenci</option>
          <option>H-1B İş Vizesi</option>
          <option>Green Card Bekleyen</option>
          <option>On-campus iş</option>
          <option>SSN yok, nasıl alırım?</option>
        </select>
      </div>
      <div class="field">
        <label for="ss2">Eyalet</label>
        <select id="ss2"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select>
      </div>
    </div>
    <div class="field">
      <label for="ss3">Durum</label>
      <input id="ss3" maxlength="2000" placeholder="örn. CPT onayım var, OPT bekliyorum, ITIN var mı?">
    </div>
    <button class="btn" id="ssb" onclick="call('/ssn',{vize:g('ss1'),state:g('ss2'),durum:g('ss3')},'sso','ssb','SSN Rehberi Oluştur')">SSN Rehberi Oluştur</button>
    <div class="output-wrap">
      <div id="sso" class="output">Sonuç burada görünecek...</div>
      <button class="copy-btn" onclick="cp('sso')">Kopyala</button>
    </div>
  </div>
</div>

  <div id="banka" class="tab"><div class="card">
    <h2><i class="fas fa-university"></i> Banka Hesabı Açma</h2>
    <div class="hint">💳 <strong>İpucu:</strong> Chase/BofA pasaportla açılıyor. Secured card ile kredi skoru başlatılır.</div>
    <div class="field"><label for="ba1">Durum</label><select id="ba1"><option>SSN olmadan banka açmak istiyorum</option><option>Kredi kartı almak istiyorum</option><option>Credit score sıfırdan nasıl yaparım?</option><option>En iyi ücretsiz banka hangisi?</option></select></div>
    <button class="btn" id="bb" onclick="call('/banka',{durum:g('ba1')},'bo','bb','Banka Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="bo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('bo')">Kopyala</button></div>
  </div></div>

  <div id="telefon" class="tab"><div class="card">
    <h2><i class="fas fa-phone"></i> ABD Telefon Numarası</h2>
    <div class="hint">📱 <strong>İpucu:</strong> Google Voice ile SSN olmadan ücretsiz Amerikan numarası alabilirsin.</div>
    <div class="field"><label for="p1">Konu</label><select id="p1"><option>Ücretsiz numara (Google Voice)</option><option>Ucuz hat (Mint, Visible, T-Mobile)</option><option>SSN olmadan kontrat hat</option><option>Türkiye'yi ucuz arama</option></select></div>
    <button class="btn" id="pb" onclick="call('/telefon',{konu:g('p1')},'po','pb','Telefon Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="po" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('po')">Kopyala</button></div>
  </div></div>

  <div id="arac" class="tab"><div class="card">
    <h2><i class="fas fa-car-side"></i> Araç Kiralama / Satın Alma</h2>
    <div class="hint">🚗 <strong>İpucu:</strong> SSN olmadan araç satın alınabiliyor. CarMax/Carvana ile başla.</div>
    <div class="form-row">
      <div class="field"><label for="ar1">Eyalet</label><select id="ar1"><option value="Alabama">Alabama</option><option value="Alaska">Alaska</option><option value="Arizona">Arizona</option><option value="Arkansas">Arkansas</option><option value="California">California</option><option value="Colorado">Colorado</option><option value="Connecticut">Connecticut</option><option value="Delaware">Delaware</option><option value="Florida">Florida</option><option value="Georgia">Georgia</option><option value="Hawaii">Hawaii</option><option value="Idaho">Idaho</option><option value="Illinois">Illinois</option><option value="Indiana">Indiana</option><option value="Iowa">Iowa</option><option value="Kansas">Kansas</option><option value="Kentucky">Kentucky</option><option value="Louisiana">Louisiana</option><option value="Maine">Maine</option><option value="Maryland">Maryland</option><option value="Massachusetts">Massachusetts</option><option value="Michigan">Michigan</option><option value="Minnesota">Minnesota</option><option value="Mississippi">Mississippi</option><option value="Missouri">Missouri</option><option value="Montana">Montana</option><option value="Nebraska">Nebraska</option><option value="Nevada">Nevada</option><option value="New Hampshire">New Hampshire</option><option value="New Jersey" selected>New Jersey</option><option value="New Mexico">New Mexico</option><option value="New York">New York</option><option value="North Carolina">North Carolina</option><option value="North Dakota">North Dakota</option><option value="Ohio">Ohio</option><option value="Oklahoma">Oklahoma</option><option value="Oregon">Oregon</option><option value="Pennsylvania">Pennsylvania</option><option value="Rhode Island">Rhode Island</option><option value="South Carolina">South Carolina</option><option value="South Dakota">South Dakota</option><option value="Tennessee">Tennessee</option><option value="Texas">Texas</option><option value="Utah">Utah</option><option value="Vermont">Vermont</option><option value="Virginia">Virginia</option><option value="Washington">Washington</option><option value="West Virginia">West Virginia</option><option value="Wisconsin">Wisconsin</option><option value="Wyoming">Wyoming</option></select></div>
      <div class="field"><label for="ar2">Konu</label><select id="ar2"><option>İkinci el araç almak istiyorum</option><option>Araç kiralamak istiyorum</option><option>Araç sigortası almak istiyorum</option><option>SSN olmadan araç alınır mı?</option></select></div>
    </div>
    <button class="btn" id="arb" onclick="call('/arac',{state:g('ar1'),konu:g('ar2')},'aro','arb','Araç Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="aro" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('aro')">Kopyala</button></div>
  </div></div>

  <div id="wise" class="tab"><div class="card">
    <h2><i class="fas fa-exchange-alt"></i> Para Transfer (Wise / Zelle)</h2>
    <div class="hint">💸 <strong>İpucu:</strong> Wise ile TL/$ kurunu en düşük komisyonla gönder.</div>
    <div class="field"><label for="w1">Konu</label><select id="w1"><option>Wise ile Türkiye'ye para gönderme</option><option>Wise limitleri ve ücretleri</option><option>Zelle nasıl kullanılır?</option><option>Venmo / CashApp rehberi</option></select></div>
    <button class="btn" id="wb" onclick="call('/wise',{konu:g('w1')},'wo','wb','Para Transfer Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="wo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('wo')">Kopyala</button></div>
  </div></div>

  <div id="ucak" class="tab"><div class="card">
    <h2><i class="fas fa-plane"></i> Uçak & Bagaj</h2>
    <div class="hint">✈️ <strong>İpucu:</strong> THY NJ-İstanbul $400-700. Bagaj fazlasını 24 saat önce öde ucuza.</div>
    <div class="form-row">
      <div class="field"><label for="u1">Havayolu</label><select id="u1"><option>Turkish Airlines</option><option>American Airlines</option><option>United</option><option>Delta</option></select></div>
      <div class="field"><label for="u2">Konu</label><select id="u2"><option>Bagaj ücretleri ve kurallar</option><option>En ucuz bilet nasıl bulunur?</option><option>Check-in rehberi</option><option>Refund / iptal kuralları</option></select></div>
    </div>
    <button class="btn" id="ub" onclick="call('/ucak',{havayolu:g('u1'),konu:g('u2')},'uo','ub','Uçak Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="uo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('uo')">Kopyala</button></div>
  </div></div>

  <div id="sorgu" class="tab"><div class="card">
    <h2><i class="fas fa-question-circle"></i> Herhangi Bir Soru Sor</h2>
    <div class="hint">🤖 ABD hayatıyla ilgili aklına takılan her şeyi sor. Türkçe cevap gelir.</div>
    <div class="field"><label for="q1">Sorun nedir?</label><textarea id="q1" rows="4" maxlength="2000" placeholder="örn. SSN olmadan iş bulabilir miyim? İlk ay ne yapmalıyım?"></textarea></div>
    <button class="btn" id="qb" onclick="call('/sorgu',{soru:g('q1')},'qo','qb','Cevapla')">Cevapla</button>
    <div class="output-wrap"><div id="qo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="cp('qo')">Kopyala</button></div>
  </div></div>


  <div id="feedback" class="tab"><div class="card">
    <h2><i class="fas fa-comment-dots"></i> Geri Bildirim</h2>
    <p>Görüş ve önerileriniz için: <a href="mailto:admin@abdyasamrehberi.com">admin@abdyasamrehberi.com</a></p>
</div></div>

</div>
<div class="footer">
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ Bu araç yalnızca bilgilendirme amaçlıdır. Yapay zeka hata yapabilir.
    Hukuki, mali veya tıbbi konularda mutlaka uzman görüşü alınız.
    Yapay zeka çıktısına dayanarak alınan kararlardan sorumluluk kabul edilmez.
  </span>
</div>
<script>
function g(id){return document.getElementById(id).value;}
function show(tab,btn){
  const target=document.getElementById(tab);
  if(!target) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>{
    b.classList.remove('active');
    b.setAttribute('aria-selected','false');
  });
  target.classList.add('active');
  if(btn){
    btn.classList.add('active');
    btn.setAttribute('aria-selected','true');
  }
}
function quickStart(tab){
  const target=document.getElementById(tab);
  if(!target) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>{
    b.classList.remove('active');
    b.setAttribute('aria-selected','false');
  });
  target.classList.add('active');
  const match=[...document.querySelectorAll('.tabs button[data-tab]')].find(b=>b.dataset.tab===tab);
  if(match){match.classList.add('active');match.setAttribute('aria-selected','true');}
  const firstInput=target.querySelector('input,select,textarea');
  if(firstInput) firstInput.focus({preventScroll:true});
  target.scrollIntoView({behavior:'smooth',block:'start'});
}
function cp(id){
  const el=document.getElementById(id);
  const txt=el.innerText;
  const btn=el.parentNode.querySelector('.copy-btn');
  function _ok(){btn.textContent='Kopyalandı!';setTimeout(()=>btn.textContent='Kopyala',2000);}
  if(navigator.clipboard && window.isSecureContext !== false){
    navigator.clipboard.writeText(txt).then(_ok).catch(()=>_fallbackCopy(txt,_ok));
  }else{
    _fallbackCopy(txt,_ok);
  }
}
function _fallbackCopy(txt,cb){
  const ta=document.createElement('textarea');
  ta.value=txt;ta.style.position='fixed';ta.style.opacity='0';
  document.body.appendChild(ta);ta.select();
  try{document.execCommand('copy');cb();}catch(e){}
  document.body.removeChild(ta);
}
function formatResult(text){
  function escape(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
  const paragraphs=text.split(/\n{2,}/);
  return paragraphs.map(para=>{
    const lines=para.split('\n').filter(l=>l.trim());
    if(!lines.length) return '';
    const isOrdered=lines.every(l=>/^\d+\.\s/.test(l.trim()));
    const isBullet=lines.every(l=>/^[-•]\s/.test(l.trim()));
    if(isOrdered){
      const items=lines.map(l=>'<li>'+escape(l.trim().replace(/^\d+\.\s/,''))+'</li>').join('');
      return '<ol>'+items+'</ol>';
    }
    if(isBullet){
      const items=lines.map(l=>'<li>'+escape(l.trim().replace(/^[-•]\s/,''))+'</li>').join('');
      return '<ul>'+items+'</ul>';
    }
    return lines.map(l=>{
      const t=l.trim();
      if(!t) return '';
      if(/^[A-ZÇĞİÖŞÜ][^a-zçğışöüa-z]{3,}$/.test(t)||(/[:\uFF1A]$/.test(t)&&t.length<80)){
        return '<p><strong>'+escape(t)+'</strong></p>';
      }
      return '<p>'+escape(t)+'</p>';
    }).join('');
  }).join('');
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId);
  const btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class=\"spinner\"></span>Yükleniyor...';
  out.classList.add('loading');
  out.textContent='Rehberiniz oluşturuluyor...';
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json().catch(()=>({}));
    if(!r.ok){
      out.classList.remove('loading');
      out.classList.add('error');
      out.textContent='Hata: '+(j.error || 'İstek işlenemedi.');
      return;
    }
    out.classList.remove('loading');
    out.innerHTML=formatResult(j.result || 'Sonuç üretilemedi.');
    out.scrollIntoView({behavior:'smooth',block:'nearest'});
  }catch(e){
    out.classList.remove('loading');
    out.classList.add('error');
    out.textContent='Bağlantı hatası: '+e.message;
  }finally{
    btn.disabled=false;
    btn.textContent=label;
  }
}

document.addEventListener('DOMContentLoaded',function(){
  document.querySelectorAll('.tabs button[data-tab]').forEach(function(btn){
    btn.addEventListener('click',function(){show(btn.dataset.tab,btn);});
  });
  document.addEventListener('click',function(e){
    const qs=e.target.closest('[data-quickstart]');
    if(qs) quickStart(qs.dataset.quickstart);
  });
  const h1=document.getElementById('siteTitle');
  if(h1) h1.addEventListener('click',function(){
    document.querySelectorAll('input,textarea').forEach(function(el){el.value='';});
    document.querySelectorAll('select').forEach(function(el){el.selectedIndex=0;});
    window.location.reload();
  });
});
</script>
</body>
</html>"""

# ─── ROUTES ──────────────────────────────────────────
@app.route('/')
def index():
    response = make_response(HTML)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response


@app.route('/healthz')
def healthz():
    return jsonify(
        status='ok',
        groq=bool(GROQ_KEY),
        cerebras=bool(os.environ.get('CEREBRAS_KEY')),
        gemini=bool(os.environ.get('GEMINI_KEY')),
        cohere=bool(os.environ.get('COHERE_KEY')),
        mistral=bool(os.environ.get('MISTRAL_KEY')),
        openrouter=bool(os.environ.get('OPENROUTER_KEY')),
        huggingface=bool(os.environ.get('HF_KEY')),
    )

@app.route('/vize', methods=['POST'])
def do_vize():
    try:
        d = require_json(["tip"])
        return llm_json(
            "ABD göçmenlik uzmanısın, Türkçe pratik rehber ver.",
            f"{d['tip']} vizesi. State: {d.get('state','')}. Durum: {d.get('durum','')}. Belgeler, formlar, ücretler, hatalar, linkler."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/vergi', methods=['POST'])
def do_vergi():
    try:
        d = require_json(["form"])
        return llm_json(
            "ABD vergi uzmanısın, Türkçe sade anlat.",
            f"Form: {d['form']}. Kazanç: ${d.get('kazanc',0)}. Vize: {d.get('vize','')}. State: {d.get('state','')}. Doldurma rehberi, iade tahmini, deadline'lar."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/rideshare', methods=['POST'])
def do_rideshare():
    try:
        d = require_json(["app"])
        return llm_json(
            "Rideshare ve gig economy uzmanısın, Türkçe yaz.",
            f"{d['app']} - {d.get('state','')}. Konu: {d.get('konu','')}. Belgeler, kazanç, vergi, ipuçları."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/ev', methods=['POST'])
def do_ev():
    try:
        d = require_json()
        return llm_json(
            "ABD emlak uzmanısın, Türkçe yaz.",
            f"{d.get('sehir','')} ${d.get('butce','')} bütçe. Durum: {d.get('durum','')}. Siteler, belgeler, müzakere tüyoları."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/saglik', methods=['POST'])
def do_saglik():
    try:
        d = require_json()
        return llm_json(
            "ABD sağlık sistemi uzmanısın, Türkçe pratik yaz.",
            f"{d.get('state','')} - {d.get('durum','')}. Adresler, belgeler, Medicaid, ücretsiz klinikler."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/ehliyet', methods=['POST'])
def do_ehliyet():
    try:
        d = require_json()
        return llm_json(
            "ABD DMV uzmanısın, Türkçe anlat.",
            f"{d.get('state','')} ehliyet: {d.get('durum','')}. 6 Points belgeler, sınav, randevu, ücretler."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/ssn', methods=['POST'])
def do_ssn():
    try:
        d = require_json(["vize"])
        return llm_json(
            "ABD SSN uzmanısın. Türk göçmenler için Türkçe pratik rehber ver. NJ odaklı.",
            f"Vize: {d['vize']}. State: {d.get('state','NJ')}. Durum: {d.get('durum','')}. "
            "SSN için gerekli belgeler, başvuru adımları, NJ SSA ofis adresleri, "
            "F-1/J-1 için CPT/OPT şartı, ITIN alternatifi, sık hatalar."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/banka', methods=['POST'])
def do_banka():
    try:
        d = require_json()
        return llm_json(
            "ABD bankacılık uzmanısın, Türkçe yaz.",
            f"Konu: {d.get('durum','')}. Hangi banka, belgeler, credit score, secured card."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/telefon', methods=['POST'])
def do_telefon():
    try:
        d = require_json()
        return llm_json(
            "ABD telekomünikasyon uzmanısın, Türkçe rehber.",
            f"Konu: {d.get('konu','')}. Adım adım kurulum, fiyatlar, alternatifler."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/arac', methods=['POST'])
def do_arac():
    try:
        d = require_json()
        return llm_json(
            "ABD otomotiv uzmanısın, Türkçe yaz.",
            f"{d.get('state','')} - {d.get('konu','')}. Belgeler, sigorta, fiyat, CarMax/Carvana."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/wise', methods=['POST'])
def do_wise():
    try:
        d = require_json()
        return llm_json(
            "Para transferi uzmanısın, Türkçe anlat.",
            f"Konu: {d.get('konu','')}. Adımlar, komisyonlar, limitler, alternatifler."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/ucak', methods=['POST'])
def do_ucak():
    try:
        d = require_json()
        return llm_json(
            "Havacılık uzmanısın, Türkçe pratik rehber.",
            f"{d.get('havayolu','')} - {d.get('konu','')}. Detaylı bilgi, ücretler, ipuçları."
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

@app.route('/sorgu', methods=['POST'])
def do_sorgu():
    try:
        d = require_json(['soru'])
        return llm_json(
            "ABD'de yaşayan Türkler için pratik rehber uzmanısın. Cevapları sade, adım adım ve güvenli şekilde ver.",
            d.get('soru', '')
        )
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()


_feedback_store = deque(maxlen=500)


@app.route('/feedback', methods=['POST'])
def do_feedback():
    try:
        d = require_json(['mesaj'])
        _feedback_store.append({
            'mesaj': d.get('mesaj', '').strip(),
            'iletisim': d.get('iletisim', '').strip(),
            'ts': int(time.time())
        })
        return jsonify(result='Teşekkürler! Geri bildirimin alındı ve iyileştirme listesine eklendi.', total_feedback=len(_feedback_store))
    except BadRequestError:
        raise
    except Exception:
        return _internal_error()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

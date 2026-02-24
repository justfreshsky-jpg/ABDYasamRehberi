import os
import traceback
import requests
import threading
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from groq import Groq
os.environ['HTTPX_PROXIES'] = 'null'

app = Flask(__name__)
GROQ_KEY = os.environ.get('GROQ_KEY')
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# ─── ARKA PLANDA BLOG İÇERİĞİ ─────────────────────────────
_cache = {"content": "", "last": 0}

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
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            posts = soup.find_all("div", class_=lambda c: c and "post" in c.lower())
            for p in posts[:15]:
                text = p.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    combined += text[:800] + "\n---\n"
        if combined:
            _cache["content"] = combined[:6000]
            _cache["last"] = time.time()
    except Exception:
        _cache["content"] = FALLBACK

def _bg_refresh():
    while True:
        _fetch_blog()
        time.sleep(3600)  # Her 1 saatte güncelle

threading.Thread(target=_bg_refresh, daemon=True).start()

def get_context():
    if not _cache["content"]:
        return FALLBACK
    return _cache["content"]

# ─── AI ─────────────────────────────────────────────
def llm(system, user):
    if not client:
        return "GROQ_KEY eksik. Render > Environment Variables'a ekle."
    full_system = system + "\n\nPratik kaynak bilgileri:\n" + get_context()
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": user}
        ],
        max_tokens=2000,
        temperature=0.7
    )
    return r.choices[0].message.content

# ─── HTML ─────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<title>ABD Yaşam Rehberi</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
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
.tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:24px 0}
@media(max-width:600px){.tabs{grid-template-columns:repeat(2,1fr)}}
.tabs button{background:#fff;border:2px solid #e2e8f0;padding:12px 8px;border-radius:12px;cursor:pointer;font-size:12px;font-weight:600;color:#1e293b;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:4px}
.tabs button i{font-size:1.4em;color:#3b82f6}
.tabs button.active{background:#1e3a8a;color:#fff;border-color:#3b82f6}
.tabs button.active i{color:#fff}
.tabs button:hover:not(.active){background:#f0f4ff;transform:translateY(-2px)}
.tab{display:none}
.tab.active{display:block;animation:fadeIn .4s}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card{background:#fff;border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
.card h2{color:#1e3a8a;font-size:1.5em;margin-bottom:12px;display:flex;align-items:center;gap:10px}
.hint{background:#f0f9ff;border-left:4px solid #10b981;padding:14px 16px;border-radius:0 10px 10px 0;margin-bottom:20px;font-size:.9em;color:#0f4c75}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:500px){.form-row{grid-template-columns:1fr}}
.field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
label{font-weight:600;font-size:.9em;color:#334155}
input,select,textarea{padding:12px 14px;border:2px solid #e2e8f0;border-radius:10px;font-size:15px;transition:border .2s;background:#fafbfc;width:100%}
input:focus,select:focus,textarea:focus{border-color:#3b82f6;outline:none;background:#fff}
textarea{resize:vertical;min-height:90px}
.btn{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;border:none;padding:14px;width:100%;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;margin:16px 0 8px;box-shadow:0 4px 15px rgba(30,64,175,.3);transition:all .2s}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(30,64,175,.4)}
.btn:disabled{opacity:.65;cursor:not-allowed;transform:none}
.output-wrap{position:relative;margin-top:8px}
.output{background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:20px;min-height:100px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.copy-btn{position:absolute;top:10px;right:10px;background:#10b981;color:#fff;border:none;border-radius:8px;padding:6px 14px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.footer{text-align:center;padding:32px 20px;color:#64748b;font-size:.88em;line-height:2;background:#fff;margin-top:20px;border-radius:16px}
</style>
</head>
<body>
<div class="hero">
  <h1>🇺🇸 ABD'ye Hoş Geldin!</h1>
  <p>Türkler için pratik AI rehberi — sıfırdan adım adım</p>
  <div class="steps">
    <span class="step">1️⃣ Vize Al</span>
    <span class="step">2️⃣ SSN Çıkar</span>
    <span class="step">3️⃣ Banka Aç</span>
    <span class="step">4️⃣ Ev Bul</span>
    <span class="step">5️⃣ Çalış / Para Kazan</span>
  </div>
</div>
<div class="features">
  <div class="feat"><i class="fas fa-passport"></i><h3>Vize & Green Card</h3><p>J-1, H1B, E-2</p></div>
  <div class="feat"><i class="fas fa-calculator"></i><h3>Vergi İadesi</h3><p>$500-2000 geri al</p></div>
  <div class="feat"><i class="fas fa-car"></i><h3>Rideshare</h3><p>Uber/Lyft başla</p></div>
  <div class="feat"><i class="fas fa-home"></i><h3>Ucuz Ev</h3><p>NJ $900 kiralık</p></div>
  <div class="feat"><i class="fas fa-heartbeat"></i><h3>Ücretsiz Sağlık</h3><p>Medicaid, free clinic</p></div>
  <div class="feat"><i class="fas fa-university"></i><h3>Banka Aç</h3><p>SSN olmadan</p></div>
</div>
<div class="container">
  <div class="tabs">
    <button class="active" onclick="show('vize',this)"><i class="fas fa-passport"></i>Vize</button>
    <button onclick="show('vergi',this)"><i class="fas fa-calculator"></i>Vergi</button>
    <button onclick="show('rideshare',this)"><i class="fas fa-car"></i>Rideshare</button>
    <button onclick="show('ev',this)"><i class="fas fa-home"></i>Ev</button>
    <button onclick="show('saglik',this)"><i class="fas fa-heartbeat"></i>Sağlık</button>
    <button onclick="show('ehliyet',this)"><i class="fas fa-id-card"></i>Ehliyet</button>
    <button onclick="show('banka',this)"><i class="fas fa-university"></i>Banka</button>
    <button onclick="show('telefon',this)"><i class="fas fa-phone"></i>Telefon</button>
    <button onclick="show('arac',this)"><i class="fas fa-car-side"></i>Araç</button>
    <button onclick="show('wise',this)"><i class="fas fa-exchange-alt"></i>Wise</button>
    <button onclick="show('ucak',this)"><i class="fas fa-plane"></i>Uçak</button>
    <button onclick="show('sorgu',this)"><i class="fas fa-question-circle"></i>Soru Sor</button>
  </div>

  <div id="vize" class="tab active"><div class="card">
    <h2><i class="fas fa-passport"></i> Vize & Green Card</h2>
    <div class="hint">🎯 <strong>Yeni gelen için:</strong> J-1 ile başla, iş bulunca H1B'e geç.</div>
    <div class="form-row">
      <div class="field"><label>Vize Tipi</label><select id="v1"><option>J-1 Öğrenci</option><option>H-1B İş</option><option>E-2 Yatırım</option><option>Yeşil Kart (EB)</option><option>F-1 Öğrenci</option><option>Ziyaretçi B-2</option></select></div>
      <div class="field"><label>State</label><input id="v2" placeholder="örn. New Jersey"></div>
    </div>
    <div class="field"><label>Özel Durum</label><input id="v3" placeholder="örn. İlk başvuru, uzatma, reddedildim"></div>
    <button class="btn" id="vb" onclick="call('/vize',{tip:g('v1'),state:g('v2'),durum:g('v3')},'vo','vb','Vize Rehberi Oluştur')">Vize Rehberi Oluştur</button>
    <div class="output-wrap"><div id="vo" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('vo')">Kopyala</button></div>
  </div></div>

  <div id="vergi" class="tab"><div class="card">
    <h2><i class="fas fa-calculator"></i> Vergi İadesi & Formlar</h2>
    <div class="hint">💰 <strong>İpucu:</strong> İlk yıl 1040NR doldur. Rideshare varsa 1099 da ekle.</div>
    <div class="form-row">
      <div class="field"><label>Form Tipi</label><select id="t1"><option>W-4 (Bordro)</option><option>1040NR (Uluslararası)</option><option>1099-K (Rideshare)</option><option>W-2 (Çalışan)</option></select></div>
      <div class="field"><label>Yıllık Kazanç ($)</label><input id="t2" type="number" placeholder="örn. 35000"></div>
    </div>
    <div class="form-row">
      <div class="field"><label>Vize Tipin</label><select id="t3"><option>F-1 / J-1</option><option>H-1B</option><option>Green Card</option><option>Vatandaş</option></select></div>
      <div class="field"><label>State</label><input id="t4" placeholder="New Jersey"></div>
    </div>
    <button class="btn" id="tb" onclick="call('/vergi',{form:g('t1'),kazanc:g('t2'),vize:g('t3'),state:g('t4')},'to','tb','Vergi Rehberi Oluştur')">Vergi Rehberi Oluştur</button>
    <div class="output-wrap"><div id="to" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('to')">Kopyala</button></div>
  </div></div>

  <div id="rideshare" class="tab"><div class="card">
    <h2><i class="fas fa-car"></i> Uber / Lyft ile Para Kazan</h2>
    <div class="hint">🚗 <strong>Yeni gelen için:</strong> Ehliyet + araba + SSN yeter. Haftada $800-1500.</div>
    <div class="form-row">
      <div class="field"><label>Uygulama</label><select id="r1"><option>Uber</option><option>Lyft</option><option>Her İkisi</option></select></div>
      <div class="field"><label>State</label><input id="r2" placeholder="New Jersey"></div>
    </div>
    <div class="field"><label>Konu</label><select id="r3"><option>Nasıl başlarım?</option><option>1099 formu / vergi</option><option>Haftada ne kadar kazanırım?</option><option>Masraf düşümü (deduction)</option></select></div>
    <button class="btn" id="rb" onclick="call('/rideshare',{app:g('r1'),state:g('r2'),konu:g('r3')},'ro','rb','Rideshare Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="ro" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('ro')">Kopyala</button></div>
  </div></div>

  <div id="ev" class="tab"><div class="card">
    <h2><i class="fas fa-home"></i> Ev / Daire Kiralama</h2>
    <div class="hint">🏠 <strong>İpucu:</strong> NJ Newark/Paterson 1+1 $900-1200. Craigslist ve Zillow dene.</div>
    <div class="form-row">
      <div class="field"><label>Şehir / Bölge</label><input id="e1" placeholder="örn. Newark NJ, Jersey City"></div>
      <div class="field"><label>Bütçe ($/ay)</label><input id="e2" type="number" placeholder="1200"></div>
    </div>
    <div class="field"><label>Özel Durum</label><input id="e3" placeholder="örn. SSN yok, kredi skoru yok, evcil hayvan var"></div>
    <button class="btn" id="eb" onclick="call('/ev',{sehir:g('e1'),butce:g('e2'),durum:g('e3')},'eo','eb','Ev Bulma Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="eo" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('eo')">Kopyala</button></div>
  </div></div>

  <div id="saglik" class="tab"><div class="card">
    <h2><i class="fas fa-heartbeat"></i> Ücretsiz Sağlık Sigortası</h2>
    <div class="hint">🏥 <strong>İpucu:</strong> NJ'de düşük gelirle Medicaid ücretsiz. Bazı kliniklerde belge gerekmez.</div>
    <div class="form-row">
      <div class="field"><label>State</label><input id="h1" placeholder="New Jersey"></div>
      <div class="field"><label>Durum</label><select id="h2"><option>Sigorta yok, nasıl alırım?</option><option>Medicaid nasıl başvururum?</option><option>Ücretsiz klinik nerede?</option><option>SSN olmadan sigorta olur mu?</option></select></div>
    </div>
    <button class="btn" id="hb" onclick="call('/saglik',{state:g('h1'),durum:g('h2')},'ho','hb','Sağlık Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="ho" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('ho')">Kopyala</button></div>
  </div></div>

  <div id="ehliyet" class="tab"><div class="card">
    <h2><i class="fas fa-id-card"></i> Ehliyet Alma (DMV)</h2>
    <div class="hint">🪪 <strong>İpucu:</strong> NJ'de 6 Points of ID sistemi. Undocumented bile ehliyet alabiliyor.</div>
    <div class="form-row">
      <div class="field"><label>State</label><input id="l1" placeholder="New Jersey"></div>
      <div class="field"><label>Durum</label><select id="l2"><option>İlk kez alıyorum</option><option>Türk ehliyetimi çevirmek istiyorum</option><option>SSN / ITIN yok</option><option>Real ID lazım</option></select></div>
    </div>
    <button class="btn" id="lb" onclick="call('/ehliyet',{state:g('l1'),durum:g('l2')},'lo','lb','Ehliyet Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="lo" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('lo')">Kopyala</button></div>
  </div></div>

  <div id="banka" class="tab"><div class="card">
    <h2><i class="fas fa-university"></i> Banka Hesabı Açma</h2>
    <div class="hint">💳 <strong>İpucu:</strong> Chase/BofA pasaportla açılıyor. Secured card ile kredi skoru başlatılır.</div>
    <div class="field"><label>Durum</label><select id="ba1"><option>SSN olmadan banka açmak istiyorum</option><option>Kredi kartı almak istiyorum</option><option>Credit score sıfırdan nasıl yaparım?</option><option>En iyi ücretsiz banka hangisi?</option></select></div>
    <button class="btn" id="bb" onclick="call('/banka',{durum:g('ba1')},'bo','bb','Banka Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="bo" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('bo')">Kopyala</button></div>
  </div></div>

  <div id="telefon" class="tab"><div class="card">
    <h2><i class="fas fa-phone"></i> ABD Telefon Numarası</h2>
    <div class="hint">📱 <strong>İpucu:</strong> Google Voice ile SSN olmadan ücretsiz Amerikan numarası alabilirsin.</div>
    <div class="field"><label>Konu</label><select id="p1"><option>Ücretsiz numara (Google Voice)</option><option>Ucuz hat (Mint, Visible, T-Mobile)</option><option>SSN olmadan kontrat hat</option><option>Türkiye'yi ucuz arama</option></select></div>
    <button class="btn" id="pb" onclick="call('/telefon',{konu:g('p1')},'po','pb','Telefon Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="po" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('po')">Kopyala</button></div>
  </div></div>

  <div id="arac" class="tab"><div class="card">
    <h2><i class="fas fa-car-side"></i> Araç Kiralama / Satın Alma</h2>
    <div class="hint">🚗 <strong>İpucu:</strong> SSN olmadan araç satın alınabiliyor. CarMax/Carvana ile başla.</div>
    <div class="form-row">
      <div class="field"><label>State</label><input id="ar1" placeholder="New Jersey"></div>
      <div class="field"><label>Konu</label><select id="ar2"><option>İkinci el araç almak istiyorum</option><option>Araç kiralamak istiyorum</option><option>Araç sigortası almak istiyorum</option><option>SSN olmadan araç alınır mı?</option></select></div>
    </div>
    <button class="btn" id="arb" onclick="call('/arac',{state:g('ar1'),konu:g('ar2')},'aro','arb','Araç Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="aro" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('aro')">Kopyala</button></div>
  </div></div>

  <div id="wise" class="tab"><div class="card">
    <h2><i class="fas fa-exchange-alt"></i> Para Transfer (Wise / Zelle)</h2>
    <div class="hint">💸 <strong>İpucu:</strong> Wise ile TL/$ kurunu en düşük komisyonla gönder.</div>
    <div class="field"><label>Konu</label><select id="w1"><option>Wise ile Türkiye'ye para gönderme</option><option>Wise limitleri ve ücretleri</option><option>Zelle nasıl kullanılır?</option><option>Venmo / CashApp rehberi</option></select></div>
    <button class="btn" id="wb" onclick="call('/wise',{konu:g('w1')},'wo','wb','Para Transfer Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="wo" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('wo')">Kopyala</button></div>
  </div></div>

  <div id="ucak" class="tab"><div class="card">
    <h2><i class="fas fa-plane"></i> Uçak & Bagaj</h2>
    <div class="hint">✈️ <strong>İpucu:</strong> THY NJ-İstanbul $400-700. Bagaj fazlasını 24 saat önce öde ucuza.</div>
    <div class="form-row">
      <div class="field"><label>Havayolu</label><select id="u1"><option>Turkish Airlines</option><option>American Airlines</option><option>United</option><option>Delta</option></select></div>
      <div class="field"><label>Konu</label><select id="u2"><option>Bagaj ücretleri ve kurallar</option><option>En ucuz bilet nasıl bulunur?</option><option>Check-in rehberi</option><option>Refund / iptal kuralları</option></select></div>
    </div>
    <button class="btn" id="ub" onclick="call('/ucak',{havayolu:g('u1'),konu:g('u2')},'uo','ub','Uçak Rehberi')">Rehber Oluştur</button>
    <div class="output-wrap"><div id="uo" class="output">Sonuç burada çıkacak...</div><button class="copy-btn" onclick="cp('uo')">Kopyala</button></div>
  </div></div>

  <div id="sorgu" class="tab"><div class="card">
    <h2><i class="fas fa-question-circle"></i> Herhangi Bir Soru Sor</h2>
    <div class="hint">🤖 ABD hayatıyla ilgili aklına takılan her şeyi sor. Türkçe cevap gelir.</div>
    <div class="field"><label>Sorun nedir?</label><textarea id="q1" rows="4" placeholder="örn. SSN olmadan iş bulabilir miyim? İlk ay ne yapmalıyım?"></textarea></div>
    <button class="btn" id="qb" onclick="call('/sorgu',{soru:g('q1')},'qo','qb','Cevapla')">Cevapla</button>
    <div class="output-wrap"><div id="qo" class="output">Cevap burada çıkacak...</div><button class="copy-btn" onclick="cp('qo')">Kopyala</button></div>
  </div></div>

</div>
<div class="footer">
  <strong>🇺🇸 ABD Yaşam Rehberi</strong> | NJ Odaklı<br>
  Hiçbir kişisel veri saklanmaz<br>
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ Bu araç yalnızca bilgilendirme amaçlıdır. Yapay zeka hata yapabilir.
    Hukuki, mali veya tıbbi konularda mutlaka uzman görüşü alınız.
    Yapay zeka çıktısına dayanarak alınan kararlardan sorumluluk kabul edilmez.
  </span>
</div>
<script>
function g(id){return document.getElementById(id).value;}
function show(tab,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  document.getElementById(tab).classList.add('active');
  btn.classList.add('active');
}
function cp(id){
  navigator.clipboard.writeText(document.getElementById(id).innerText).then(()=>{
    const btn=document.querySelector('#'+id).parentNode.querySelector('.copy-btn');
    btn.textContent='Kopyalandı!';
    setTimeout(()=>btn.textContent='Kopyala',2000);
  });
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId);
  const btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Üretiliyor...';
  out.textContent='AI düşünüyor...';
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json();
    out.textContent=j.result;
  }catch(e){
    out.textContent='Hata: '+e.message;
  }finally{
    btn.disabled=false;
    btn.textContent=label;
  }
}
</script>
</body>
</html>"""

# ─── ROUTES ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/vize', methods=['POST'])
def do_vize():
    try:
        d = request.json
        return jsonify(result=llm("ABD göçmenlik uzmanısın, Türkçe pratik rehber ver.",
            f"{d['tip']} vizesi. State: {d.get('state','')}. Durum: {d.get('durum','')}. Belgeler, formlar, ücretler, hatalar, linkler."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/vergi', methods=['POST'])
def do_vergi():
    try:
        d = request.json
        return jsonify(result=llm("ABD vergi uzmanısın, Türkçe sade anlat.",
            f"Form: {d['form']}. Kazanç: ${d.get('kazanc',0)}. Vize: {d.get('vize','')}. State: {d.get('state','')}. Doldurma rehberi, iade tahmini, deadline'lar."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/rideshare', methods=['POST'])
def do_rideshare():
    try:
        d = request.json
        return jsonify(result=llm("Rideshare ve gig economy uzmanısın, Türkçe yaz.",
            f"{d['app']} - {d.get('state','')}. Konu: {d.get('konu','')}. Belgeler, kazanç, vergi, ipuçları."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/ev', methods=['POST'])
def do_ev():
    try:
        d = request.json
        return jsonify(result=llm("ABD emlak uzmanısın, Türkçe yaz.",
            f"{d.get('sehir','')} ${d.get('butce','')} bütçe. Durum: {d.get('durum','')}. Siteler, belgeler, müzakere tüyoları."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/saglik', methods=['POST'])
def do_saglik():
    try:
        d = request.json
        return jsonify(result=llm("ABD sağlık sistemi uzmanısın, Türkçe pratik yaz.",
            f"{d.get('state','')} - {d.get('durum','')}. Adresler, belgeler, Medicaid, ücretsiz klinikler."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/ehliyet', methods=['POST'])
def do_ehliyet():
    try:
        d = request.json
        return jsonify(result=llm("ABD DMV uzmanısın, Türkçe anlat.",
            f"{d.get('state','')} ehliyet: {d.get('durum','')}. 6 Points belgeler, sınav, randevu, ücretler."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/banka', methods=['POST'])
def do_banka():
    try:
        d = request.json
        return jsonify(result=llm("ABD bankacılık uzmanısın, Türkçe yaz.",
            f"Konu: {d.get('durum','')}. Hangi banka, belgeler, credit score, secured card."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/telefon', methods=['POST'])
def do_telefon():
    try:
        d = request.json
        return jsonify(result=llm("ABD telekomünikasyon uzmanısın, Türkçe rehber.",
            f"Konu: {d.get('konu','')}. Adım adım kurulum, fiyatlar, alternatifler."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/arac', methods=['POST'])
def do_arac():
    try:
        d = request.json
        return jsonify(result=llm("ABD otomotiv uzmanısın, Türkçe yaz.",
            f"{d.get('state','')} - {d.get('konu','')}. Belgeler, sigorta, fiyat, CarMax/Carvana."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/wise', methods=['POST'])
def do_wise():
    try:
        d = request.json
        return jsonify(result=llm("Para transferi uzmanısın, Türkçe anlat.",
            f"Konu: {d.get('konu','')}. Adımlar, komisyonlar, limitler, alternatifler."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/ucak', methods=['POST'])
def do_ucak():
    try:
        d = request.json
        return jsonify(result=llm("Havacılık uzmanısın, Türkçe pratik rehber.",
            f"{d.get('havayolu','')} - {d.get('konu','')}. Detaylı bilgi, ücretler, ipuçları."))
    except Exception: return jsonify(result=traceback.format_exc())

@app.route('/sorgu', methods=['POST'])
def do_sorgu():
    try:
        d = request.json
        return jsonify(result=llm("ABD'deki Türkler için pratik rehber uzmanısın. Türkçe, net, adım adım cevapla.",
            d.get('soru', '')))
    except Exception: return jsonify(result=traceback.format_exc())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

import os
import traceback
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

app = Flask(__name__)
GROQ_KEY = os.environ.get('GROQ_KEY')
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

def llm(system, user):
    if not client:
        return "GROQ_KEY environment variable is missing."
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    return r.choices[0].message.content

HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <title>ABD Yaşam Rehberi - Abdyasam AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root { --blue-dark:1e3a8a; --blue-mid:2563eb; --blue-light:3b82f6; --blue-pale:f0f9ff; --white:#fff; --gray:#666; --radius:10px; }
        * { box-sizing:border-box; margin:0; padding:0; }
        body { font-family:Segoe UI,Arial,sans-serif; background:var(--blue-pale); color:#222; }
        .header { background:linear-gradient(135deg,var(--blue-dark),var(--blue-light)); color:white; padding:24px 20px; text-align:center; }
        .header h1 { font-size:2em; margin-bottom:6px; letter-spacing:1px; }
        .badges { display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-top:10px; }
        .badge { background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.4); border-radius:20px; padding:4px 12px; font-size:.8em; }
        .container { max-width:900px; margin:24px auto; padding:0 16px; }
        .tabs { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:6px; margin-bottom:20px; }
        .tabs button { background:var(--blue-mid); color:white; border:none; padding:12px 8px; border-radius:var(--radius); cursor:pointer; font-size:12px; font-weight:600; transition:all .2s; }
        .tabs button:hover { background:var(--blue-dark); transform:translateY(-1px); }
        .tabs button.active { background:var(--blue-dark); border-bottom:3px solid var(--blue-light); }
        .tab { display:none; }
        .tab.active { display:block; }
        .card { background:var(--white); padding:24px; border-radius:14px; box-shadow:0 4px 16px rgba(0,0,0,.08); }
        .card h2 { color:var(--blue-dark); margin-bottom:6px; font-size:1.3em; }
        .card .hint { color:var(--gray); font-size:.85em; margin-bottom:16px; }
        .form-row { display:grid; grid-template-columns:1fr; gap:12px; margin-bottom:16px; }
        .form-row.two { grid-template-columns:1fr 1fr; }
        @media (max-width:500px) { .form-row.two { grid-template-columns:1fr; } }
        .field { display:flex; flex-direction:column; gap:4px; }
        label { font-weight:600; color:var(--blue-dark); font-size:.9em; }
        input, select, textarea { width:100%; padding:10px 12px; border:1.5px solid #ddd; border-radius:var(--radius); font-size:14px; transition:border .2s; background:#fafafa; }
        input:focus, select:focus, textarea:focus { border-color:var(--blue-light); outline:none; background:white; }
        textarea { resize:vertical; min-height:100px; }
        .btn { background:linear-gradient(135deg,#1d4ed8,#3b82f6); color:white; border:none; padding:14px; width:100%; border-radius:var(--radius); font-size:15px; cursor:pointer; margin:14px 0 8px; font-weight:bold; letter-spacing:.5px; transition:all .2s; box-shadow:0 3px 8px rgba(0,0,0,.15); }
        .btn:hover { transform:translateY(-2px); box-shadow:0 5px 14px rgba(0,0,0,.2); }
        .btn:disabled { opacity:.6; cursor:not-allowed; transform:none; }
        .output-wrap { position:relative; margin-top:16px; }
        .output { background:#f6fdf6; border:1.5px solid #bfdbfe; border-radius:var(--radius); padding:16px; min-height:80px; white-space:pre-wrap; font-size:14px; line-height:1.7; }
        .copy-btn { position:absolute; top:8px; right:8px; background:var(--blue-mid); color:white; border:none; border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer; opacity:0; transition:opacity .2s; }
        .output-wrap:hover .copy-btn { opacity:1; }
        .spinner { display:inline-block; width:16px; height:16px; border:3px solid rgba(255,255,255,.3); border-top-color:white; border-radius:50%; animation:spin .8s linear infinite; vertical-align:middle; margin-right:6px; }
        @keyframes spin { to { transform:rotate(360deg); } }
        hr { border:none; border-top:1px solid #e0e0e0; margin:16px 0; }
        .footer { text-align:center; padding:24px 16px; color:var(--gray); font-size:13px; line-height:2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ABD Yaşam Rehberi AI</h1>
        <p>Abdyasam gibi pratik rehberler için AI tabanlı araçlar</p>
        <div class="badges">
            <span class="badge">Türkçe/English</span>
            <span class="badge">Vergi & Vize</span>
            <span class="badge">Günlük Hayat</span>
            <span class="badge">Ücretsiz</span>
        </div>
    </div>
    <div class="container">
        <div class="tabs">
            <button class="active" onclick="showTab('vize')">Vize</button>
            <button onclick="showTab('vergi')">Vergi</button>
            <button onclick="showTab('arac')">Araç</button>
            <button onclick="showTab('ev')">Ev</button>
            <button onclick="showTab('rideshare')">Rideshare</button>
            <button onclick="showTab('saglik')">Sağlık</button>
            <button onclick="showTab('banka')">Banka</button>
            <button onclick="showTab('sorgu')">Genel Soru</button>
        </div>
        <!-- Vize Tab -->
        <div id="vize" class="tab active">
            <div class="card">
                <h2>Vize & Göçmenlik Rehberi</h2>
                <p class="hint">J-1, H1B veya yeşil kart için adım adım rehber.</p>
                <hr>
                <div class="form-row">
                    <div class="field">
                        <label>Vize Tipi</label>
                        <select id="v1">
                            <option>J-1 Öğrenci</option>
                            <option>H-1B İş</option>
                            <option>E-2 Yatırım</option>
                            <option>Yeşil Kart</option>
                            <option>Diğer</option>
                        </select>
                    </div>
                </div>
                <div class="field">
                    <label>Durum (State)</label>
                    <input id="v2" placeholder="e.g. New Jersey">
                </div>
                <button class="btn" id="vb" onclick="callApi('vize', {tip: v1.value, state: v2.value}, 'vo', 'vb', 'Vize Rehberi Oluştur')">Vize Rehberi Oluştur</button>
                <div class="output-wrap"><div id="vo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="copyOut('vo')">Kopyala</button></div>
            </div>
        </div>
        <!-- Vergi Tab -->
        <div id="vergi" class="tab">
            <div class="card">
                <h2>Vergi İadesi & Formlar</h2>
                <p class="hint">W-4, 1040NR, iade hesaplama.</p>
                <hr>
                <div class="form-row two">
                    <div class="field">
                        <label>Form Tipi</label>
                        <select id="t1">
                            <option>W-4 Bordro</option>
                            <option>1040NR İade</option>
                            <option>1099 Rideshare</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Gelir ($)</label>
                        <input id="t2" type="number" placeholder="e.g. 30000">
                    </div>
                </div>
                <button class="btn" id="tb" onclick="callApi('vergi', {form: t1.value, gelir: t2.value}, 'to', 'tb', 'Vergi Hesapla')">Vergi Hesapla</button>
                <div class="output-wrap"><div id="to" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="copyOut('to')">Kopyala</button></div>
            </div>
        </div>
        <!-- Diğer Tablar Basitleştirildi - Gerçek app'te genişlet -->
        <div id="arac" class="tab"><div class="card"><h2>Araç Alım/Kiralama</h2><p>Paste your query or use form...</p><!-- Form ekle --></div></div>
        <div id="ev" class="tab"><div class="card"><h2>Ev Kiralama/Satın Alma</h2><!-- Form --></div></div>
        <div id="rideshare" class="tab"><div class="card"><h2>Rideshare (Uber/Lyft)</h2><!-- Form --></div></div>
        <div id="saglik" class="tab"><div class="card"><h2>Sağlık Sigortası</h2><!-- Form --></div></div>
        <div id="banka" class="tab"><div class="card"><h2>Banka & Para Transfer</h2><!-- Form --></div></div>
        <div id="sorgu" class="tab">
            <div class="card">
                <h2>Genel ABD Soru</h2>
                <p class="hint">Herhangi bir ABD hayat sorusu sor.</p>
                <div class="field">
                    <label>Soru</label>
                    <textarea id="q1" rows="4" placeholder="ABD'de ev kiralama nasıl yapılır?"></textarea>
                </div>
                <button class="btn" id="qb" onclick="callApi('sorgu', {soru: q1.value}, 'qo', 'qb', 'Cevapla')">Cevapla</button>
                <div class="output-wrap"><div id="qo" class="output">Sonuç burada görünecek...</div><button class="copy-btn" onclick="copyOut('qo')">Kopyala</button></div>
            </div>
        </div>
    </div>
    <div class="footer">
        <strong>ABD Yaşam Rehberi AI</strong> | Abdyasam inspired | No data stored
    </div>
    <script>
        function v(id) { return document.getElementById(id).value; }
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
            document.getElementById(tab).classList.add('active');
            event.target.classList.add('active');
        }
        function copyOut(id) {
            const text = document.getElementById(id).innerText;
            navigator.clipboard.writeText(text).then(() => {
                const btn = document.querySelector(`#${id}`).parentNode.querySelector('.copy-btn');
                btn.textContent = 'Kopyalandı!';
                setTimeout(() => btn.textContent = 'Kopyala', 2000);
            });
        }
        async function callApi(endpoint, data, outId, btnId, label) {
            const out = document.getElementById(outId);
            const btn = document.getElementById(btnId);
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>Üretiliyor...';
            out.innerHTML = 'AI düşünüyor...';
            try {
                const r = await fetch(`/${endpoint}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                if (!r.ok) {
                    const txt = await r.text();
                    out.innerHTML = 'Sunucu hatası: ' + txt.substring(0,300);
                    return;
                }
                const j = await r.json();
                out.innerHTML = j.result;
            } catch(e) {
                out.innerHTML = 'Hata: ' + e.message;
            } finally {
                btn.disabled = false;
                btn.innerHTML = label;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/vize', methods=['POST'])
def do_vize():
    try:
        d = request.json
        return jsonify(result=llm(
            "Türkçe konuşan ABD göçmeni uzmanısın. Pratik, adım adım rehber ver.",
            f"{d['tip']} vizesi için {d.get('state', 'ABD')} rehberi oluştur. Form linkleri, ücretler, hatalar dahil."
        ))
    except Exception:
        return jsonify(result=f"Hata: {traceback.format_exc()}"), 200

@app.route('/vergi', methods=['POST'])
def do_vergi():
    try:
        d = request.json
        return jsonify(result=llm(
            "ABD vergi uzmanısın, Türk göçmenlere Türkçe anlat.",
            f"{d['form']} için vergi rehberi. Gelir: ${d.get('gelir',0)}. İade tahmini, linkler, SSN ITIN."
        ))
    except Exception:
        return jsonify(result=f"Hata: {traceback.format_exc()}"), 200

@app.route('/sorgu', methods=['POST'])
def do_sorgu():
    try:
        d = request.json
        return jsonify(result=llm(
            "Abdyasam.blogspot.com gibi pratik ABD yaşam rehberi uzmanısın. Türkçe, net cevaplar ver.",
            d['soru']
        ))
    except Exception:
        return jsonify(result=f"Hata: {traceback.format_exc()}"), 200

# Diğer route'lar için genişlet: /arac, /ev vb. aynı pattern

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

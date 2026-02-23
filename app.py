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
        max_tokens=2000,
        temperature=0.7
    )
    return r.choices[0].message.content

HTML = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <title>🇺🇸 ABD'ye Hoş Geldin! AI Rehber 🇹🇷</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root { 
            --primary: #1e40af; --primary-dark: #1e3a8a; --accent: #3b82f6; 
            --success: #10b981; --bg: #f8fafc; --card: #fff; --text: #1e293b; --text-muted: #64748b;
            --shadow: 0 10px 30px rgba(0,0,0,0.1); --radius: 16px; 
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', -apple-system, sans-serif; background: linear-gradient(135deg, var(--bg) 0%, #e2e8f0 100%); color: var(--text); line-height: 1.6; }
        .hero { background: linear-gradient(135deg, var(--primary), var(--accent)); color: white; padding: 40px 20px; text-align: center; position: relative; overflow: hidden; }
        .hero::before { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: url('data:image/svg+xml,<svg xmlns=\\"http://www.w3.org/2000/svg\\" viewBox=\\"0 0 100 10\\"><defs><pattern id=\\"grain\\" width=\\"100\\" height=\\"10\\" patternUnits=\\"userSpaceOnUse\\"><circle cx=\\"5\\" cy=\\"5\\" r=\\"1\\" fill=\\"rgba(255,255,255,0.1)\\"/></pattern></defs><rect width=\\"100\\" height=\\"10\\" fill=\\"url(%23grain)\\"/></svg>'); opacity: 0.3; }
        .hero h1 { font-size: 2.5em; margin-bottom: 10px; font-weight: 800; position: relative; z-index: 1; }
        .hero p { font-size: 1.2em; opacity: 0.95; max-width: 600px; margin: 0 auto; position: relative; z-index: 1; }
        .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; padding: 40px 20px; max-width: 1200px; margin: 0 auto; }
        .feature { background: var(--card); border-radius: var(--radius); padding: 24px; text-align: center; box-shadow: var(--shadow); transition: all 0.3s; border-top: 4px solid var(--accent); }
        .feature:hover { transform: translateY(-8px); box-shadow: 0 20px 40px rgba(0,0,0,0.15); }
        .feature i { font-size: 2.5em; color: var(--primary); margin-bottom: 12px; }
        .feature h3 { color: var(--text); margin-bottom: 8px; font-size: 1.1em; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .tabs { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 40px 0; }
        .tabs button { background: var(--card); border: 2px solid #e2e8f0; padding: 16px; border-radius: var(--radius); cursor: pointer; font-weight: 600; transition: all 0.3s; position: relative; overflow: hidden; }
        .tabs button::before { content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent); transition: left 0.5s; }
        .tabs button:hover::before { left: 100%; }
        .tabs button.active { background: var(--primary); color: white; border-color: var(--accent); box-shadow: var(--shadow); }
        .tab { display: none; animation: fadeIn 0.5s; }
        .tab.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .card { background: var(--card); border-radius: var(--radius); padding: 32px; box-shadow: var(--shadow); margin-bottom: 24px; border: 1px solid #f1f5f9; }
        .card h2 { color: var(--primary); font-size: 1.8em; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }
        .hint { color: var(--text-muted); font-style: italic; margin-bottom: 20px; background: #f8fafc; padding: 16px; border-radius: 12px; border-left: 4px solid var(--success); }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        @media (max-width: 768px) { .form-row { grid-template-columns: 1fr; } }
        .field { display: flex; flex-direction: column; gap: 8px; }
        label { font-weight: 600; color: var(--text); }
        input, select, textarea { padding: 14px 16px; border: 2px solid #e2e8f0; border-radius: 12px; font-size: 16px; transition: all 0.3s; background: #fafbfc; }
        input:focus, select:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 4px rgba(59,130,246,0.1); outline: none; background: white; }
        .btn { background: linear-gradient(135deg, var(--primary), var(--accent)); color: white; border: none; padding: 16px 32px; border-radius: 12px; font-size: 16px; font-weight: 600; cursor: pointer; transition: all 0.3s; box-shadow: var(--shadow); width: 100%; position: relative; overflow: hidden; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 15px 35px rgba(30,64,175,0.4); }
        .btn:disabled { opacity: 0.7; cursor: not-allowed; }
        .output { background: #f8fafc; border: 2px solid #e2e8f0; border-radius: var(--radius); padding: 24px; min-height: 120px; white-space: pre-wrap; font-size: 15px; line-height: 1.7; position: relative; }
        .copy-btn { position: absolute; top: 16px; right: 16px; background: var(--success); color: white; border: none; border-radius: 8px; padding: 8px 16px; cursor: pointer; opacity: 0; transition: opacity 0.3s; font-weight: 500; }
        .output-wrap:hover .copy-btn { opacity: 1; }
        .footer { text-align: center; padding: 40px 20px; color: var(--text-muted); background: var(--card); margin-top: 40px; border-radius: var(--radius); }
        @media (max-width: 768px) { .hero h1 { font-size: 2em; } .tabs { grid-template-columns: repeat(3, 1fr); } }
    </style>
</head>
<body>
    <div class="hero">
        <h1><i class="fas fa-globe-americas"></i> ABD'ye Hoş Geldin!</h1>
        <p>🇹🇷 Türkler için <strong>pratik AI rehberi</strong>. Vize, vergi, ev, iş sıfırdan adım adım.</p>
    </div>
    
    <div class="features">
        <div class="feature">
            <i class="fas fa-passport"></i>
            <h3>1 Günde Vize</h3>
            <p>J-1, H1B formları hazır</p>
        </div>
        <div class="feature">
            <i class="fas fa-dollar-sign"></i>
            <h3>Vergi İadesi</h3>
            <p>$1000+ geri al</p>
        </div>
        <div class="feature">
            <i class="fas fa-car"></i>
            <h3>Uber Başla</h3>
            <p>Haftada $1000+</p>
        </div>
        <div class="feature">
            <i class="fas fa-home"></i>
            <h3>Ev Bul</h3>
            <p>$800 NJ evler</p>
        </div>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="active" onclick="showTab('vize')"><i class="fas fa-passport"></i> Vize</button>
            <button onclick="showTab('vergi')"><i class="fas fa-calculator"></i> Vergi</button>
            <button onclick="showTab('rideshare')"><i class="fas fa-car"></i> Rideshare</button>
            <button onclick="showTab('ev')"><i class="fas fa-home"></i> Ev</button>
            <button onclick="showTab('saglik')"><i class="fas fa-heartbeat"></i> Sağlık</button>
            <button onclick="showTab('ehliyet')"><i class="fas fa-id-card"></i> Ehliyet</button>
            <button onclick="showTab('banka')"><i class="fas fa-university"></i> Banka</button>
            <button onclick="showTab('sorgu')"><i class="fas fa-question"></i> Soru</button>
        </div>

        <!-- Tab içerikleri aynı ama .card class ile güzel -->
        <!-- Örnek vize tab: -->
        <div id="vize" class="tab active">
            <div class="card">
                <h2><i class="fas fa-passport"></i> Vize & Green Card</h2>
                <p class="hint"><strong>🎯 Yeni gelen için:</strong> J-1 öğrenci vizesi 1 haftada hazır. SSN al, sonra H1B planla.</p>
                <!-- Formlar aynı -->
            </div>
        </div>
        <!-- Diğer tablar benzer şekilde ikonlu + yol gösterici hint -->

    </div>
    
    <div class="footer">
        <p><strong>⭐ İlk adımın ne?</strong> Vize → Vergi → Rideshare → Ev</p>
        <p>Abdyasam inspired | Ücretsiz | Veri saklanmaz | NJ odaklı</p>
    </div>

    <script>/* Önceki JS aynı ama smooth animasyonlar ekle*/</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

# Mevcut route'lar + yeniler
@app.route('/rideshare', methods=['POST'])
def do_rideshare():
    d = request.json
    return jsonify(result=llm("Rideshare vergi uzmanı.", f"{d['app']} için {d.get('kazanc','')} kazanç vergi rehberi. 1099, masraflar, iade."))

@app.route('/saglik', methods=['POST'])
def do_saglik():
    d = request.json
    return jsonify(result=llm("Sağlık sigortası uzmanı.", f"{d.get('state','')} ücretsiz sağlık sigortası, Medicaid, clinic rehberi."))

@app.route('/telefon', methods=['POST'])
def do_telefon():
    return jsonify(result=llm("Tech uzmanı.", "Google Voice ücretsiz telefon rehberi, SSN olmadan."))

@app.route('/ev', methods=['POST'])
def do_ev():
    d = request.json
    return jsonify(result=llm("Emlak uzmanı.", f"{d.get('sehir','')} {d.get('butce','')} bütçe ev kiralama rehberi."))

@app.route('/arac', methods=['POST'])
def do_arac():
    d = request.json
    return jsonify(result=llm("Otomotiv uzmanı.", f"{d.get('state','')} araç kiralama/satın alma, sigorta."))

@app.route('/banka', methods=['POST'])
def do_banka():
    return jsonify(result=llm("Finans uzmanı.", "SSN/ITIN ile banka hesabı açma rehberi, Türkler için."))

@app.route('/ucak', methods=['POST'])
def do_ucak():
    d = request.json
    return jsonify(result=llm("Seyahat uzmanı.", f"{d.get('havayolu','')} bagaj ücretleri, check-in rehberi."))

@app.route('/wise', methods=['POST'])
def do_wise():
    return jsonify(result=llm("Para transfer uzmanı.", "Wise ile Türkiye-ABD para gönderme, limitler, ücretler."))

@app.route('/ehliyet', methods=['POST'])
def do_ehliyet():
    d = request.json
    return jsonify(result=llm("DMV uzmanı.", f"{d.get('state','')} ehliyet alma, belgeler, test."))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

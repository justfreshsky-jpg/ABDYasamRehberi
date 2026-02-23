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
    <title>ABD Yaşam Rehberi - Abdyasam AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>/* Önceki CSS aynı kalıyor - kısaltmak için tekrar etmiyorum */</style>
</head>
<body>
    <!-- Header aynı -->
    <div class="container">
        <div class="tabs">
            <button class="active" onclick="showTab('vize')">Vize</button>
            <button onclick="showTab('vergi')">Vergi</button>
            <button onclick="showTab('rideshare')">Rideshare</button>
            <button onclick="showTab('saglik')">Sağlık</button>
            <button onclick="showTab('telefon')">Telefon</button>
            <button onclick="showTab('ev')">Ev</button>
            <button onclick="showTab('arac')">Araç</button>
            <button onclick="showTab('banka')">Banka</button>
            <button onclick="showTab('ucak')">Uçak</button>
            <button onclick="showTab('wise')">Wise</button>
            <button onclick="showTab('ehliyet')">Ehliyet</button>
            <button onclick="showTab('sorgu')">Soru</button>
        </div>

        <!-- Vize -->
        <div id="vize" class="tab active"> <!-- Önceki aynı --> </div>

        <!-- Vergi -->
        <div id="vergi" class="tab"> <!-- Önceki aynı --> </div>

        <!-- Rideshare -->
        <div id="rideshare" class="tab">
            <div class="card">
                <h2>Rideshare (Uber/Lyft)</h2>
                <p class="hint">1099 form, vergi, kazanç ipuçları.</p>
                <div class="form-row two">
                    <div class="field"><label>App</label><select id="r1"><option>Uber</option><option>Lyft</option></select></div>
                    <div class="field"><label>2025 Kazanç ($)</label><input id="r2" type="number"></div>
                </div>
                <button class="btn" id="rb" onclick="callApi('rideshare', {app: r1.value, kazanc: r2.value}, 'ro', 'rb', 'Rehber Oluştur')">Rehber Oluştur</button>
                <div class="output-wrap"><div id="ro" class="output"></div><button class="copy-btn" onclick="copyOut('ro')">Kopyala</button></div>
            </div>
        </div>

        <!-- Sağlık -->
        <div id="saglik" class="tab">
            <div class="card">
                <h2>Ücretsiz Sağlık Sigortası</h2>
                <p class="hint">NJ Medicaid, NY free clinic'ler.</p>
                <div class="field"><label>State</label><input id="h1" placeholder="New Jersey"></div>
                <button class="btn" id="hb" onclick="callApi('saglik', {state: h1.value}, 'ho', 'hb', 'Sigorta Bul')">Sigorta Bul</button>
                <div class="output-wrap"><div id="ho" class="output"></div><button class="copy-btn" onclick="copyOut('ho')">Kopyala</button></div>
            </div>
        </div>

        <!-- Telefon -->
        <div id="telefon" class="tab">
            <div class="card">
                <h2>Ücretsiz Telefon (Google Voice)</h2>
                <p class="hint">SSN olmadan telefon numarası.</p>
                <button class="btn" id="tb" onclick="callApi('telefon', {}, 'to', 'tb', 'Adım Adım Rehber')">Rehber Oluştur</button>
                <div class="output-wrap"><div id="to" class="output"></div><button class="copy-btn" onclick="copyOut('to')">Kopyala</button></div>
            </div>
        </div>

        <!-- Ev -->
        <div id="ev" class="tab">
            <div class="card">
                <h2>Ev Kiralama</h2>
                <div class="form-row two">
                    <div class="field"><label>Şehir</label><input id="e1" placeholder="Newark NJ"></div>
                    <div class="field"><label>Kira Bütçe ($)</label><input id="e2" type="number"></div>
                </div>
                <button class="btn" id="eb" onclick="callApi('ev', {sehir: e1.value, butce: e2.value}, 'eo', 'eb', 'Ev Bul')">Ev Rehberi</button>
                <div class="output-wrap"><div id="eo" class="output"></div><button class="copy-btn" onclick="copyOut('eo')">Kopyala</button></div>
            </div>
        </div>

        <!-- Araç -->
        <div id="arac" class="tab">
            <div class="card">
                <h2>Araç Kiralama/Satın Alma</h2>
                <div class="field"><label>Durum</label><input id="a1"></div>
                <button class="btn" id="ab" onclick="callApi('arac', {state: a1.value}, 'ao', 'ab', 'Araç Rehberi')">Rehber</button>
                <div class="output-wrap"><div id="ao" class="output"></div><button class="copy-btn" onclick="copyOut('ao')">Kopyala</button></div>
            </div>
        </div>

        <!-- Banka -->
        <div id="banka" class="tab">
            <div class="card">
                <h2>Banka Hesabı Açma</h2>
                <button class="btn" id="bb" onclick="callApi('banka', {}, 'bo', 'bb', 'Banka Rehberi')">Rehber</button>
                <div class="output-wrap"><div id="bo" class="output"></div><button class="copy-btn" onclick="copyOut('bo')">Kopyala</button></div>
            </div>
        </div>

        <!-- Uçak -->
        <div id="ucak" class="tab">
            <div class="card">
                <h2>Uçak Bilet & Bagaj</h2>
                <div class="field"><label>Havayolu</label><input id="u1"></div>
                <button class="btn" id="ub" onclick="callApi('ucak', {havayolu: u1.value}, 'uo', 'ub', 'Bagaj Rehberi')">Rehber</button>
                <div class="output-wrap"><div id="uo" class="output"></div><button class="copy-btn" onclick="copyOut('uo')">Kopyala</button></div>
            </div>
        </div>

        <!-- Wise -->
        <div id="wise" class="tab">
            <div class="card">
                <h2>Para Transfer Wise</h2>
                <button class="btn" id="wb" onclick="callApi('wise', {}, 'wo', 'wb', 'Wise Rehberi')">Rehber</button>
                <div class="output-wrap"><div id="wo" class="output"></div><button class="copy-btn" onclick="copyOut('wo')">Kopyala</button></div>
            </div>
        </div>

        <!-- Ehliyet -->
        <div id="ehliyet" class="tab">
            <div class="card">
                <h2>Ehliyet Alımı</h2>
                <div class="field"><label>State</label><input id="l1"></div>
                <button class="btn" id="lb" onclick="callApi('ehliyet', {state: l1.value}, 'lo', 'lb', 'Ehliyet Rehberi')">Rehber</button>
                <div class="output-wrap"><div id="lo" class="output"></div><button class="copy-btn" onclick="copyOut('lo')">Kopyala</button></div>
            </div>
        </div>

        <!-- Genel Soru -->
        <div id="sorgu" class="tab"><!-- Önceki aynı --></div>
    </div>
    <!-- Script aynı -->
</body>
</html>
"""  # Tam HTML'i önceki mesajdan kopyala, CSS ve script ekle

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

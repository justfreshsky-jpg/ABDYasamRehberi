# AmerikaRehberSitesi

Türkler için ABD yaşam konularına odaklı, Flask tabanlı tek sayfa AI rehber uygulaması.

## Özellikler
- Vize, vergi, SSN, banka, sağlık, ev, rideshare gibi başlıklarda yönlendirme.
- Groq LLM üzerinden Türkçe, adım adım cevap üretimi.
- Arka planda blog içeriği çekerek prompt bağlamını zenginleştirme.

## Gereksinimler
- Python 3.10+
- `GROQ_KEY` ortam değişkeni

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma (geliştirme)
```bash
export GROQ_KEY="your_key_here"
python app.py
```

Uygulama varsayılan olarak `http://localhost:5000` adresinde çalışır.

## Çalıştırma (production)
```bash
export GROQ_KEY="your_key_here"
export PORT=5000
gunicorn -b 0.0.0.0:${PORT} app:app
```


## Docker (Cloud Run uyumlu)
```bash
docker build -t amerika-rehber .
docker run --rm -p 8080:8080 -e GROQ_KEY="your_key_here" amerika-rehber
```

Cloud Run için Dockerfile kök dizine eklendi ve `PORT` env değişkeni üzerinden `gunicorn` ile başlatılır.

## Notlar
- `GROQ_KEY` yoksa API yanıtı olarak "GROQ_KEY eksik" mesajı döner.
- Harici blog kaynağı çekilemezse uygulama fallback metin ile devam eder.

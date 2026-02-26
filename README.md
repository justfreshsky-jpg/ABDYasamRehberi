# AmerikaRehberSitesi

Türkler için ABD yaşam konularına odaklı, Flask tabanlı tek sayfa AI rehber uygulaması.

## Özellikler
- Vize, vergi, SSN, banka, sağlık, ev, rideshare gibi başlıklarda yönlendirme.
- Google Vertex AI (Gemini) üzerinden Türkçe, adım adım cevap üretimi.
- Arka planda blog içeriği çekerek prompt bağlamını zenginleştirme.

## Gereksinimler
- Python 3.10+
- `GOOGLE_CLOUD_PROJECT` ortam değişkeni
- (Opsiyonel) `VERTEX_LOCATION` (varsayılan: `us-central1`)
- (Opsiyonel) `GEMINI_MODEL` (varsayılan: `gemini-1.5-flash`)

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma (geliştirme)
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export VERTEX_LOCATION="us-east4"
export GEMINI_MODEL="gemini-1.5-flash"
python app.py
```

Uygulama varsayılan olarak `http://localhost:5000` adresinde çalışır.

## Çalıştırma (production)
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export VERTEX_LOCATION="us-east4"
export GEMINI_MODEL="gemini-1.5-flash"
export PORT=5000
gunicorn -b 0.0.0.0:${PORT} app:app
```


## Docker (Cloud Run uyumlu)
```bash
docker build -t amerika-rehber .
docker run --rm -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT="your-gcp-project-id" \
  -e VERTEX_LOCATION="us-east4" \
  -e GEMINI_MODEL="gemini-1.5-flash" \
  amerika-rehber
```

Cloud Run için Dockerfile kök dizine eklendi ve `PORT` env değişkeni üzerinden `gunicorn` ile başlatılır.

## Notlar
- Vertex AI yapılandırması yoksa API yanıtı fallback özet modunda çalışır.
- Harici blog kaynağı çekilemezse uygulama fallback metin ile devam eder.


## Cloud Run ortam değişkeni
`GOOGLE_CLOUD_PROJECT` veya yetkiler eksikse uygulama AI yerine fallback özet döner. Tam AI yanıtı için servis hesabı yetkisini ve env değerlerini ayarlayın:

```bash
gcloud run services update amerikarrehbersitesi \
  --region us-east4 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-gcp-project-id,VERTEX_LOCATION=us-east4,GEMINI_MODEL=gemini-1.5-flash
```

Ayrıca Cloud Run servis hesabına en az `roles/aiplatform.user` rolü verin.

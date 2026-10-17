# PaddleOCR API (Render free tier)

PaddleOCR PP-OCRv4 models running on ONNX Runtime (via RapidOCR) instead of PaddlePaddle,
so it fits in Render's free 512 MB instance. Tested peak ~410 MB on a 12 MP, 60-line image.

## Use

    curl -H "x-api-key: $API_KEY" -F file=@scan.jpg https://<your-service>.onrender.com/ocr

Returns `{"text": "...", "lines": [{"text", "confidence", "box"}]}`.

## Settings (env vars)

| Var | Default | Note |
|---|---|---|
| API_KEY | (generated) | Sent as `x-api-key` header. Empty = no auth. |
| MAX_SIDE | 1280 | Images are downscaled to this before OCR. |
| DET_LIMIT | 736 | Detector input size. Raising it raises RAM. |
| MAX_BYTES | 8388608 | Upload limit. |

## Free tier limits

- Sleeps after 15 min idle; first request after that takes ~1 min (cold start).
- 0.1 CPU: expect a few seconds per page.
- Images only (jpg/png/webp/bmp). Convert PDFs to images first (e.g. in n8n).

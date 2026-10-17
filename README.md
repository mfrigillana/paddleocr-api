# PaddleOCR API (Render free tier)

PaddleOCR PP-OCRv4 models running on ONNX Runtime (via RapidOCR) instead of PaddlePaddle,
so it fits in Render's free 512 MB instance. Tested peak ~410 MB on a 12 MP, 60-line image.

## Use

    curl -H "x-api-key: $API_KEY" -F file=@scan.jpg https://<your-service>.onrender.com/ocr

Returns `{"text": "...", "lines": [{"text", "confidence", "box"}]}`.

## Telegram bot

Send the bot a photo; it replies with the text.

1. In Telegram, message @BotFather, send /newbot, copy the token.
2. In Render > paddleocr-api > Environment, add `TELEGRAM_BOT_TOKEN` = that token. Save (it redeploys).
3. The service registers its own webhook on boot. Message your bot to get your chat ID.
4. Add `TELEGRAM_ALLOWED_CHAT_IDS` = your chat ID(s), comma-separated, so only you/staff can use it.

## Settings (env vars)

| Var | Default | Note |
|---|---|---|
| API_KEY | (generated) | Sent as `x-api-key` header. Empty = no auth. |
| MAX_SIDE | 1280 | Images are downscaled to this before OCR. |
| DET_LIMIT | 736 | Detector input size. Raising it raises RAM. |
| MAX_BYTES | 8388608 | Upload limit. |
| TELEGRAM_BOT_TOKEN | (none) | Enables the bot. |
| TELEGRAM_ALLOWED_CHAT_IDS | (none) | Comma-separated. Empty = anyone can use it. |

## Free tier limits

- Sleeps after 15 min idle; first request after that takes ~1 min (cold start).
- 0.1 CPU: expect a few seconds per page.
- Images only (jpg/png/webp/bmp). Convert PDFs to images first (e.g. in n8n).

"""PaddleOCR (PP-OCRv4 via ONNX Runtime) API + Telegram bot, sized for Render's free tier (512 MB RAM)."""
import hashlib
import os
import re
import threading
from contextlib import asynccontextmanager

import cv2
import httpx
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from rapidocr_onnxruntime import RapidOCR

API_KEY = os.getenv("API_KEY", "")
MAX_SIDE = int(os.getenv("MAX_SIDE", "1280"))        # downscale big photos to save RAM
MAX_BYTES = int(os.getenv("MAX_BYTES", str(8 * 1024 * 1024)))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_ALLOWED = {c.strip() for c in os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",") if c.strip()}
TG_SECRET = hashlib.sha256(TG_TOKEN.encode()).hexdigest()[:32] if TG_TOKEN else ""
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
USE_CLS = os.getenv("USE_CLS", "0") == "1"  # off: the rotate step flips LCD digits (97.50 -> "0516")
PUMP_LABELS = [x.strip() for x in os.getenv("PUMP_LABELS", "Regular,Premium 95,Diesel").split(",")]
SHEET_URL = os.getenv("GSHEET_WEBHOOK_URL", "")  # Apps Script web app /exec URL
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "")  # set automatically by Render

# One thread each: free tier has 0.1 CPU; more threads only add memory.
engine = RapidOCR(
    intra_op_num_threads=1,
    inter_op_num_threads=1,
    det_limit_side_len=int(os.getenv("DET_LIMIT", "736")),
    det_limit_type="max",   # cap detector input size -> caps peak RAM
    rec_batch_num=1,
)
lock = threading.Lock()  # one OCR job at a time keeps peak RAM predictable


@asynccontextmanager
async def lifespan(_app):
    # Register the Telegram webhook on every boot, so there's nothing to set up by hand.
    if TG_TOKEN and PUBLIC_URL:
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.post(f"{TG_API}/setWebhook", json={
                    "url": f"{PUBLIC_URL}/telegram",
                    "secret_token": TG_SECRET,
                    "allowed_updates": ["message"],
                    "drop_pending_updates": False,
                })
                print("telegram setWebhook:", r.json().get("description"))
        except Exception as e:  # never block the API from starting
            print("telegram setWebhook failed:", e)
    yield


app = FastAPI(title="PaddleOCR API", lifespan=lifespan)


def ocr_bytes(data: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image (jpg/png/webp/bmp)")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    with lock:
        result = engine(img, use_cls=USE_CLS)[0]
    lines = [
        {"text": t, "confidence": round(float(c), 4), "box": [[int(x), int(y)] for x, y in b]}
        for b, t, c in (result or [])
    ]
    return {"text": "\n".join(l["text"] for l in lines), "lines": lines}


NUM = re.compile(r"^\d{1,7}\.\d{2}$")
STAMP = re.compile(r"(\d{4}/\d{2}/\d{2})\s*(\d{2}:\d{2})")


def pump_readings(lines: list) -> dict | None:
    """Pump panel photo -> {reading_time, pumps:[{name, litres, price}]}. Top row = litres, row below = price."""
    nums = []
    for l in lines:
        t = l["text"].replace(" ", "")
        if NUM.match(t):
            ys = [p[1] for p in l["box"]]
            xs = [p[0] for p in l["box"]]
            nums.append({"v": t, "x": sum(xs) / 4, "y": sum(ys) / 4, "h": max(ys) - min(ys)})
    if len(nums) < 4:
        return None
    nums.sort(key=lambda n: n["y"])
    rows, cur = [], [nums[0]]
    for n in nums[1:]:
        if abs(n["y"] - cur[-1]["y"]) < 0.6 * max(n["h"], cur[-1]["h"]):
            cur.append(n)
        else:
            rows.append(cur)
            cur = [n]
    rows.append(cur)
    if len(rows) < 2 or len(rows[0]) < 2 or len(rows[0]) != len(rows[1]):
        return None
    litres = sorted(rows[0], key=lambda n: n["x"])
    prices = sorted(rows[1], key=lambda n: n["x"])
    stamp = None
    for l in lines:
        m = STAMP.search(l["text"].replace(" ", ""))
        if m:
            stamp = f"{m.group(1)} {m.group(2)}"
            break
    pumps = [{"name": PUMP_LABELS[i] if i < len(PUMP_LABELS) else f"Pump {i + 1}",
              "litres": float(L["v"]), "price": float(P["v"])}
             for i, (L, P) in enumerate(zip(litres, prices))]
    return {"reading_time": stamp, "pumps": pumps}


def format_readings(r: dict) -> str:
    out = [f"Reading: {r['reading_time']}"] if r["reading_time"] else []
    out += [f"{p['name']}: {p['litres']:,.2f} L @ ₱{p['price']:.2f}/L" for p in r["pumps"]]
    return "\n".join(out)


async def log_to_sheet(r: dict, msg: dict) -> str:
    """POST readings to the Google Apps Script web app. Returns a short status line for the reply."""
    if not SHEET_URL:
        return ""
    sender = msg.get("from") or {}
    payload = {**r, "message_id": f"tg{msg['chat']['id']}_{msg.get('message_id')}",
               "sent_by": sender.get("username") or sender.get("first_name") or ""}
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            res = (await c.post(SHEET_URL, json=payload)).json()
    except Exception as e:
        print("sheet log failed:", repr(e))
        return "\n\nNot saved to sheet (connection error)."
    if not res.get("ok"):
        print("sheet log error:", res)
        return "\n\nNot saved to sheet."
    if res.get("duplicate"):
        return "\n\nAlready in sheet."
    lines = ["\n\nSaved to sheet."]
    for s in res.get("sold", []):
        if s.get("litres") != "":
            lines.append(f"{s['pump']} since last: {s['litres']:,.2f} L = ₱{s['sales']:,.2f}")
    return "\n".join(lines)


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid api key")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    try:
        return await run_in_threadpool(ocr_bytes, data)
    except ValueError as e:
        raise HTTPException(400, str(e))


# ---------------- Telegram ----------------

HELP = ("Send me a photo of a document and I'll reply with the text.\n"
        "Tip: send it as a File (not compressed) for small print.")


async def tg(method: str, **payload):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{TG_API}/{method}", json=payload)
        return r.json()


async def tg_download(file_id: str) -> bytes:
    info = await tg("getFile", file_id=file_id)
    path = info["result"]["file_path"]
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{path}")
        r.raise_for_status()
        return r.content


async def tg_reply(chat_id, text: str, reply_to=None):
    for i in range(0, max(len(text), 1), 4000):  # Telegram limit is 4096 chars
        await tg("sendMessage", chat_id=chat_id, text=text[i:i + 4000] or "-",
                 reply_to_message_id=reply_to if i == 0 else None)


async def handle_message(msg: dict):
    chat_id = msg["chat"]["id"]
    mid = msg.get("message_id")

    if TG_ALLOWED and str(chat_id) not in TG_ALLOWED:
        await tg_reply(chat_id, f"Not authorised. Your chat ID is {chat_id}.")
        return

    file_id = None
    if msg.get("photo"):
        file_id = msg["photo"][-1]["file_id"]  # largest size Telegram kept
    elif (msg.get("document") or {}).get("mime_type", "").startswith("image/"):
        file_id = msg["document"]["file_id"]
        if msg["document"].get("file_size", 0) > MAX_BYTES:
            await tg_reply(chat_id, "That file is too big (8 MB max).", mid)
            return
    elif (msg.get("document") or {}).get("mime_type") == "application/pdf":
        await tg_reply(chat_id, "PDFs aren't supported yet. Please send a photo or screenshot of the page.", mid)
        return

    if not file_id:
        await tg_reply(chat_id, f"{HELP}\n\nYour chat ID: {chat_id}")
        return

    try:
        await tg("sendChatAction", chat_id=chat_id, action="typing")
        data = await tg_download(file_id)
        result = await run_in_threadpool(ocr_bytes, data)
        r = pump_readings(result["lines"])
        if r:
            reply = format_readings(r) + await log_to_sheet(r, msg)
        else:
            reply = result["text"] or "No text found."
        await tg_reply(chat_id, reply, mid)
    except Exception as e:
        print("telegram OCR error:", repr(e))
        await tg_reply(chat_id, "Sorry, I couldn't read that image. Try a clearer photo.", mid)


@app.post("/telegram")
async def telegram_webhook(request: Request, background: BackgroundTasks,
                           x_telegram_bot_api_secret_token: str = Header(default="")):
    if not TG_TOKEN or x_telegram_bot_api_secret_token != TG_SECRET:
        raise HTTPException(403)
    update = await request.json()
    msg = update.get("message")
    if msg:
        # Answer Telegram right away; OCR runs after, so slow pages don't cause re-deliveries.
        background.add_task(handle_message, msg)
    return {"ok": True}

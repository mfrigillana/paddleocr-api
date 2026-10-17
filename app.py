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


def ocr_image(data: bytes):
    """Returns (lines in working-image coords, full-res image, scale full->working)."""
    full = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if full is None:
        raise ValueError("not an image (jpg/png/webp/bmp)")
    h, w = full.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    img = full if scale == 1.0 else cv2.resize(full, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    with lock:
        result = engine(img, use_cls=USE_CLS)[0]
    del img
    lines = [
        {"text": t, "confidence": round(float(c), 4), "box": [[int(x), int(y)] for x, y in b]}
        for b, t, c in (result or [])
    ]
    return lines, full, scale


def ocr_bytes(data: bytes) -> dict:
    lines, _, _ = ocr_image(data)
    return {"text": "\n".join(l["text"] for l in lines), "lines": lines}


NUM = re.compile(r"^\d{1,7}\.\d{2}$")
PRICE = re.compile(r"^\d{2,3}\.\d{2}$")
STAMP = re.compile(r"(\d{4}/\d{2}/\d{2})\s*(\d{2}:\d{2})")


def _tok(l: dict) -> dict:
    xs = [p[0] for p in l["box"]]
    ys = [p[1] for p in l["box"]]
    return {"t": l["text"].replace(" ", ""), "raw": l["text"], "conf": l["confidence"],
            "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
            "cx": sum(xs) / 4, "cy": sum(ys) / 4, "h": max(ys) - min(ys), "w": max(xs) - min(xs)}


def _reread(full, tok: dict, scale: float) -> str | None:
    """Re-read one LCD number from a full-resolution crop around its box (wider, to catch missed digits)."""
    if full is None:
        return None
    H, W = full.shape[:2]
    best = None
    for padx, pady in ((0.12, 0.25), (0.25, 0.35), (0.06, 0.15)):
        x0 = int(max(0, (tok["x0"] - padx * tok["w"]) / scale))
        x1 = int(min(W, (tok["x1"] + padx * tok["w"]) / scale))
        y0 = int(max(0, (tok["y0"] - pady * tok["h"]) / scale))
        y1 = int(min(H, (tok["y1"] + pady * tok["h"]) / scale))
        crop = full[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Close small gaps between LCD segments so digits read as solid strokes.
        dark = cv2.bitwise_not(gray)
        dark = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1)
        variants = [crop, cv2.cvtColor(cv2.bitwise_not(dark), cv2.COLOR_GRAY2BGR)]
        for v in variants:
            with lock:
                res = engine(v, use_det=False, use_cls=False)[0]
            text, score = (res[0][0], float(res[0][1])) if res else ("", 0.0)
            t = text.replace(" ", "")
            if NUM.match(t) and (best is None or score > best[1]):
                best = (t, score)
    return best[0] if best else None


def pump_readings(lines: list, full=None, scale: float = 1.0, width: float | None = None) -> dict | None:
    """Pump panel photo -> {reading_time, pumps:[{name, litres|None, price, raw}]}.
    Prices are the anchor (they read reliably); each pump's litres is the number directly above its price."""
    toks = [_tok(l) for l in lines]
    for t in toks:  # LCD price read without its decimal point, e.g. "9700" -> "97.00"
        if re.fullmatch(r"\d{4}", t["t"]) and 2000 <= int(t["t"]) <= 30000:
            t["t"] = t["t"][:-2] + "." + t["t"][-2:]
    prices = [t for t in toks if PRICE.match(t["t"]) and 20 <= float(t["t"]) <= 300]
    if len(prices) < 2:
        return None
    prices.sort(key=lambda t: t["cy"])
    rows, cur = [], [prices[0]]
    for t in prices[1:]:
        if abs(t["cy"] - cur[-1]["cy"]) < 0.6 * max(t["h"], cur[-1]["h"]):
            cur.append(t)
        else:
            rows.append(cur)
            cur = [t]
    rows.append(cur)
    row = max(rows, key=len)
    if len(row) < 2:
        return None
    n = len(PUMP_LABELS)
    if len(row) > n:  # keep the big price screens
        row = sorted(row, key=lambda t: t["h"], reverse=True)[:n]
    row.sort(key=lambda t: t["cx"])

    # Fixed layout: left-to-right is always PUMP_LABELS (Regular, Premium, Diesel).
    # If a pump's price wasn't found, place the others by which third of the photo they sit in.
    if len(row) == n:
        slots = list(range(n))
    else:
        if width is None:
            width = (full.shape[1] * scale) if full is not None else max(t["x1"] for t in toks)
        slots = [min(n - 1, int(P["cx"] / (width / n))) for P in row]
        if len(set(slots)) != len(slots):
            slots = list(range(len(row)))

    found = {}
    for slot, P in zip(slots, row):
        above = [t for t in toks if t is not P and t["cy"] < P["cy"] and P["cy"] - t["cy"] < 3 * P["h"]
                 and t["x1"] > P["x0"] - P["w"] and t["x0"] < P["x1"] + 0.5 * P["w"]
                 and sum(c.isdigit() for c in t["t"]) >= 4]
        L = max(above, key=lambda t: t["cy"]) if above else None
        litres = None
        if L is not None:
            if NUM.match(L["t"]) and L["conf"] >= 0.9:
                litres = L["t"]
            else:
                litres = _reread(full, L, scale) or (L["t"] if NUM.match(L["t"]) else None)
        found[slot] = {"name": PUMP_LABELS[slot], "litres": float(litres) if litres else None,
                       "price": float(P["t"]), "raw": L["raw"] if L else ""}
    pumps = [found.get(i, {"name": PUMP_LABELS[i], "litres": None, "price": None, "raw": ""})
             for i in range(n)]

    stamp = None
    for l in lines:
        m = STAMP.search(l["text"].replace(" ", ""))
        if m:
            stamp = f"{m.group(1)} {m.group(2)}"
            break
    return {"reading_time": stamp, "pumps": pumps}


MANUAL = re.compile(r"^\s*(.+?)\s+([\d,]+\.\d{1,2})\s+(\d{2,3}(?:\.\d{1,2})?)\s*$")


def manual_readings(text: str) -> dict | None:
    """Typed correction, one pump per line: '<pump> <litres> <price>', e.g. 'Regular 65815.01 97.50'."""
    names = {n.lower(): n for n in PUMP_LABELS}
    pumps = []
    for line in text.strip().splitlines():
        m = MANUAL.match(line)
        if not m or m.group(1).strip().lower() not in names:
            return None
        pumps.append({"name": names[m.group(1).strip().lower()],
                      "litres": float(m.group(2).replace(",", "")), "price": float(m.group(3))})
    return {"reading_time": None, "pumps": pumps} if pumps else None


def format_readings(r: dict) -> str:
    if all(p["litres"] is None for p in r["pumps"]):
        prices = ", ".join(f"{p['name']} ₱{p['price']:.2f}" for p in r["pumps"] if p["price"])
        return (f"I can see the prices ({prices}) but not the meter totals, so nothing was saved.\n"
                "Show the totalizer (L) screen on each pump, then take the photo again.")
    out = [f"Reading: {r['reading_time']}"] if r["reading_time"] else []
    bad = []
    for p in r["pumps"]:
        if p["litres"] is None:
            out.append(f"{p['name']}: couldn't read the meter" + (f" (saw \"{p['raw']}\")" if p.get("raw") else ""))
            bad.append(p)
        else:
            out.append(f"{p['name']}: {p['litres']:,.2f} L @ ₱{p['price']:.2f}/L")
    if bad:
        out.append("\nNot saved for the pump(s) above. Retake the photo, or reply with the meter value:")
        out += [f"{p['name']} <litres> {p['price']:.2f}" if p["price"] else f"{p['name']} <litres> <price>" for p in bad]
    return "\n".join(out)


async def log_to_sheet(r: dict, msg: dict) -> str:
    """POST readings to the Google Apps Script web app. Returns a short status line for the reply."""
    if not SHEET_URL:
        return ""
    sender = msg.get("from") or {}
    r = {**r, "pumps": [p for p in r["pumps"] if p["litres"] is not None]}
    if not r["pumps"]:
        return ""
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

HELP = ("Send me a photo of the pump meters and I'll read and log them.\n"
        "To enter a reading by hand: Regular 65815.01 97.50 (one pump per line).\n"
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
        r = manual_readings(msg.get("text") or "")
        if r:
            await tg_reply(chat_id, format_readings(r) + await log_to_sheet(r, msg), mid)
        else:
            await tg_reply(chat_id, f"{HELP}\n\nYour chat ID: {chat_id}")
        return

    try:
        await tg("sendChatAction", chat_id=chat_id, action="typing")
        data = await tg_download(file_id)
        def work():
            lines, full, scale = ocr_image(data)
            return lines, pump_readings(lines, full, scale)

        lines, r = await run_in_threadpool(work)
        if r:
            reply = format_readings(r) + await log_to_sheet(r, msg)
        else:
            reply = "\n".join(l["text"] for l in lines) or "No text found."
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

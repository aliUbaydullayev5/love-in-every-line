import asyncio, io, json, os, shutil, tempfile, time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError

import pipeline

Image.MAX_IMAGE_PIXELS = 24_000_000
MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "8")) * 1024 * 1024)
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SEC", "90"))
RATE = int(os.environ.get("RATE_LIMIT_PER_MIN", "4"))
TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"
DOCS = os.environ.get("ENABLE_DOCS") == "1"
ALLOWED = {"JPEG", "PNG", "WEBP"}

app = FastAPI(docs_url="/api/docs" if DOCS else None, redoc_url=None, openapi_url="/api/openapi.json" if DOCS else None)
slots = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_JOBS", "1")))
hits: dict = defaultdict(deque)  # ponytail: in-memory, один процесс; Redis при нескольких воркерах


@app.exception_handler(HTTPException)
async def http_err(_, e: HTTPException):
    return JSONResponse({"detail": e.detail}, status_code=e.status_code)


@app.get("/api/health")
def health():
    if not pipeline.models_ok():
        return JSONResponse({"status": "models missing"}, status_code=503)
    return {"status": "ok"}


def client_ip(req: Request) -> str:
    if TRUST_PROXY:
        xff = req.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return req.client.host if req.client else "?"


def rate_limit(ip: str):
    now, q = time.monotonic(), hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "Слишком много запросов, попробуйте через минуту")
    q.append(now)


async def read_capped(f: UploadFile) -> bytes:
    buf = bytearray()
    while chunk := await f.read(256 * 1024):
        buf += chunk
        if len(buf) > MAX_BYTES:
            raise HTTPException(413, f"Файл больше {MAX_BYTES // 1024 // 1024} МБ")
    return bytes(buf)


def validate(data: bytes) -> str:
    """Проверка по реальному содержимому (не по расширению/Content-Type). Возвращает суффикс файла."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            if fmt not in ALLOWED:
                raise HTTPException(415, "Нужен JPG, PNG или WebP")
            if min(w, h) < 400:
                raise HTTPException(422, f"Разрешение {w}x{h} слишком мало (минимум 400 px по меньшей стороне)")
            if w > 6000 or h > 6000 or w * h > 24_000_000:
                raise HTTPException(422, "Разрешение слишком велико (максимум 6000x6000, 24 МП)")
            im.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError):
        raise HTTPException(415, "Файл не является корректным изображением JPG/PNG/WebP")
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[fmt]


@app.post("/api/draw")
async def draw(request: Request, file: UploadFile = File(...)):
    rate_limit(client_ip(request))
    data = await read_capped(file)
    suffix = validate(data)
    if slots.locked():
        raise HTTPException(503, "Сервер занят, попробуйте через минуту")
    async with slots:
        tmp = tempfile.mkdtemp(prefix="lie_req_")
        try:
            src, out = Path(tmp) / f"in{suffix}", Path(tmp) / "out.json"
            src.write_bytes(data)
            del data
            try:
                await asyncio.to_thread(pipeline.run, src, out, JOB_TIMEOUT)
            except TimeoutError:
                raise HTTPException(504, "Обработка заняла слишком много времени")
            except pipeline.PipelineError as e:
                raise HTTPException(422, str(e).removeprefix("Ошибка: "))
            return Response(out.read_bytes(), media_type="application/json")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

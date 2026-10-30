import io, json, sys, time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


@pytest.fixture(autouse=True)
def fresh_limiter():
    appmod.hits.clear()


def img_bytes(size=(500, 500), fmt="JPEG"):
    b = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(b, fmt)
    return b.getvalue()


def post(data, name="a.jpg", ctype="image/jpeg"):
    return client.post("/api/draw", files={"file": (name, data, ctype)})


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_text_as_jpg_rejected():
    r = post(b"not an image at all")
    assert r.status_code == 415 and "detail" in r.json()


def test_wrong_format_rejected():
    assert post(img_bytes(fmt="GIF"), "a.gif", "image/gif").status_code == 415


def test_too_small():
    assert post(img_bytes((100, 100))).status_code == 422


def test_too_large_resolution():
    b = io.BytesIO()
    Image.new("L", (7000, 500), 0).save(b, "PNG")
    assert post(b.getvalue(), "a.png", "image/png").status_code == 422


def test_oversize(monkeypatch):
    monkeypatch.setattr(appmod, "MAX_BYTES", 1000)
    assert post(img_bytes()).status_code == 413


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(appmod, "RATE", 2)
    codes = [post(b"x").status_code for _ in range(3)]
    assert codes == [415, 415, 429]


def test_end_to_end_real_generation(tmp_path):
    skimage = pytest.importorskip("skimage.data")
    p = tmp_path / "astro.jpg"
    Image.fromarray(skimage.astronaut()).save(p, quality=92)
    t = time.time()
    r = post(p.read_bytes())
    assert r.status_code == 200, r.text
    d = json.loads(r.content)
    assert d["version"] == 1 and d["width"] > 0 and d["height"] > 0 and len(d["strokes"]) > 1500
    for s in d["strokes"]:
        assert {"id", "cat", "order", "w", "o", "color", "dur", "pts"} <= s.keys()
        assert len(s["pts"]) >= 4 and len(s["pts"]) % 2 == 0
    print(f"\ngenerated {len(d['strokes'])} strokes in {time.time() - t:.1f}s")

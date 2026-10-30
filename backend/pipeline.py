"""Фото -> strokes.json: два подпроцесса (изоляция памяти, жёсткий таймаут). Фото нигде не сохраняется."""
import os, subprocess, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["face_landmarker.task", "selfie_segmenter.tflite", "informative_anime_512.onnx"]


class PipelineError(Exception):
    pass


def models_ok() -> bool:
    return all((ROOT / "tools" / "models" / m).is_file() for m in MODELS)


def _run(cmd, timeout):
    env = {**os.environ, "LIE_OFFLINE": "1"}
    try:
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # subprocess.run убивает процесс при таймауте
        raise TimeoutError("превышено время обработки")
    if r.returncode != 0:
        msg = (r.stdout + r.stderr).strip().splitlines()
        err = next((l for l in reversed(msg) if l.startswith("Ошибка")), "обработка не удалась")
        raise PipelineError(err)


def run(in_path, out_path, timeout=90):
    tmp = tempfile.mkdtemp(prefix="lie_")
    try:
        maps = str(Path(tmp) / "maps.npz")
        _run([sys.executable, "tools/make_strokes_v4.py", str(in_path), "--profile", "D2", "--out", str(Path(tmp) / "unused.json"), "--dump-maps", maps], timeout)
        _run([sys.executable, "tools/engine_e.py", maps, "--target", "sketch", "--sketch-model", "tools/models/informative_anime_512.onnx", "--out", str(out_path)], timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python backend/pipeline.py IN OUT")
    run(sys.argv[1], sys.argv[2], int(os.environ.get("JOB_TIMEOUT_SEC", "90")))
    print("готово:", sys.argv[2])

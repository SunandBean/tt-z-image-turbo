#!/usr/bin/env python3
"""Stage the release Z-Image service (in the locally built model image) on
127.0.0.1:20017 and drive it over HTTP exactly like the worker does. Same card hand-off contract as
run_device_check.py: control lock, stop worker then model, refuse other accelerator users, always restore.
Writes stage/result.json and one PNG per case."""
import base64
import fcntl
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
HOME = Path.home()
NAME = "zimage-stage"
RELEASE = json.loads((PROJECT / "deploy/model-release-zimage.json").read_text())
BASE = "http://127.0.0.1:20017"
OUT = ROOT / "stage"
OUT.mkdir(exist_ok=True)
result = {"status": "starting", "cases": [], "baseline_restored": False}
CASES = [
    ("teapot1024", "A croissant beside a cup of coffee, food photograph", 1024, 1024, 42),
    ("repeat1024", "A croissant beside a cup of coffee, food photograph", 1024, 1024, 42),
    ("fox576x1024", "Portrait of a red fox in fresh snow, wildlife photography", 576, 1024, 9),
    ("nook848x624", "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light", 848, 624, 7),
    ("korean512", "비 오는 밤 서울 골목의 따뜻한 조명, 수채화", 512, 512, 3),
    ("nook848x624-again", "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light", 848, 624, 7),
]


def save():
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


def run(*args, check=True, timeout=1000):
    return subprocess.run(args, check=check, text=True, capture_output=True, cwd=PROJECT, timeout=timeout)


def http(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_ready(url, bound=900):
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                body = json.loads(r.read())
            if body.get("status") == "ok":
                return body
            if body.get("status") == "error":
                raise RuntimeError(str(body))
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError("Model readiness exceeded bound")


def interrupted(signum, frame):
    raise InterruptedError(f"Stage interrupted by signal {signum}")


def main():
    from PIL import Image, ImageStat

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, interrupted)
    lock = (PROJECT / ".runtime/control.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = {svc: bool(run("docker", "compose", "ps", "--status", "running", "-q", svc).stdout.strip())
              for svc in ("model", "worker")}
    result["before"] = before
    save()
    try:
        if run("docker", "image", "inspect", RELEASE["tag"], "--format", "{{.Id}}").stdout.strip() != RELEASE["image_id"]:
            raise RuntimeError("Release tag does not point to the pinned image")
        run("docker", "compose", "stop", "-t", "900", "worker")
        run("docker", "compose", "stop", "-t", "60", "model")
        ids = run("docker", "ps", "-q").stdout.split()
        if ids:
            for item in json.loads(run("docker", "inspect", *ids).stdout):
                host = item["HostConfig"]
                if host.get("Privileged") or any("tenstorrent" in json.dumps(x) for x in (host.get("Devices") or [])):
                    raise RuntimeError("Another accelerator workload is active: " + item["Name"])
        run("docker", "rm", "-f", NAME, check=False)
        cache = PROJECT / "data/model-cache-zimage"
        cache.mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "create", "--name", NAME, "--ipc", "host", "--device", "/dev/tenstorrent",
               "-p", "127.0.0.1:20017:20000", "--log-opt", "max-size=30m", "--log-opt", "max-file=3"]
        for src, dst, ro in [(HOME / ".cache/huggingface", "/hf", True), (cache, "/cache", False),
                             (Path("/dev/hugepages-1G"), "/dev/hugepages-1G", False), (PROJECT / "deploy", "/service", True)]:
            cmd += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")]
        env = {"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TT_METAL_VISIBLE_DEVICES": "0",
               "MESH_DEVICE": "P100", "PYTHONUNBUFFERED": "1", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "90",
               "Z_IMAGE_SNAPSHOT": "/hf/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/" + RELEASE["weights_revision"],
               "PHOTO_MODEL_DEFAULT_SIZE": "1024"}
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [RELEASE["tag"], "python", "-m", "uvicorn", "zimage_service:app", "--app-dir", "/service",
                "--host", "0.0.0.0", "--port", "20000", "--workers", "1", "--lifespan", "on"]
        run(*cmd)
        t0 = time.monotonic()
        run("docker", "start", NAME)
        result["initial_health"] = wait_ready(BASE)
        result["startup_wall_s"] = time.monotonic() - t0
        result["info"] = http("GET", "/info")[1]
        save()
        pixels = {}
        for name, prompt, w, h, seed in CASES:
            started = time.monotonic()
            code, body = http("POST", "/predict", {"prompt": prompt, "seed": seed, "width": w, "height": h,
                                                   "num_steps": 9, "return_rgba": False})
            if code != 200:
                raise RuntimeError(f"{name}: HTTP {code} {body}")
            raw = base64.b64decode(body.pop("image"), validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                assert im.size == (w, h), im.size
                std = max(ImageStat.Stat(im.convert("RGB")).stddev)
                assert std > 3, "blank output"
                digest = hashlib.sha256(im.tobytes()).hexdigest()
            (OUT / f"{name}.png").write_bytes(raw)
            key = (prompt, w, h, seed)
            rec = {"name": name, "width": w, "height": h, "wall_s": time.monotonic() - started, "stddev": std,
                   "pixels_sha256": digest, "deterministic_repeat": pixels.get(key) == digest if key in pixels else None,
                   "model": body.get("model"), "license": body.get("license"), "timing_ms": body.get("timing_ms")}
            pixels.setdefault(key, digest)
            result["cases"].append(rec)
            save()
            print("PASS", name, round(rec["wall_s"], 2), flush=True)
        # contract checks: rejected requests must not take the model down
        code, _ = http("POST", "/predict", {"prompt": "x", "seed": 1, "width": 512, "height": 512, "num_steps": 9,
                                            "images": [base64.b64encode(raw).decode()]})
        result["reference_rejected_status"] = code
        code, _ = http("POST", "/predict", {"prompt": "x", "seed": 1, "width": 512, "height": 512, "num_steps": 40})
        result["wrong_steps_status"] = code
        result["health_after_rejections"] = http("GET", "/health")[1].get("status")
        assert result["reference_rejected_status"] == 422 and result["wrong_steps_status"] == 422
        assert result["health_after_rejections"] == "ok"
        assert all(c["deterministic_repeat"] in (None, True) for c in result["cases"])
        result["status"] = "pass"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        run("docker", "stop", "-t", "60", NAME, check=False, timeout=100)
        logs = run("docker", "logs", NAME, check=False)
        (OUT / "model.log").write_text(logs.stdout + logs.stderr)
        run("docker", "rm", "-f", NAME, check=False)
        try:
            if before["model"]:
                run("docker", "compose", "start", "model")
                result["baseline_health"] = wait_ready("http://127.0.0.1:20014")
            if before["worker"]:
                run("docker", "compose", "start", "worker")
            result["baseline_restored"] = True
        finally:
            save()
            lock.close()


if __name__ == "__main__":
    sys.exit(main())

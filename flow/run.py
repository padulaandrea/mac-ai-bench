"""Run the studio's real jobs on this Mac and time them. Started by ../macbench.sh (which installs everything into a
temporary folder); not meant to be run by hand.

The jobs (jobs.json) are the studio's own: the picture model FLUX.2-klein-4B through mlx-gen 0.38.0 and the video
model LTX-2.5 (8-bit) through ltx-2-mlx 0.18.0, with the exact prompts, reference pictures, seeds and settings of
approved Episode 1 takes. Models are pinned to the same revisions the studio uses, and only the files the studio's
flow reads are downloaded.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MODELS = [   # repo, revision the studio uses, files the studio's flow reads
    ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294",
     ["tokenizer/*", "transformer/*", "text_encoder/*", "vae/*"]),
    ("dgrauet/ltx-2.5-mlx-q8", "746ca9aacb697d2c739f544d68b584214dedcc75",
     ["*.json", "*.jinja", "transformer-distilled.safetensors", "text_encoder.safetensors", "connector.safetensors",
      "vae_encoder_*.safetensors", "vae_decoder_*.safetensors", "spatial_upscaler_x2_v1_0.safetensors",
      "vocoder.safetensors", "audio_vae.safetensors", "duration_head.safetensors"]),
]
UPLOAD = "https://dpaste.com/api/v2/"
NOTIFY = "https://ntfy.sh/mac-ai-bench-padulaandrea-results"   # the result link is also posted here (kept 12 hours)


def sh(*a) -> str:
    try:
        return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def machine() -> dict:
    gpu = sh("system_profiler", "SPDisplaysDataType")
    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "model": sh("sysctl", "-n", "hw.model"),
            "gpu_cores": next((l.split(":")[1].strip() for l in gpu.splitlines() if "Total Number of Cores" in l), "?"),
            "memory_gb": int(sh("sysctl", "-n", "hw.memsize") or 0) // 2**30,
            "cpu_cores": sh("sysctl", "-n", "hw.ncpu"), "macos": platform.mac_ver()[0]}


def memory() -> dict:
    vm = sh("vm_stat")
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    get = lambda k: int(next(l for l in vm.splitlines() if l.startswith(k)).split()[-1].rstrip(".")) * page
    used = get("Pages active") + get("Pages wired down") + get("Pages occupied by compressor")
    swap = sh("sysctl", "-n", "vm.swapusage")
    return {"used_gb": round(used / 2**30, 1), "swap_gb": round(float(swap.split("used = ")[1].split("M")[0]) / 1024, 2)}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.samples = []; self.stop = False

    def run(self):
        while not self.stop:
            self.samples.append(memory()); time.sleep(2)


def download(hf_home: Path, video: bool = True) -> dict:
    """Fetch the pinned model files (skips what is already there). Returns size and speed."""
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download
    out = {}
    for repo, rev, patterns in MODELS:
        if not video and "ltx" in repo:
            continue
        print(f"  {repo} ...", flush=True)
        t = time.time()
        path = Path(snapshot_download(repo, revision=rev, allow_patterns=patterns))
        # the tools ask for the repo by name (revision "main") and run offline: point "main" at the pinned revision
        ref = hf_home / "hub" / ("models--" + repo.replace("/", "--")) / "refs/main"
        ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(rev)
        gb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30
        sec = time.time() - t
        out[repo] = {"gb": round(gb, 1), "seconds": round(sec), "mb_per_s": round(gb * 1024 / max(sec, 1))}
        print(f"    {gb:.1f} GB in {sec / 60:.1f} min", flush=True)
    return out


def frame_at(mp4: Path, seconds: float):
    import av
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            if f.time is not None and f.time >= seconds:
                return f.to_image()


def difference(out: Path, expected: Path, at_s=None) -> float:
    """Mean pixel difference (0-255) between this Mac's result and the studio's approved result, at thumbnail size."""
    from PIL import Image, ImageChops, ImageStat
    exp = Image.open(expected).convert("RGB")
    img = frame_at(out, at_s) if at_s is not None else Image.open(out)
    img = img.convert("RGB").resize(exp.size, Image.LANCZOS)
    return round(sum(ImageStat.Stat(ImageChops.difference(img, exp)).mean) / 3, 1)


def run_job(job: dict, a, work: Path) -> dict:
    tool = a.mlxgen if job["tool"] == "mlxgen" else a.ltx
    args = [x.replace("{in}", str(a.kit / "inputs")).replace("{out}", str(work)) for x in job["args"]]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **job.get("env", {})}
    log = work / f"{job['name']}.log"
    m = Sampler(); m.start()
    t = time.time()
    with open(log, "w") as f:
        p = subprocess.run(["/usr/bin/caffeinate", "-i", "-s", "/usr/bin/time", "-l", tool, *args],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    sec = round(time.time() - t, 1)
    m.stop = True; m.join()
    text = log.read_text(errors="replace")
    peak = re.search(r"(\d+)\s+peak memory footprint", text)
    r = {"seconds": sec, "ok": p.returncode == 0 and (work / job["output"]).exists(),
         "peak_memory_gb": round(int(peak.group(1)) / 2**30, 1) if peak else None,
         "system_memory_peak_gb": max(x["used_gb"] for x in m.samples),
         "swap_grew_gb": round(max(x["swap_gb"] for x in m.samples) - m.samples[0]["swap_gb"], 2)}
    if r["ok"]:
        r["difference_from_studio"] = difference(work / job["output"], a.kit / job["expected"], job.get("expected_at_s"))
    else:
        r["error"] = text[-1500:]
    return r


def upload(res: dict, path: Path):
    want = os.environ.get("MACBENCH_UPLOAD")
    if want is None:      # ask, but upload anyway after 30 s so an unattended run still reports
        import select
        try:
            with open("/dev/tty") as tty:
                print("\nUpload these results to dpaste.com (no account; only the numbers above: Mac model, memory, "
                      "macOS version, timings) so they can be shared by link? [Y/n] (yes in 30 s) ", end="", flush=True)
                ready, _, _ = select.select([tty], [], [], 30)
                want = "0" if ready and tty.readline().strip().lower().startswith("n") else "1"
        except OSError:
            want = "1"
    if want != "1":
        return None
    title = (f"mac-ai-bench {res['machine']['chip']} {res['machine']['gpu_cores']}-core "
             f"{res['machine']['memory_gb']} GB")
    # the Mac's own curl (system certificates): Python's bundled ones reject dpaste.com's certificate chain
    r = subprocess.run(["/usr/bin/curl", "-sS", "--fail", "-m", "30", "-A", "mac-ai-bench", "-F", f"content=<{path}",
                        "-F", "syntax=json", "-F", "expiry_days=365", "-F", f"title={title}", UPLOAD],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip().startswith("https://"):
        link = r.stdout.strip()
        subprocess.run(["/usr/bin/curl", "-sS", "-m", "15", "-H", f"Title: {title}", "-d", link, NOTIFY],
                       capture_output=True)
        return link
    print(f"Upload failed ({(r.stderr or r.stdout).strip()[:200]}). The results are printed above and saved in {path}.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--hf-home", type=Path, required=True)
    ap.add_argument("--mlxgen", required=True)
    ap.add_argument("--ltx", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((a.kit / "jobs.json").read_text())["jobs"]
    ref = json.loads((a.kit / "reference.json").read_text()) if (a.kit / "reference.json").exists() else None
    res = {"benchmark": "mac-ai-bench studio flow", "date": time.strftime("%Y-%m-%d %H:%M"), "machine": machine(),
           "versions": {"mlx-gen": "0.38.0", "ltx-2-mlx": "0.18.0",
                        "models": {r: rev[:12] for r, rev, _ in MODELS}}}
    print("Machine:", json.dumps(res["machine"]), flush=True)
    print("\nDownloading the two models (only the files the studio uses)...", flush=True)
    video = any(j["tool"] == "ltx-2-mlx" for j in jobs if not a.only or j["name"] in a.only)
    res["download"] = download(a.hf_home, video)
    import static_ffmpeg
    static_ffmpeg.add_paths()      # the video tool writes its mp4 through ffmpeg, as in the studio
    res["jobs"] = {}
    for job in jobs:
        if a.only and job["name"] not in a.only:
            continue
        print(f"\n{job['name']}: {job['what']} ...", flush=True)
        r = res["jobs"][job["name"]] = run_job(job, a, a.work)
        base = (ref or {}).get("jobs", {}).get(job["name"], {}).get("seconds")
        if base and r["ok"]:
            r["speed_vs_reference"] = round(base / r["seconds"], 2)
        print(f"  {'done' if r['ok'] else 'FAILED'} in {r['seconds']} s" + (f"  ({r['speed_vs_reference']}x the reference Mac)"
              if r.get("speed_vs_reference") else "") + f", peak {r['peak_memory_gb']} GB", flush=True)
    j = res["jobs"]
    if all(j.get(k, {}).get("ok") for k in ("picture", "composed_scene", "composed_repaint", "video")):
        per_s = j["video"]["seconds"] / 5.04
        res["summary"] = {"start_picture_s": j["picture"]["seconds"],
                          "composed_picture_s": round(j["composed_scene"]["seconds"] + j["composed_repaint"]["seconds"], 1),
                          "video_per_clip_second_s": round(per_s, 1),
                          "video_5s_shot_min": round(per_s * 5 / 60, 2),
                          "video_100s_total_min": round(per_s * 100 / 60, 1)}
        if j.get("lipsync_video", {}).get("ok"):
            res["summary"]["lipsync_video_s"] = j["lipsync_video"]["seconds"]
    res.setdefault("summary", {}).update({
        "all_jobs_min": round(sum(r["seconds"] for r in j.values()) / 60, 1),
        "download_min": round(sum(d["seconds"] for d in res["download"].values()) / 60, 1),
        "download_gb": round(sum(d["gb"] for d in res["download"].values()), 1)})
    if ref:
        res["reference"] = {"machine": ref["machine"], "summary": ref.get("summary")}
    path = a.work / "results.json"
    path.write_text(json.dumps(res, indent=1))
    print("\n=== Results ===\n" + json.dumps({k: res[k] for k in ("machine", "download", "jobs", "summary") if k in res}, indent=1))
    if ref:
        print(f"\nReference Mac ({ref['machine']['chip']}, {ref['machine']['gpu_cores']}-core GPU, "
              f"{ref['machine']['memory_gb']} GB): " + json.dumps(ref.get("summary")))
    link = upload(res, path)
    if link:
        print(f"\nResults uploaded: {link}\nClaude can also find this link by itself (it is posted to the results channel).")
    keep = Path(os.environ.get("MACBENCH_SAVE", "")) if os.environ.get("MACBENCH_SAVE") else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True); shutil.copy2(path, keep / f"results_{time.strftime('%Y%m%d-%H%M%S')}.json")


if __name__ == "__main__":
    main()

#!/bin/sh
# Standalone Mac speed check for local AI picture + video generation (nothing installed permanently).
#
#   sh macbench.sh          (about 5 minutes, ~100 MB download)
#
# Installs a temporary Python + Apple MLX into one temp folder, times the same GPU maths as a local AI studio's picture model
# (FLUX.2-klein-4B) and video model (LTX-2.5 q8) at that studio's sizes, converts that into estimated real times using
# the reference Mac's measured times, prints a summary, and deletes the temp folder. Nothing is uploaded.
set -e
case "$(uname -m)" in arm64) ;; *) echo "This needs an Apple Silicon Mac (M1 or later)."; exit 1;; esac
D=$(mktemp -d /tmp/macbench.XXXXXX)
trap 'rm -rf "$D"' EXIT INT TERM
echo "Setting up a temporary Python + MLX in $D (deleted at the end)..."
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$D/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
export UV_CACHE_DIR="$D/cache" UV_PYTHON_INSTALL_DIR="$D/python"
"$D/bin/uv" run --quiet --python 3.12 --no-project --with "mlx==0.32.4" python - <<'PY'
import json, platform, subprocess, time
import mlx.core as mx

# reference: the studio's M5 Pro (MacBook Pro), measured with this same script and with the real models (2026-10-25)
REF = {"machine": "Apple M5 Pro, 20-core GPU, 48 GB",
       "picture_ms": 46.4, "video_ms": 251.0,                 # this script on the reference Mac (cool runs)
       "picture_s": 36.7, "video5s_s": 154.7, "composed_extra_s": 14.0}

def sh(*a):
    try: return subprocess.run(a, capture_output=True, text=True).stdout.strip()
    except Exception: return "?"

gpu = sh("system_profiler", "SPDisplaysDataType")
info = {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "gpu_cores": next((l.split(":")[1].strip() for l in gpu.splitlines() if "Total Number of Cores" in l), "?"),
        "memory_gb": int(sh("sysctl", "-n", "hw.memsize") or 0) // 2**30,
        "cpu_cores": sh("sysctl", "-n", "hw.ncpu"), "macos": platform.mac_ver()[0], "mlx": mx.__version__}
print("\nMachine:", json.dumps(info))

def bench(fn, seconds=3.0, rounds=2):
    """Best of `rounds` timed rounds (ms per call), after a warm-up (kernel compile, allocation)."""
    for _ in range(3): mx.eval(fn())
    best = 1e9
    for _ in range(rounds):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            mx.eval(fn()); n += 1
        best = min(best, (time.perf_counter() - t0) / n * 1000)
    return best

mx.random.seed(0)
r = {}
# picture model (FLUX.2-klein-4B): ~5,700 tokens (864x1536 image + prompt), width 3072, 24 heads, MLP 3x
xp = mx.random.normal((5700, 3072)).astype(mx.bfloat16)
wp = mx.random.normal((3072, 9216)).astype(mx.bfloat16) * 0.02
qp = mx.random.normal((1, 24, 5700, 128)).astype(mx.bfloat16)
r["picture_matmul_ms"] = bench(lambda: xp @ wp)
r["picture_attention_ms"] = bench(lambda: mx.fast.scaled_dot_product_attention(qp, qp, qp, scale=128 ** -0.5))
# video model (LTX-2.5, 8-bit weights): ~14,000 tokens (704x1280, 121 frames), width 4096, 32 heads, MLP 4x
xv = mx.random.normal((14080, 4096)).astype(mx.float16)
wv = mx.random.normal((16384, 4096)).astype(mx.float16) * 0.02
wq, sc, bi = mx.quantize(wv, group_size=64, bits=8)
qv = mx.random.normal((1, 32, 14080, 128)).astype(mx.float16)
r["video_matmul_q8_ms"] = bench(lambda: mx.quantized_matmul(xv, wq, sc, bi, transpose=True, group_size=64, bits=8))
r["video_attention_ms"] = bench(lambda: mx.fast.scaled_dot_product_attention(qv, qv, qv, scale=128 ** -0.5))
# image decoder (VAE), used at the end of both models: a 3x3 convolution, 128 channels, at full 864x1536 resolution
img = mx.random.normal((1, 1536, 864, 128)).astype(mx.bfloat16)
kern = mx.random.normal((128, 3, 3, 128)).astype(mx.bfloat16) * 0.02
r["decoder_conv_ms"] = bench(lambda: mx.conv2d(img, kern, padding=1))
del img, kern
# memory speed: read 1 GB and write 1 GB
a = mx.ones((256 * 1024 * 1024,), dtype=mx.float32)
ms = bench(lambda: a * 1.0001)
r["memory_gb_per_s"] = round(2 * a.nbytes / 2**30 / (ms / 1000), 1)
del xp, wp, qp, xv, wv, wq, sc, bi, qv, a

# per-block cost of each model (one transformer block = attention + MLP-like matmuls), used to scale real times
picture_ms = r["picture_matmul_ms"] * 3 + r["picture_attention_ms"]
video_ms = r["video_matmul_q8_ms"] * 2 + r["video_attention_ms"]
r = {k: round(v, 2) for k, v in r.items()}
print("Measured:", json.dumps({**r, "picture_block_ms": round(picture_ms, 2), "video_block_ms": round(video_ms, 2)}))

if REF["picture_ms"]:
    fp, fv = REF["picture_ms"] / picture_ms, REF["video_ms"] / video_ms
    pic, vid5 = REF["picture_s"] / fp, REF["video5s_s"] / fv
    est = {"picture_s": round(pic, 1), "composed_picture_s": round(pic + REF["composed_extra_s"] / fp, 1),
           "video_per_clip_second_s": round(vid5 / 5, 1), "video_5s_shot_min": round(vid5 / 60, 2),
           "video_100s_total_min": round(vid5 / 5 * 100 / 60, 1)}
    print(f"\n=== Compared with the {REF['machine']} ===")
    print(f"Picture work: {fp:.2f}x the speed   Video work: {fv:.2f}x the speed")
    print("Estimated real times on this Mac (about +-10%):", json.dumps(est))
    print(f"Reference Mac, real: picture {REF['picture_s']} s, 5-second video {REF['video5s_s'] / 60:.2f} min")
print("\nDone. Take a photo of the lines above. The temporary folder is now deleted.")
PY

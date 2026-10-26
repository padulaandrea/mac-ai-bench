# mac-ai-bench

Times a **real** local AI animation flow on an Apple Silicon Mac: the same two models, tool versions, prompts,
reference pictures, seeds and settings a working studio uses to make a kids' cartoon. Made to decide between Macs
(for example an M5 Pro and an M5 Max) on the actual work, not on a generic GPU score.

## Run it

Open **Terminal** and paste:

```sh
curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/macbench.sh | sh
```

On a slow connection, pictures only (15 GB download instead of 57 GB; the three picture jobs):

```sh
curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/macbench.sh | sh -s pictures
```

Or, from a copy of this folder (for example via AirDrop): `sh ~/Downloads/mac-ai-bench/macbench.sh`

At the end it asks whether to upload the results. Say yes, and send the link it prints.

- Needs an Apple Silicon Mac, **about 65 GB free**, and a fast connection: it downloads about **57 GB** of models
  (FLUX.2-klein-4B 15 GB, LTX-2.5 8-bit 42 GB). On fast Wi-Fi (~50 MB/s) that is about 20 minutes; the jobs then take
  about 8 minutes on an M5 Pro.
- Installs nothing permanently: Python, the tools, the models and the studio's pictures go into one temporary folder,
  deleted at the end (also on Ctrl-C or when the Terminal window is closed). If a run is ever cut off (crash,
  force-quit), this removes any leftovers and says what it deleted:
  `curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/cleanup.sh | sh`
- The upload (optional) goes to [dpaste.com](https://dpaste.com) with no account, kept for a year and unlisted:
  only the Mac model, memory, macOS version, download speed and timings. No names, no serial numbers, no files.

## What it runs

Five jobs taken from approved takes of the studio's first episode (`flow/jobs.json`):

| Job | Model | What it is |
|---|---|---|
| picture | FLUX.2-klein-4B (mlx-gen 0.38.0) | a start picture, 864x1536, 2 reference pictures |
| composed_scene | FLUX.2-klein-4B | a scene with 3 reference pictures (step 1 of a composed picture) |
| composed_repaint | FLUX.2-klein-4B + a character LoRA | a 768x768 masked repaint (step 2) |
| video | LTX-2.5 8-bit (ltx-2-mlx 0.18.0) | 5-second clip from a start picture, 704x1280, 24 fps, distilled 8+2 steps, fp16, low-RAM |
| lipsync_video | LTX-2.5 8-bit | 5.4-second clip whose mouths follow a voice track (audio-to-video) |

Each job runs as a fresh process, as in the studio, so model loading is included. For each one it records the time,
the peak memory, swap growth, and how close the result is to the studio's approved result (mean pixel difference at
thumbnail size, 0-255: a small number means this Mac made the same picture).

Models are pinned to the studio's exact revisions, and only the files the flow reads are downloaded.

## Results

Each run's full numbers are in [`results/`](results).

| Mac | Start picture | Composed picture | 5 s video | 5.4 s lip-sync video | Download |
|---|---|---|---|---|---|
| M5 Pro, 20-core GPU, 48 GB (reference, 2026-10-26, run 2) | 40 s | 72 s | 2.7 min | 2.7 min | (already there) |
| M5 Pro, 20-core GPU, 48 GB (run 1) | 42 s | 73 s | 2.7 min | 2.8 min | (already there) |

On the reference Mac the pictures came out pixel-identical to the studio's approved ones (difference 0.0) and the
videos nearly identical (1.1 and 1.2 out of 255).

## Quick check (no model download)

`quick.sh` is a 5-minute estimate without the models: it times the same kinds of GPU maths at the same sizes and
scales the reference Mac's real times (about +-10%). Use it when there is no time or space for the real run.

```sh
curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/quick.sh | sh
```

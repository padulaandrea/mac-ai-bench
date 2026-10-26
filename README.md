# mac-ai-bench

A 5-minute speed check for Apple Silicon Macs, for local AI **picture** and **video** generation with
[MLX](https://github.com/ml-explore/mlx). Made to compare Macs (for example an M5 Pro and an M5 Max) without
downloading tens of gigabytes of models.

## Run it

Open **Terminal** and paste:

```sh
curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/macbench.sh | sh
```

Then take a photo of the last lines.

- Needs an Apple Silicon Mac (M1 or later) and an internet connection (~100 MB download).
- Installs nothing permanently: a temporary Python and MLX go into one folder in `/tmp`, deleted at the end.
- Uploads nothing: results are only printed.

## What it measures

The GPU work two real models spend their time on, at real sizes:

| Test | Stands for |
|---|---|
| Matrix multiply, 5,700 x 3,072 x 9,216 (bfloat16) | FLUX.2-klein-4B picture model, 864x1536 image |
| Attention, 24 heads x 5,700 tokens | the same picture model |
| 8-bit matrix multiply, 14,080 x 4,096 x 16,384 | LTX-2.5 video model (q8), 704x1280, 5 seconds |
| Attention, 32 heads x 14,080 tokens | the same video model |
| 3x3 convolution, 128 channels at 864x1536 | the image decoder (VAE) both models end with |
| Read + write 1 GB | memory speed |

It then compares the result with a reference Mac where the real models were timed (M5 Pro, 20-core GPU, 48 GB:
picture 36.7 s, 5-second video 2.58 min) and prints estimated real times for the Mac it runs on (about +-10%).

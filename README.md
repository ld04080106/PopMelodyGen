---
title: PopMelodyGen
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.35.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# PopMelodyGen

PopMelodyGen is a reproducible system for controllable symbolic pop-melody generation.
It combines a rule-based expert synthesizer, synthetic-data pretraining, and an
interactive Gradio demo for melody generation, playback, and MIDI export.

This public release focuses on the synthetic-data pipeline and demo:

- `01_generate_synthetic_data.py`: generate synthetic melody MIDI and metadata
- `02_preprocess_synthetic_data.py`: tokenize synthetic data and create train/eval/test splits
- `03_train_synth_base.py`: train or resume the synthetic-pretrained Transformer
- `08_launch_demo.py`: launch the interactive demo with Expert, Synth-Base, and Real-FT systems
- `app.py`: Hugging Face Spaces entry point for the same demo

The RWC real-pop processing, finetuning, generation/evaluation scripts, and paper
experiment utilities are intentionally not part of this initial code release. They
will be released after paper acceptance.

## Features

- Rule-based symbolic pop-melody synthesis
- Explicit musical controls:
  - tempo class
  - key and mode
  - vocal register
  - song structure
  - chord progression
- MIDI plus JSON metadata export
- Event-based tokenizer for symbolic melody modeling
- Decoder-only Transformer training
- Optional checkpoint initialization and resume training
- Interactive Gradio demo with melody visualization, playback, accompaniment, and MIDI export

## System Overview

```text
Stage 1: Synthetic melody generation
musical conditions -> expert synthesizer -> MIDI + metadata

Stage 2: Synthetic preprocessing
MIDI + metadata -> tokenizer -> event sequences + meta.jsonl

Stage 3: Synthetic pretraining
event sequences -> decoder-only Transformer -> checkpoint

Stage 4: Interactive generation
user controls -> Expert / Synth-Base / Real-FT -> MIDI + playback
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA environment from
the official PyTorch instructions before running training.

## Generate Synthetic Data

Generate one tempo class (slow/mid/fast):

```bash
python 01_generate_synthetic_data.py --config configs/expert_mid.yaml
```

Generate all tempo classes in one run:

```bash
python 01_generate_synthetic_data.py --config_dir configs --tempos SLOW MID FAST
```

By default, generated files are written to `_data_synth/`. Each sample contains a
`.mid` file and a matching `.json` metadata file.

## Preprocess Synthetic Data

```bash
python 02_preprocess_synthetic_data.py --in_dir _data_synth --out_dir _preprocessed_synth
```

The script tokenizes each MIDI/metadata pair and writes:

- `_preprocessed_synth/meta.jsonl`
- one `.npy` token-id file per sample
- one `.txt` token dump per sample
- `_preprocessed_synth/preprocess_report.json`

The train/eval/test split is computed independently within each tempo class, so
SLOW, MID, and FAST are all represented in each split when enough samples exist.

## Train Synth-Base

```bash
python 03_train_synth_base.py --config configs/train_synth_base_100k.yaml
```

Checkpoints are written under the configured `out_dir`; `best.pt` is used by the
demo as the default Synth-Base checkpoint.

## Launch Demo

Local launch:

```bash
python 08_launch_demo.py --base_ckpt _ckpt_synth_base_100k/best.pt --finetune_ckpt _ckpt_finetune_100k_mix2to1/best.pt
```

Open the printed local URL in a browser. Add `--share` if you want Gradio to create
a public temporary URL.

For Hugging Face Spaces, `app.py` loads `08_launch_demo.py` and exposes the same
Gradio app. Upload checkpoints to the expected paths, or set:

```bash
POPMELODYGEN_BASE_CKPT=/path/to/base/best.pt
POPMELODYGEN_FINETUNE_CKPT=/path/to/finetune/best.pt
```

If a checkpoint is missing, the demo still starts and reports the loading error on
the page; the rule-based Expert system remains available.

## Repository Layout

```text
configs/         YAML configs for synthesis and training
expert_melody/   rule-based melody synthesizer
expert_acc/      lightweight accompaniment generator
model/           tokenizer, dataset, Transformer, and generation utilities
utils/           MIDI and accompaniment helpers
```

Generated data, checkpoints, evaluation outputs, RWC-derived files, and paper
working files are ignored by default. See `.gitignore` for the release boundary.

## License

MIT License. See `LICENSE`.

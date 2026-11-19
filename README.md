# PopMelodyGen

**A Copyright-Aware Platform for Controllable Pop-Melody Generation**

**Accepted to the AAAI-27 Demonstration Program.**

PopMelodyGen is a copyright-aware platform for controllable symbolic pop-melody generation. It combines a rule-based expert synthesizer, synthetic-data pretraining, and adaptation with a small licensed real-pop corpus. The project supports interactive melody creation for **creators** and batch construction of labeled symbolic datasets for **researchers**.

- **AAAI-27 demo video:** https://youtu.be/U_R1MEZFOjI
- **Paper:** *PopMelodyGen: A Copyright-Aware Platform for Controllable Pop-Melody Generation* (AAAI-27 Demonstration Program; proceedings link forthcoming)
- **Code:** this repository

> **Important: this repository is a partial research-code release, not the complete deployed platform.**
> It currently contains the synthetic-data pipeline, Synth-Base training code, and a **local Python/Gradio demo**. The production web interface described in the AAAI-27 paper uses **Node.js and a backend inference API on a dedicated server**; that production frontend is **not included here**. The paper's Researcher Mode is part of the deployed system, not a documented feature of the Gradio demo in this repository.

## System overview

The paper describes four connected stages:

1. **Expert:** rule-based synthesis of melodies and structured annotations from general musical rules and curated templates.
2. **Synth-Base:** a decoder-only Transformer pretrained on rule-generated synthetic sequences.
3. **Real-FT:** the pretrained model adapted with licensed RWC real-pop data and synthetic replay.
4. **Interactive platform:** model selection, configurable musical conditions, playback, and export in Creator Mode; batch labeled dataset generation in Researcher Mode.

The synthetic pretraining corpus contains 100,000 examples; the paper reports adaptation using 847 tokenized examples obtained from 98 prepared RWC melodies after preprocessing and augmentation. The RWC corpus and adapted checkpoints are **not bundled with this repository**.

## What is publicly available here

| Component | Current public release |
| --- | --- |
| Expert melody synthesizer | Included |
| Synthetic MIDI + JSON metadata generation | Included |
| Synthetic preprocessing and tokenization | Included |
| Synth-Base training | Included |
| Local Gradio demonstration | Included |
| RWC processing and Real-FT fine-tuning scripts | Not included in this release |
| Full paper evaluation and similarity-analysis scripts | Not included in this release |
| Production Node.js web interface and Researcher Mode | Not included in this repository |
| Neural model checkpoints | Not bundled; provide separately to enable neural backends |

This table describes the current repository state; it is not a commitment that missing components or data can be released under their respective licenses.

## Features of the local demo

- Select among **Expert**, **Synth-Base** (checkpoint required), and **Real-FT** (checkpoint required).
- Control tempo class, key/mode, vocal register, song structure, and chord progression.
- Generate symbolic melodies with accompaniment for audition.
- Export multi-track MIDI.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA environment before running the training script.

## Model checkpoints

The local demo can run **Expert** without a neural checkpoint. To use the two neural backends, supply compatible checkpoints at the following default paths, or override them using the CLI flags below:

```text
_ckpt_synth_base_100k/best.pt
_ckpt_finetune_100k_mix2to1/best.pt
```

These paths are configuration conventions, **not download links**. Model availability and redistribution terms must be checked separately.

## Generate synthetic data

Generate one tempo class (slow, mid, or fast):

```bash
python 01_generate_synthetic_data.py --config configs/expert_mid.yaml
```

Generate all three tempo classes:

```bash
python 01_generate_synthetic_data.py --config_dir configs --tempos SLOW MID FAST
```

Generated files are written to `_data_synth/` by default, with a `.mid` and matching `.json` file for each example.

## Preprocess synthetic data

```bash
python 02_preprocess_synthetic_data.py --in_dir _data_synth --out_dir _preprocessed_synth
```

The script writes tokenized examples and metadata, including:

- `_preprocessed_synth/meta.jsonl`
- A `.npy` token-ID file and a `.txt` token dump per sample
- `_preprocessed_synth/preprocess_report.json`

Train/evaluation/test splits are computed within each tempo class when enough examples are available.

## Train Synth-Base

```bash
python 03_train_synth_base.py --config configs/train_synth_base_100k.yaml
```

Checkpoints are written to the configured output directory. The default local demo expects the compatible `best.pt` checkpoint.

## Launch the local Gradio demo

```bash
python 08_launch_demo.py --base_ckpt _ckpt_synth_base_100k/best.pt --finetune_ckpt _ckpt_finetune_100k_mix2to1/best.pt
```

Open the locally printed URL. Optionally supply `--share` to request a temporary Gradio share URL. If a checkpoint is missing, its neural backend will be unavailable; Expert can still run.

## Repository layout

```text
01_generate_synthetic_data.py     Generate rule-based synthetic MIDI and metadata
02_preprocess_synthetic_data.py  Tokenize data and prepare splits
03_train_synth_base.py           Train Synth-Base
08_launch_demo.py                Launch the local Gradio demo
configs/                         Configuration files
expert_melody/                   Expert melody synthesizer
expert_acc/                      Accompaniment generator
model/                           Tokenizer, dataset, neural modeling, and generation utilities
utils/                           MIDI and accompaniment helpers
```

## Copyright-aware design and data licensing

The expert synthesizer constructs melodies from general musical rules and element-level templates rather than transforming existing pop melodies. The Real-FT training stage described in the paper uses licensed RWC data; the 2026 RWC release is distributed under **CC BY-NC 4.0**. The RWC data are not redistributed in this repository.

**Copyright-aware does not mean legally copyright-safe.** Generated music may require additional review before redistribution or commercial use. The paper's nearest-neighbor analysis checks for possible near-copy risks against the selected RWC reference set; it is not a legal guarantee.

## License

The **repository code** is released under the [MIT License](LICENSE). This does **not** grant rights to third-party datasets, separately distributed model weights, or generated outputs beyond applicable licenses and law.

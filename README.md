# 🎵 PopMelodyGen

PopMelodyGen is a **copyright-safe symbolic pop melody synthesis toolkit**.

## Main Features
- 🎵 Fully copyright-safe data synthesis pipeline
- 🎛️ Fine-grained conditional control (key/mode, tempo, harmony, vocal register, structure.)
- 📊 Aesthetic-score–based supervised for higher-quality training data
- 🎹 REMI-based music tokenizer
- 🧠 Decoder-only Transformer architecture
- 🌐 Easy-to-deploy web demo interface

## Release
This early release open-sources **Step 01–03 only**:

- ✅ **01** Expert synthesizer → large-scale MIDI + metadata  
- ✅ **02** MIDI → WAVE rendering with FluidSynth + SoundFont
- ✅ **03** Aesthetic scoring using [**SongEval**](https://github.com/ASLP-lab/SongEval)

🚧 **Training + demo code (Step 04+) will be released after ACL demo acceptance.**

---
## ✨ What’s Included

### ✅ Step 01 — Expert Melody Synthesis
Generate large-scale symbolic pop melodies using the **rule-based expert synthesizer**, with rich metadata.

Outputs:
- `.mid` MIDI files
- `.json` metadata files

### ✅ Step 02 — MIDI → Audio Rendering
Render MIDI to WAV using **FluidSynth** + **SoundFont (.sf2)** for step 03.

Outputs:
- `.wav` WAVE files

### ✅ Step 03 — Aesthetic Scoring
Batch-score generated melodies using [**SongEval**](https://github.com/ASLP-lab/SongEval).

---

## 📦 Installation

```bash
conda create -n popmelodygen python=3.11
conda activate popmelodygen

# install PyTorch (choose the correct build for your machine if needed)
conda install pytorch torchaudio -c pytorch

pip install -r requirements.txt
```

## 🔌 Third-party Dependencies

**1. Install [FluidSynth](https://www.fluidsynth.org/) (for Step 02)**

```bash
sudo apt-get update
sudo apt-get install fluidsynth
```

**2. Download SoundFont (.sf2)**

Download SoundFont file and place it at

```
PopMelodyGen/render/sf2/xxx.sf2
```

In our paper, we used `FluidR3_GM.sf2` which can be found [here](https://member.keymusician.com/Member/FluidR3_GM/).

**3. Install SongEval (for Step 03)**

Clone SongEval into the repository root:

```bash
git clone https://github.com/ASLP-lab/SongEval.git SongEval
```

> ⚠️ SongEval is a third-party project. Please comply with its license and weight distribution policy.

## 🚀 Usage (Step 01–03)

**🧪 Step 01 — Generate Expert-Synthesized MIDI Dataset**

```bash
python 01_batch_expert_syn.py --config configs/expert_slow.yaml
python 01_batch_expert_syn.py --config configs/expert_mid.yaml
python 01_batch_expert_syn.py --config configs/expert_fast.yaml
```
✅ Note: each tempo class is geberated independly.

---

**🔊 Step 02 — Render MIDI → WAVE**

1. Set the following fields in `configs/expert_render.yaml`:

* fluidsynth_bin: path to fluidsynth executable
* path_sf2: path to your .sf2 file
* in_dir: directory of MIDI outputs from Step 01
* out_dir: output directory for WAV files

2. Run:

```bash
python 02_batch_render_audio.py --config configs/expert_render.yaml
```

---

**⭐ Step 03 — Aesthetic Scoring with SongEval**

1. Ensure:

* `SongEval/` exists in repo root
* `SongEval/eval.py` exists

2. Run:

```bash
python 03_batch_eval.py --in_dir _dataset
```

This script calls `SongEval/eval.py` internally and writes batch scores for all MIDI files in `--in_dir`.

---

**🧩 Planned Release (After ACL Demo)**
🚧 The following components are not included in this early release:

* Step 04: preprocessing (meta.jsonl, train/val/test split)
* Step 05: training (decoder-only Transformer)
* Step 06: generation with trained checkpoints
* Demo UI (Gradio)

---

**📜 License**

* Code in this repository: MIT
* Third-party: SongEval follows its own license (see [`SongEval`](https://github.com/ASLP-lab/SongEval)).

---

**📌 Citation**

If you use this code, please cite our paper: 
```bibtex
Coming soon.
```

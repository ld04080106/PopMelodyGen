#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_batch_render_audio.py

Batch render MIDI files to WAV audio using FluidSynth + an SF2 soundfont.

Prerequisites:
  1) Install FluidSynth (system dependency)[https://www.fluidsynth.org/]
  2) Download an SF2 soundfont (FluidR3_GM.sf2)[https://member.keymusician.com/Member/FluidR3_GM/]
  3) Set paths in `configs/expert_render.yaml`:
       - fluidsynth_bin: path to the FluidSynth executable
       - soundfont_path: path to the .sf2 file
       - midi_dir: path to the dataset folder produced by Step—01
       - out_dir: directory to write rendered .wav files

What this script does:
  - Loads `expert_render.yaml`.
  - Scans `midi_dir` for MIDI files generated in Step 01.
  - Invokes FluidSynth in batch mode to render each MIDI into a WAV file.

Inputs:
  - --config: YAML config containing `fluidsynth_bin`, `path_sf2`, `in_dir`, and `out_dir`.

Outputs:
  - One `.wav` file per MIDI under `out_dir` (same base filename).

Example:
  python 02_batch_render_audio.py --config configs/expert_render.yaml

Notes:
  - This step is required for aesthetic scoring.

Author:      <ludy>
Modified:    2026/01/22
Licence:     <MIT>
"""

import os
import random
import json
import time
import pretty_midi
import argparse
import yaml
from datetime import date, datetime

from render.renderer import render_batch

##-----------------------------
# Params
##-----------------------------
SKIP_EXISTING = True  # bool

##-----------------------------
# Logger
##-----------------------------
def _log(s):
    print(s)

## =========================
# batch render
## =========================
def render_dataset(cfg):
    """
      Batch render all .mid -> .wav
    """
    # read params
    in_dir = cfg["in_dir"]
    out_dir = cfg["out_dir"]

    # check input dir
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Error: input dir not found ({in_dir})")

    _log("----------------- Parameters ----------------- ")
    _log(f" > Input dir: {in_dir}") 
    _log(f" > Output dir: {out_dir}") 

    ##############################
    ## Render                   ##
    ############################## 
    _log("----------------- Start render ----------------- ")
    report = render_batch(in_dir, out_dir, cfg,
                          pattern="*.mid",
                          skip_existing=SKIP_EXISTING
                          )

    _log(f"Total: {report['count_total']}, OK: {report['count_ok']}, Failed: {report['count_failed']}")
    if report["count_failed"] > 0:
        _log("WARNING: Some files failed. Please check the report.")

    # save report 
    _log(f"-----------------Writing report ----------------- ")
    today = date.today()
    t = int(time.time())
    path_report = os.path.join(out_dir, f"_log_render_{today}_{t}.json")
    with open(path_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def load_config(path_yaml):
    with open(path_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Error: YAML root must be a mapping/dict: {path_yaml}")
    return data

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/expert_render.yaml")
    return p.parse_args()

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()

    print(f"> Loaded config : {args.config}")

    cfg = load_config(args.config)

    render_dataset(cfg)





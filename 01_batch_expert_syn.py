#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_batch_expert_syn.py

Synthesize a large-scale symbolic pop-melody dataset using the rule-based expert synthesizer.

This module enumerates global conditions (key × vocal_register) and generates multiple songs
per condition. For each song, the synthesizer randomly samples tempo, structure, and chord
progression templates, then outputs both the MIDI and a metadata JSON file.

Inputs:
  - --config: Path to a YAML config file specifying synthesis options and output directory.
    In particular, `N` (songs per (key, vocal_register)) is defined in the config.

Outputs (written to the output directory specified in the config):
  - MIDI files (.mid), e.g.:
      30001_MID_female_key0_mode0_1_melody.mid
  - Metadata files (.json), e.g.:
      30001_MID_female_key0_mode0_1_melody.json

Generation order:
  1) Enumerate `key` (12 keys) and `vocal_register` (2 registers) as global condition.
  2) For each (key, vocal_register), generate `N` songs (N is defined in cfg).
     For each song, sample:
       - tempo (random)
       - structure template (random)
       - chord progression template(s) (random)
       - rhythm template (random)
     as song condition.
  3) Generate additional `n_melody_per_cond` candidates under the same condition.
     Optionally, set n_melody_per_cond>1 for comparing multiple synthesized melodies with identical global settings.

Notes:
  - `N` controls dataset scale per (key, vocal_register).
  - `n_melody_per_cond` controls the number of candidates per condition for comparison;
     set it to 1 for standard dataset synthesis, or >1 when you want multiple candidates.

Example:
  python 01_batch_expert_syn.py --config configs/expert_slow.yaml

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

from expert_melody.pop_settings import Tempo
from expert_melody.pop_expert import ExpertMelodyGenerator
from utils.midi import melody_to_pretty_midi

##----------------------------------------
# Global parameters
##----------------------------------------

# Number of songs to generate for each fixed condition (key × vocal_register × tempo × structure × chord_progression).
# Set to 1 for standard dataset synthesis; 
# set >1 to generate multiple candidates under identical conditions.
N_RESULT_PER_CON = 1
 
# Melody instrument used while exporting melody.mid
# MIDI program: 0=Piano; 73=Flute; 85=LeadVocal   
PROGRAM_NO = 85

# Pitch offset while exporting melody.mid
# default = 0
# You can +12 to adjust melody pitch register from singing-voice to instrument(flute), 
# which may earn higher eval-score at step3(aesthetic scoring) after rendering.
# But if you +12, DO remember to -12 while preprocessing at preprocess before training.
MIDI_PITCH_OFF = 12

##----------------------------------------
# Logger
##----------------------------------------
def _log(s):
    print(s)

## ===================================
# batch syn
## ===================================
def generate_melody_dataset(cfg):
    """Generate expert-synthesized melodies under enumerated global conditions.

    Tempo_class is specified before generation.
    Then, the function enumerates:
      - key: 12 keys
      - vocal_register: 2 registers (Male/Female)
    For each (key, vocal_register), it generates `N` songs where `N` is defined in `cfg`.
    Each song is synthesized by randomly sampling tempo, structure, and chord-progression
    templates, then saving:
      (1) a MIDI file and (2) a JSON metadata file.

    Optionally, the function can generate multiple candidates (`n_melody_per_cond`) under the
    same condition(key/vocal_register/tempo/structure/chord_progression/rhythm) to support side-by-side comparison.

    Args:
        cfg: Parsed configuration object or dict loaded from a YAML file. Must include:
            - output directory path
            - `N` (songs per (key, vocal_register))
            - seed/tempo_class/start_id

    Returns:
        None. Writes `.mid` and `.json` files to disk.

    Side Effects:
        Creates output directories if missing and writes dataset artifacts.
        No overwrite checks are made.

    Example:
        >>> cfg = load_config("configs/expert_slow.yaml")
        >>> generate_melody_dataset(cfg, n_try_per_cond=1)
    """
    # read params
    base_seed = cfg["seed"]
    out_dir = cfg["out_dir"]
    n_songs = cfg["n_songs"]
    tempo_class= Tempo._value2member_map_[cfg["tempo"]]
    start_id = cfg["start_id"]

    # out dir
    os.makedirs(out_dir, exist_ok=True)

    _log("----------------- Parameters ----------------- ")
    _log(f" > output dir: {out_dir}")
    _log(f" > N songs (For each (key, vocal_register)): {n_songs}")
    _log(f" > N song(s) under same condition: {N_RESULT_PER_CON}")
    _log(f" > tempo_class: {tempo_class.value}")
    _log(f" > seed: {base_seed}")
    _log(f" > start_id: {start_id}")
    _log(f"--------------------------------------------- ")

    # foreach male/female, key0~11, mode0only
    cnter = 0
    for g, gender in enumerate(["male", "female"]):
        for key in range(12):
            for mode in [0,]:
                for i in range(n_songs):
                    song_id = f"{start_id + i + 1:05d}"
                    cur_seed = (start_id + base_seed + i)*10000 + key*100 + (g+8)*10
                    seed_global = cur_seed
                    seed_rhythm = cur_seed
                    seed_pitch = cur_seed

                    _log(f" > {song_id} : Start @ {datetime.now()}")

                    # synthesize N songs under same condition for side-by-side comparison
                    for j in range(N_RESULT_PER_CON):
                        gen = ExpertMelodyGenerator(
                                    tempo_class=tempo_class,
                                    gender=gender,
                                    key=key,
                                    mode=mode,
                                    seed_global=seed_global, 
                                    seed_rhythm=seed_rhythm,   # + j to compare rhythm
                                    seed_pitch=seed_pitch+j,   # + j to compare pitch 
                                    _log=_log,
                                )
                        melody_seq, song_meta = gen.generate_song()

                        # add song_id into meta
                        parts = song_id.split('_')
                        new_song_id = '_'.join([song_id, tempo_class.value, gender, f"key{key}", f"mode{mode}", str(j+1), "melody"])
                        song_meta["song_id"] = new_song_id

                        # export MIDI
                        pm = melody_to_pretty_midi(melody_seq, song_meta, program=PROGRAM_NO, midi_pitch_offset=MIDI_PITCH_OFF)
                        path_midi = os.path.join(out_dir, f"{new_song_id}.mid")
                        pm.write(path_midi)

                        # export JSON
                        path_json = os.path.join(out_dir, f"{new_song_id}.json")
                        with open(path_json, "w", encoding="utf-8") as f:
                            json.dump(song_meta, f, ensure_ascii=False, indent=2)

                        _log(f" > {new_song_id} : Generated @ {datetime.now()}")

                        # update
                        cnter += 1

    # end
    _log(f"----------------- Done @ {datetime.now()}----------------- ")
    _log(f" Total count = {cnter}")


def load_config(path_yaml):
    with open(path_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Error: YAML root must be a mapping/dict: {path_yaml}")
    return data

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/expert_xxx.yaml")
    return p.parse_args()

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()

    print(f"> Loaded config : {args.config}")

    cfg = load_config(args.config)

    generate_melody_dataset(cfg)





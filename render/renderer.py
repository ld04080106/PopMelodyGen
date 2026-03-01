#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/render/renderer.py

Pop mid -> audio by soundfont2

Author:      <ludy>
Modified:    2025/12/29
Licence:     <MIT>
"""

import os
import json
import glob
import shutil
import subprocess
from tqdm import tqdm

import numpy as np
from scipy.io import wavfile


def _find_fluidsynth(exe_hint: str) -> str:
    """
    exe_hint: "fluidsynth" or "C:/path/to/fluidsynth.exe"
    """
    if os.path.isfile(exe_hint):
        return exe_hint
    which = shutil.which(exe_hint)
    if which:
        return which
    raise FileNotFoundError(
        f"Cann't find fluidsynth: {exe_hint}\n"
        f"Please install fluidsynth (https://www.fluidsynth.org/)"
    )

def render_one_midi_to_wav(path_mid, path_wav, cfg):
    """
      Render with fluidsynth.
    """
    os.makedirs(os.path.dirname(path_wav), exist_ok=True)

    fluidsynth_bin = _find_fluidsynth(cfg["fluidsynth_bin"])

    # fluidsynth CLI:
    # fluidsynth -ni -r 44100 -g 0.8 soundfont.sf2 input.mid -F output.wav
    # -n: no shell, -i: no interactive
    cmd: List[str] = [
        fluidsynth_bin,
        "-ni",
        "-r", str(cfg["sample_rate"]),
        "-g", str(cfg["gain"]),
    ]

    if cfg["disable_reverb"]:
        cmd += ["-o", "synth.reverb.active=0"]
    if cfg["disable_chorus"]:
        cmd += ["-o", "synth.chorus.active=0"]

    # Optional:
    # cmd += ["-o", "synth.polyphony=256"]
    # cmd += ["-o", "synth.cpu-cores=1"]

    cmd += [
        cfg["path_sf2"],
        path_mid,
        "-F", path_wav,
    ]

    # run fluidsynth
    #proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc = subprocess.run(cmd, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"FluidSynth failed: {path_mid}\n"
            f"Cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    # Optinal
    if cfg["normalize"]:
        sr, audio = wavfile.read(path_wav)
        # audio : int16 / float32 → float32
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            x = audio.astype(np.float32) / 2147483648.0
        else:
            x = audio.astype(np.float32)

        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1e-8:
            x = x * (cfg["target_peak"] / peak)

        # → int16 
        y = np.clip(x, -1.0, 1.0)
        y16 = (y * 32767.0).astype(np.int16)

        # remove start silence
        while len(y16) > 0 and abs(y16[0][0]) < 80:
            y16 = y16[1:, :]

        # remove end silence
        while len(y16) > 0 and abs(y16[-1][0]) < 80:
            y16 = y16[:-1, :]

        # save wave
        wavfile.write(path_wav, sr, y16)


def render_batch(in_midi_dir, out_audio_dir, cfg,
                 pattern="*.mid", skip_existing = True):
    """
        render and report.
    """
    midi_paths = sorted(glob.glob(os.path.join(in_midi_dir, pattern)))
    os.makedirs(out_audio_dir, exist_ok=True)

    report = {
        "in_midi_dir": in_midi_dir,
        "out_audio_dir": out_audio_dir,
        "count_total": len(midi_paths),
        "count_ok": 0,
        "count_failed": 0,
        "ok": [],
        "failed": [],  # each: {"midi":..., "error":...}
        "config": {
            "fluidsynth_bin": cfg["fluidsynth_bin"],
            "path_sf2": cfg["path_sf2"],
            "sample_rate": cfg["sample_rate"],
            "gain": cfg["gain"],
            "disable_reverb": cfg["disable_reverb"],
            "disable_chorus": cfg["disable_chorus"],
            "normalize": cfg["normalize"],
            "target_peak": cfg["target_peak"],
        }
    }

    for path_mid in tqdm(midi_paths):
        song_id = os.path.splitext(os.path.basename(path_mid))[0]
        path_wav = os.path.join(out_audio_dir, f"{song_id}.wav")
        #if "_song" in path_mid:
        #    continue

        if skip_existing and os.path.exists(path_wav) and os.path.getsize(path_wav) > 1024:
            report["ok"].append({"midi": path_mid, "wav": path_wav, "skipped": True})
            report["count_ok"] += 1
            continue

        try:
            render_one_midi_to_wav(path_mid, path_wav, cfg)
            report["ok"].append({"midi": path_mid, "wav": path_wav, "skipped": False})
            report["count_ok"] += 1
        except Exception as e:
            report["failed"].append({"midi": path_mid, "error": str(e)})
            report["count_failed"] += 1

    return report

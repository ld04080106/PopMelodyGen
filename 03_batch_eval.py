#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_batch_eval.py

Batch score synthesized melodies with the third-party SongEval evaluator.

Setup:
  1) Pull/clone SongEval into the local folder `SongEval/`:
       git clone https://github.com/ASLP-lab/SongEval.git SongEval

  2) Make sure SongEval is importable by setting:
     CWD = "/path/to/SongEval"

How it works:
  - This script locates `SongEval/eval.py` (assigned to `path_eval_py`) and invokes it via
    a subprocess call to score all WAV files under `--in_dir`.
  - Scores are written to an output file (`result.json`) in the input folder.

Inputs:
  - --in_dir: Directory containing WAV files produced by Step 02.

Outputs:
  - Aesthetic scores for each sample (format depends on the SongEval eval script).

Example:
  python 03_batch_eval.py --in_dir _dataset

Notes:
  - SongEval is a third-party project. Please comply with its license and do not redistribute
    its weights unless permitted.
  - Other scoring model can be used as needed.

Author:      <ludy>
Modified:    2026/01/10
Licence:     <MIT>
"""

import os
import sys
import json
import time
import glob
import shutil
import subprocess
import argparse

# ===== Config =====
CWD = os.path.join(os.getcwd(), "SongEval")
path_eval_py = os.path.join(CWD, "eval.py")

python_exe = sys.executable

# ===== Eval =====
def run_eval(in_dir):
    """
    Cmd: python eval.py -i in_dir -o out_dir
    """
    # check input dir
    if not os.path.isabs(in_dir):
        in_dir = os.path.abspath(in_dir)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Error: input dir not found ({in_dir})")
    
    # check SongEval
    if not os.path.exists(path_eval_py):
        raise FileNotFoundError(f"Error: SongEval not found ({path_eval_py})")

    # output dir
    out_dir = in_dir 
    os.makedirs(out_dir, exist_ok=True)
 
    cmd = [
        python_exe,
        path_eval_py,
        "-i", in_dir,
        "-o", out_dir,
    ]

    print(" Running cmd: ", ' '.join(cmd))
    print(" [Info]: You can also choose to run above command in your terminal.")

    start = time.time()
    proc = subprocess.run(
        cmd,
        #stdout=subprocess.PIPE,
        #stderr=subprocess.PIPE,
        text=True,
        cwd=CWD
    )
    end = time.time()

    run_info = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(end - start, 3),
        "in_dir": in_dir,
        "out_dir": out_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
 
    if proc.returncode != 0:
        raise RuntimeError(
            f"Error: eval.py failed (returncode={proc.returncode})\n" 
        )

    print(f" [Info] Output of SongEval is located in {out_dir}")

    return run_info
 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="/path/to/PopMelodyGen/_dataset")
    return p.parse_args()

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()

    run_info = run_eval(args.in_dir)

    print("Eval finished:", run_info)
 

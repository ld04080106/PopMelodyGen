#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator.py

Author:      <ludy>
Modified:    2026/06/19
Licence:     <MIT>
"""

import torch
import numpy as np
import os
import json
import random
import argparse
import pretty_midi
from itertools import combinations
from model.tokenizer import REMITokenizer 
from model.model import GPTMelody
from model.generator import gen_token_by_prompt, build_prompt, gen_sections, gen_chord_prog
from expert_melody.pop_chordprog import Chord

"""
1. fixed prompts:
 - key: 1~12
 - range: male/female
 - tempo: slow/mid/fast
 - structure: AA-BB / AA-C-BB & 8/16
 - chord progression seen: 3
 - chord progression unseen: 3
2. generate tokens (5 melodies for each prompt)
3. calculate:
    - E1 ~ E6

"""
FIX_KEYS = list(range(12))
FIX_MODE = [0,] # Major only for now
FIX_TEMPO = ["SLOW", "MID", "FAST"]
FIX_GENDER = ["male", "female"]
FIX_STRUCT = ["A8A8C8B8B8", "A16A16B16B16"]
FIX_CHORDPROG_SEEN = {
    "SLOW": {"A8A8C8B8B8":[(["vi", "iii", "IV", "V", "vi", "iii", "IV", "V"], #A
                            ["vi", "iii", "IV", "V", "vi", "iii", "IV", "V"], #A 
                            [("IV","I"), ("ii","V"), ("iii","vi"), ("ii","V"), ("IV","I"), ("ii","V"), ("iii","vi"), ("ii","V")], #C
                            [("IV","V"), ("iii","vi"), "IV", "V", ("IV","V"), ("iii","vi"), ("IV","V"), "I"], #B
                            [("IV","V"), ("iii","vi"), "IV", "V", ("IV","V"), ("iii","vi"), ("IV","V"), "I"], #B
                            ),
                           (["I", "IV", "V", "vi", ("IV","V"), ("I","vi"), "IV", "V"], #A
                            ["I", "IV", "V", "vi", ("IV","V"), ("I","vi"), ("IV","V"), "I"], #A 
                            [("IV","V"), "I", ("IV","V"), ("I","V"), ("IV","V"), "I", ("IV","V"), "I"], #C
                            ["I", "V", "vi", "iii", "IV", ("iii","vi"), "ii7", "V"], #B
                            ["I", "V", "vi", "iii", "IV", ("iii","vi"), ("ii7","V"), "I"], #B
                            ),
                           ([("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V")], #A
                            [("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V")], #A 
                            [("IV","V"), ("iii7","vi7"), ("IV","ii7"), ("V","I"), ("IV","V"), ("III7","vi7"), ("ii7","II7"), ("IV","V")], #C
                            [("vi","iii"), ("IV","V"), ("vi","V"), ("IV","V"), ("vi","iii"), ("IV","V"), ("vi","V"), ("IV","V")], #B
                            [("vi","iii"), ("IV","V"), ("vi","V"), ("IV","V"), ("vi","iii"), ("IV","V"), ("vi","V"), "I"], #B
                            ),
                          ],
           "A16A16B16B16":[([("I","IV"), ("V","I"), ("vi7","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi7","II7"), ("V","I"), ("I","IV"), ("V","I"), ("vi7","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi7","II7"), ("V","I")], #A 
                            [("I","IV"), ("V","I"), ("vi7","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi7","II7"), ("V","I"), ("I","IV"), ("V","I"), ("vi7","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi7","II7"), ("V","I")], #A
                            ["IV", "I", ("ii","III"), ("vi","I"), "IV", "I", ("ii","V"), "I", "IV", "I", ("ii","III"), ("vi","I"), "IV", "I", ("ii","V"), "I"], #B
                            ["IV", "I", ("ii","III"), ("vi","I"), "IV", "I", ("ii","V"), "I", "IV", "I", ("ii","III"), ("vi","I"), "IV", "I", ("ii","V"), "I"], #B
                            ),
                           (["I", "V", "vi", "iii7", "IV", "I", "IV", "V", "I", "V", "vi", "iii7", "IV", "I", ("IV","V"), "I"], #A 
                            ["I", "V", "vi", "iii7", "IV", "I", "IV", "V", "I", "V", "vi", "iii7", "IV", "I", ("IV","V"), "I"], #A 
                            ["I", ("IV","III7"), "vi7", ("IV","V"), "I", ("IV","III7"), ("vi","I"), ("ii","V"), "I", ("IV","III7"), "vi7", ("IV","V"), "I", ("IV","III7"), ("ii","V"), "I"], #B
                            ["I", ("IV","III7"), "vi7", ("IV","V"), "I", ("IV","III7"), ("vi","I"), ("ii","V"), "I", ("IV","III7"), "vi7", ("IV","V"), "I", ("IV","III7"), ("ii","V"), "I"], #B
                            ),
                           (["vi", "iii", "IV", "V", "vi", "iii", "IV", ("V","III"), "vi", "iii", "IV", "V", "vi", "iii", "IV", "V"], #A 
                            ["vi", "iii", "IV", "V", "vi", "iii", "IV", ("V","III"), "vi", "iii", "IV", "V", "vi", "iii", "IV", "V"], #A 
                            [("IV","V"), ("iii","vi"), ("ii7","V"), ("iii","vi"), ("IV","V"), ("III","vi"), ("II","V"), "I", ("IV","V"), ("iii","vi"), ("ii7","V"), ("iii","vi"), ("IV","V"), ("III","vi"), ("II","V"), "I"], #B
                            [("IV","V"), ("iii","vi"), ("ii7","V"), ("iii","vi"), ("IV","V"), ("III","vi"), ("II","V"), "I", ("IV","V"), ("iii","vi"), ("ii7","V"), ("iii","vi"), ("IV","V"), ("III","vi"), ("II","V"), "I"], #B
                            ),
                          ],
            },
    "MID":  {"A8A8C8B8B8":[(["I", "vi", "IV", "V", "I", "vi", "IV", "V"], #A
                            ["I", "vi", "IV", "V", "I", "vi", "IV", "V"], #A 
                            ["IV", "V", "iii7", "vi", "ii", "I", "IV", "V"], #C
                            ["I", "V", "IV", "V", "I", "V", ("IV","V"), "I"], #B
                            ["I", "V", "IV", "V", "I", "V", ("IV","V"), "I"], #B
                            ),
                           (["I", "iii", "IV", "V", "I", "V", "IV", "V"], #A
                            ["I", "iii", "IV", "V", "I", "V", "IV", "V"], #A 
                            ["IV", "V", "III", "vi", "IV", "I", "iv", "V7"], #C
                            ["I", "IV", "V", "vi", "ii7", "I", "IV", "V"], #B
                            ["I", "IV", "V", "vi", "ii7", "I", ("IV","V"), "I"], #B
                            ),
                           ([("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V")], #A
                            [("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V")], #A 
                            ["iii", "vi", "iii", "vi", "iii", "vi", "ii", "V"], #C
                            ["IV", "V", "iii", "vi", "IV", "I", ("IV","V"), "I"], #B
                            ["IV", "V", "iii", "vi", "IV", "I", ("IV","V"), "I"], #B
                            ),
                          ],
           "A16A16B16B16":[(["I", "V", "vi", "iii", "IV", "I", "ii", "V",  "I", "V", "vi", "iii", "IV", "I", "ii", "V"], #A 
                            ["I", "V", "vi", "iii", "IV", "I", "ii", "V",  "I", "V", "vi", "iii", "IV", "I", "ii", "V"], #A
                            [("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), "ii7", "V7",  ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("ii7","V7"), "I"], #B
                            [("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), "ii7", "V7",  ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("ii7","V7"), "I"], #B
                            ),
                           (["I","III", "vi","iii", "IV","I", "ii","V", "I","III", "vi","iii", "IV","I", "bIII","V"], #A 
                            ["I","III", "vi","iii", "IV","I", "ii","V", "I","III", "vi","iii", "IV","I", "bIII","V"], #A 
                            ["I", "V", "vi", "iii", "IV", "I", "II7", "V",  "I", "V", "vi", "iii", "IV", "I", ("II7","V"), "I"], #B
                            ["I", "V", "vi", "iii", "IV", "I", "II7", "V",  "I", "V", "vi", "iii", "IV", "I", ("II7","V"), "I"], #B
                            ),
                           (["I", "vi", "IV", "V", "iii7", "vi", "IV", "V", "I", "vi", "IV", "V", "iii7", "vi", "IV", "V"], #A 
                            ["I", "vi", "IV", "V", "iii7", "vi", "IV", "V", "I", "vi", "IV", "V", "iii7", "vi", "IV", "V"], #A 
                            [("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I"], #B
                            [("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I"], #B
                            ),
                          ],
            },
    "FAST": {"A8A8C8B8B8":[(["I", "vi", "IV", "V", "I", "vi", "IV", "V"], #A
                            ["I", "vi", "IV", "V", "I", "vi", "IV", "V"], #A 
                            ["IV", "iii", "ii", "V", "IV", "iii", "ii", "V"], #C
                            [("I","V"), "vi", "IV", "V", ("I","V"), "vi", ("IV","V"), "I"], #B
                            [("I","V"), "vi", "IV", "V", ("I","V"), "vi", ("IV","V"), "I"], #B
                            ),
                           (["I", "vi", "IV", "V", "iii7", "vi", "IV", "V"], #A
                            ["I", "vi", "IV", "V", "iii7", "vi", "IV", "V"], #A 
                            ["IV", "V", "iii", "vi7", "IV", "V", ("ii","II"), "V"], #C
                            ["vi", "IV", "V", "I", "vi", "IV", "V", "I"], #B
                            ["vi", "IV", "V", "I", "vi", "IV", "V", "I"], #B
                            ),
                           (["I", "ii7", "iii", "vi", "IV", "ii7", "V7", "I"], #A
                            ["I", "ii7", "iii", "vi", "IV", "ii7", "V7", "I"], #A 
                            ["IV", "iv", "I", "vi", "ii", "I", "IV", "V"], #C
                            ["I", "V", "iii", "vi", "IV", "I", "bVII", "V"], #B
                            ["I", "V", "iii", "vi", "IV", "I", ("IV","V"), "I"], #B
                            ),
                          ],
           "A16A16B16B16":[(["I","V", "vi","I", "IV","V", "I","V",   "I","V", "vi","I", "IV","V", "I","I"], #A 
                            ["I","V", "vi","I", "IV","V", "I","V",   "I","V", "vi","I", "IV","V", "I","I"], #A
                            ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii7", "V",   "I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), ("ii7","V"), "I"], #B
                            ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii7", "V",   "I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), ("ii7","V"), "I"], #B
                            ),
                           (["I", "V", "vi7", "III", "IV", "I", "ii7", "V",  "I", "V", "vi7", "III", "IV", "I", "ii7", "V"], #A 
                            ["I", "V", "vi7", "III", "IV", "I", "ii7", "V",  "I", "V", "vi7", "III", "IV", "I", "ii7", "V"], #A 
                            [("I","V"), "vi", "IV", "V", ("I","V"), "vi", "IV", "V",  ("I","V"), "vi", "IV", "V", ("I","V"), "vi", ("IV","V"), "I"], #B
                            [("I","V"), "vi", "IV", "V", ("I","V"), "vi", "IV", "V",  ("I","V"), "vi", "IV", "V", ("I","V"), "vi", ("IV","V"), "I"], #B
                            ),
                           (["I", "V", "vi", "iii", "IV", "I", "V", "V",  "I", "V", "vi", "IV", "ii", "V", "I", "I"], #A 
                            ["I", "V", "vi", "iii", "IV", "I", "V", "V",  "I", "V", "vi", "IV", "ii", "V", "I", "I"], #A 
                            ["I", "vi", "IV", "V", "I", "vi", "IV", "V",   "I", "vi", "IV", "V", "I", "vi", ("IV","V"), "I"], #B
                            ["I", "vi", "IV", "V", "I", "vi", "IV", "V",   "I", "vi", "IV", "V", "I", "vi", ("IV","V"), "I"], #B
                            ),
                          ],
            }
}
FIX_CHORDPROG_UNSEEN = {
    "SLOW": {"A8A8C8B8B8":[(["vi", "IV", "V", "I", "vi", "IV", "V", "I", ], #A 
                            ["vi", "IV", "V", "I", "vi", "IV", "V", "I", ], #A
                            [("IV","V"), ("I","vi"), ("IV","V"), "I",  ("IV","V"), ("I","vi"), "ii", "III", ], #C
                            [("vi","IV"), ("V","I"), ("vi","IV"), ("V","I"), ("vi","IV"), ("V","I"), ("vi","IV"), ("V","I")], #B
                            [("vi","IV"), ("V","I"), ("vi","IV"), ("V","I"), ("vi","IV"), ("V","I"), ("vi","IV"), ("V","I")], #B
                            ),
                           ([("I","IV"),("V","I"), ("vi","iii"),("IV","V"), ("I","IV"),("V","I"), ("vi","iii"),("IV","V"), ], #A 
                            [("I","IV"),("V","I"), ("vi","iii"),("IV","V"), ("I","IV"),("V","I"), ("vi","iii"),("IV","V"), ], #A
                            ["IV", "V", "iii7", "vi", "IV", "ii", "IV", "V"], #C
                            ["I", "V", "vi", "iii", "IV", ("iii","vi"), "ii", "V"], #B
                            ["I", "V", "vi", "iii", "IV", ("iii","vi"), ("ii","V"), "I"], #B
                            ),
                           ([("I","iii"),("IV","I"),("vi","iii"),("IV","V"),("I","iii"),("IV","I"),("vi","iii"),("IV","V")], #A 
                            [("I","iii"),("IV","I"),("vi","iii"),("IV","V"),("I","iii"),("IV","I"),("vi","iii"),("IV","V")], #A
                            ["IV", "V", "iii7", "vi", "IV", "iii7", "ii7", "V"], #C
                            [("I","V"), ("vi","I"), ("IV","I"), ("ii","V"), ("I","V"), ("vi","I"), ("IV","I"), ("ii","V")], #B
                            [("I","V"), ("vi","I"), ("IV","I"), ("ii","V"), ("I","V"), ("vi","I"), ("IV","V"), "I"], #B
                            ),
                          ],
           "A16A16B16B16":[(["vi", "IV", "V", "I", "IV", "ii", "V", "III", "vi", "IV", "V", "I", "IV", "ii", "IIIsus4", "III"], #A 
                            ["vi", "IV", "V", "I", "IV", "ii", "V", "III", "vi", "IV", "V", "I", "IV", "ii", "IIIsus4", "III"], #A
                            ["I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "IV", "V", "I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "IV", "V"], #B
                            ["I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "IV", "V", "I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), ("IV","V"), "I"], #B
                            ),
                           (["vi", "IV", "V", "I", "vi", "IV", "V", "I", "IV", "V", "iii7", "vi", "IV", "ii", "IV", "V"], #A 
                            ["vi", "IV", "V", "I", "vi", "IV", "V", "I", "IV", "V", "iii7", "vi", "IV", "ii", "IV", "V"], #A
                            [("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ], #B
                            [("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("IV","V"), "I"], #B
                            ),
                           (["I", "III", "vi", ("v","I"), "IV", "I", "ii", "V", "I", "III", "vi", ("v","I"), "IV", "I", "ii", "V"], #A 
                            ["I", "III", "vi", ("v","I"), "IV", "I", "ii", "V", "I", "III", "vi", ("v","I"), "IV", "I", "ii", "V"], #A
                            [("IV","V"),("I","vi"),("IV","V"), "I", ("IV","V"),("III7","vi"),"IV", "V", ("IV","V"),("I","vi"),("IV","V"), "I", ("IV","V"),("III7","vi"),"IV", "V"], #B
                            [("IV","V"),("I","vi"),("IV","V"), "I", ("IV","V"),("III7","vi"),"IV", "V", ("IV","V"),("I","vi"),("IV","V"), "I", ("IV","V"),("III7","vi"),("IV","V"), "I"], #B
                            ),
                          ],
            },
    "MID":  {"A8A8C8B8B8":[([("I","iii7"), "vi7", ("IV","I"), ("ii7","V"), ("iii7","vi7"), ("iii7","vi7"), "ii7", "V7"], #A
                            [("I","iii7"), "vi7", ("IV","I"), ("ii7","V"), ("iii7","vi7"), ("iii7","vi7"), "ii7", "V7"], #A 
                            ["IV", "iii", "ii", "V", "IV", "iii", "ii", "V"], #C
                            ["IV", "V", "iii", "vi", "ii", "I", "ii", "V"], #B
                            ["IV", "V", "iii", "vi", "ii", "I", "IV", "I"], #B
                            ),
                           (["I", "iii", "IV", "iv", "I", "vi", "ii", "V"], #A
                            ["I", "iii", "IV", "iv", "I", "vi", "ii", "V"], #A 
                            ["vi", "V", "IV", "V", "ii7", "iii7", "IV", "V"], #C
                            ["IV", "I", ("ii7","III7"), "vi", ("IV","V"), ("iii7","vi7"), "ii7", "V"], #B
                            ["IV", "I", ("ii7","III7"), "vi", ("IV","V"), ("iii7","vi7"), ("ii7","V"), "I"], #B
                            ),
                           (["ii", "V", "I", "vi", "ii", "V", "iii", "vi"], #A
                            ["ii", "V", "I", "vi", "ii", "V", "iii", "VI"], #A 
                            [("ii","iii7"), "vi7", ("IV","III"),"vi", ("IV","iii7"), "vi7", "ii7", "V"], #C
                            [("I","V"), "vi", ("IV","V"), "I", ("IV","V"), ("iii7","vi"), "II7", "V7"], #B
                            [("I","V"), "vi", ("IV","V"), "I", ("IV","V"), ("iii7","vi"), ("II7","V7"), "I"], #B
                            ),
                          ],
           "A16A16B16B16":[(["I", "ii7", "V", "vi", "IV", "I", "II7", "V", "I", "ii7", "V", "vi", "IV", "I", "II7", "V"], #A 
                            ["I", "ii7", "V", "vi", "IV", "I", "II7", "V", "I", "ii7", "V", "vi", "IV", "I", "II7", "V"], #A
                            ["I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "IV", "V", "I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "IV", "V"], #B
                            ["I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "IV", "V", "I", "III", "vi", "iii", ("IV","V"), ("iii","vi"), "iv", "I"], #B
                            ),
                           (["I", "III", "vi", "iii", "ii", "iii", "IV", "V", "I", "III", "vi", "iii", "ii", "V", "I", "I"], #A 
                            ["I", "III", "vi", "iii", "ii", "iii", "IV", "V", "I", "III", "vi", "iii", "ii", "V", "I", "I"], #A
                            ["I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", "IV", "iv"], #B
                            ["I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", ("IV","iv"), "I"], #B
                            ),
                           (["I", "V", "IV", ("I","V"), "vi", "iii", "IV", "V",  "I", "V", "IV", ("I","V"), "vi", "iii", "IV", "V"], #A 
                            ["I", "V", "IV", ("I","V"), "vi", "iii", "IV", "V",  "I", "V", "IV", ("I","V"), "vi", "iii", "IV", "V"], #A
                            ["IV", "V", ("I","V"), ("vi","V"), "IV", "V", "vi", ("v", "I"), "IV", "V", ("I","V"), ("vi","V"), "IV", "V", "vi", "V"], #B
                            ["IV", "V", ("I","V"), ("vi","V"), "IV", "V", "vi", ("v", "I"), "IV", "V", ("I","V"), ("vi","V"), "IV", "V", "vi", "vi"], #B
                            ),
                          ],
            },
    "FAST": {"A8A8C8B8B8":[(["I", "V", "IV", "I", "V", "vi", "ii", "V"], #A 
                            ["I", "V", "IV", "I", "V", "vi", "ii", "V"], #A
                            ["IV", "V", "III", "vi", "IV", "I", "ii", "V"], #C
                            ["I", "V", "vi", ("v7","I"), ("IV","V"), ("iii7", "vi"), "iv", "V"], #B
                            ["I", "V", "vi", ("v7","I"), ("IV","V"), ("iii7", "vi"), "iv", "I"], #B
                            ),
                           (["I", "vi", "ii", "V", "I", "vi", "ii", "I"], #A 
                            ["I", "vi", "ii", "V", "I", "vi", "ii", "I"], #A
                            ["IV", "V", ("I","V"), "vi", "IV", "I", "ii", "V"], #C
                            ["I", "V", "vi", "III", ("IV","V"), ("I","vi"), "IV", "V"], #B
                            ["I", "V", "vi", "III", ("IV","V"), ("I","vi"), ("IV","V"), "I"], #B
                            ),
                           (["I", "V", "vi", "V", "IV", "III", "vi", "V"], #A 
                            ["I", "V", "vi", "V", "IV", "III", "vi", "V"], #A
                            ["IV", "iii7", "ii7", "V", "IV", "iii7", "ii7", "V"], #C
                            ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii", "V"], #B
                            ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), ("ii","iii"), "IV"], #B
                            ),
                          ],
           "A16A16B16B16":[(["I", "V", "IV", "I", "V", "vi", "ii", "V", "I", "V", "IV", "I", "IV", "V", "vi", "vi"], #A 
                            ["I", "V", "IV", "I", "V", "vi", "ii", "V", "I", "V", "IV", "I", "IV", "V", "vi", "vi"], #A
                            ["IV", "V", ("I","V"), ("vi","V"), "IV", "V", "vi", ("v", "I"), "IV", "V", ("I","V"), ("vi","V"), "IV", "V", "I", "V"], #B
                            ["IV", "V", ("I","V"), ("vi","V"), "IV", "V", "vi", ("v", "I"), "IV", "V", ("I","V"), ("vi","V"), "IV", "V", "I", "I"], #B
                            ),
                           (["I", "V", "IV", "I", "V", "vi", "ii", "V", "I", "V", "IV", "I", "V", "vi", "ii", "V"], #A 
                            ["I", "V", "IV", "I", "V", "vi", "ii", "V", "I", "V", "IV", "I", "V", "vi", "ii", "V"], #A
                            ["I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", "IV", "iv"], #B
                            ["I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", "IV", "iv", "I", "vi", ("IV","iv"), "I"], #B
                            ),
                           (["I", "vi", "ii", "V", "I", "vi", "ii", "V", "I", "vi", "ii", "V", "I", "vi", "ii", "V"], #A 
                            ["I", "vi", "ii", "V", "I", "vi", "ii", "V", "I", "vi", "ii", "V", "I", "vi", "ii", "I"], #A
                            ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii", "V", "I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii", "V"], #B
                            ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii", "V", "I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), ("ii","iii"), "IV"], #B
                            ),
                          ],
            }
}

IS_DEBUG = False

def _chord_pitches(ch, key, mode, pitch_low=0, pitch_high=128, only_root=False):
    """ Return degrees for chord """
    root = key
    if mode == 1: # Minor
        root += 3

    # chord degree
    ch = Chord(ch)
    degrees = ch.degrees

    if only_root:
        degrees = degrees[:1]
 
    scales = [(root+x)%12 for x in degrees]
    #print(" ch =", ch, "degrees =", degrees, "is_final_strong =", is_final_strong)

    # all pitches
    chord_pitches = []
    for p in range(128):
        if p%12 in scales and pitch_low <= p <= pitch_high:
            chord_pitches.append(p)
    # return
    return chord_pitches

def _scale_pitches(ch, key, mode, pitch_low=0, pitch_high=128):
    """ Return scales for chord """
    root = key
    if mode == 1: # Minor
        root += 3

    ch = Chord(ch)

    scales_pitches = [] 
    for p in range(128):
        if p%12 in [(root+x)%12 for x in ch.scales] and pitch_low <= p <= pitch_high:
            scales_pitches.append(p)
    # return
    return scales_pitches


def eval_E1_harmony(tokens, key, mode):
    # E1
    strong_total = strong_ok = 0
    weak_total = weak_ok = 0

    i = 0
    bar_idx = 7
    bar_notes = []
    note_dicts = []
    curch1 = curch2 = None
    # loop each token
    while i < len(tokens):
        t = tokens[i]
        #print(" t =", t)

        if t == "BAR":
            # analyze notes of previous bar
            if len(bar_notes) > 0:
                # loop each note in a bar
                nnote = len(bar_notes)
                j = 0
                while j < nnote:
                    (pitch, pos, dur) = bar_notes[j]
                    # strong or weak
                    is_strong = False 
                    if pos == 0:
                        is_strong = True
                        ch = cur_ch1
                        alter_ch = cur_ch2
                    elif pos == 4:
                        is_strong = True
                        ch = cur_ch2 
                        alter_ch = cur_ch2
                    elif pos == 3 and dur >= 2:
                        is_strong = True
                        ch = cur_ch2 
                        alter_ch = cur_ch2
                    elif pos == 1 and dur >= 2 and j==0:
                        is_strong = True
                        ch = cur_ch1 
                        alter_ch = cur_ch2
                    elif pos == 2 and dur >= 2 and j==0:
                        is_strong = True
                        ch = cur_ch2 
                        alter_ch = cur_ch2
                    elif pos == 2 and dur >= 2 and j==1 and bar_notes[j-1][1] > 0:
                        is_strong = True
                        ch = cur_ch1 
                        alter_ch = cur_ch2

                    # appog
                    if is_strong:
                        next_dur = bar_notes[j+1][2] if j+1 < nnote else 0
                        next_next_dur = bar_notes[j+2][2] if j+2 < nnote else 0
                        if pos==0 and dur==1 and next_dur==1 and next_next_dur > 1 \
                           or pos==4 and dur==1 and next_dur==1 and next_next_dur > 1:
                            # double appog 
                            note_dicts.append({"ch": ch, "pitch": pitch, "pos": pos, "dur": dur,
                                               "bstrong": False,
                                               "bweak": False, 
                                               "bappog": True, 
                                               })
                            note_dicts.append({"ch": ch, "pitch": bar_notes[j+1][0], "pos": bar_notes[j+1][1], "dur": bar_notes[j+1][2],
                                               "bstrong": False,
                                               "bweak": False,
                                               "bappog": True, 
                                               }) 
                            note_dicts.append({"ch": ch, "pitch": bar_notes[j+2][0], "pos": bar_notes[j+2][1], "dur": bar_notes[j+2][2],
                                               "bstrong": True, "alter_ch":alter_ch,
                                               "bweak": False,
                                               "bappog": False, 
                                               })
                            # debug
                            if IS_DEBUG: print(f" bar {bar_idx}: note_dicts[-3:] =", note_dicts[-3:])
                            # update 
                            j += 3
                            continue
                        elif pos==0 and dur==1 and next_dur>1 \
                             or pos==4 and dur==1 and next_dur > 1 \
                             or pos==0 and dur==2 and next_dur > 2 \
                             or pos==0 and dur==2 and next_dur >= 2 and j+2==nnote \
                             or pos==0 and dur==2 and next_dur >= 2 and j+2 < nnote and bar_notes[j+2][1]>bar_notes[j+1][1]+next_dur:
                            # single appog
                            note_dicts.append({"ch": ch, "pitch": pitch, "pos": pos, "dur": dur,
                                               "bstrong": False,
                                               "bweak": False,
                                               "bappog": True, 
                                               }) 
                            note_dicts.append({"ch": ch, "pitch": bar_notes[j+1][0], "pos": bar_notes[j+1][1], "dur": bar_notes[j+1][2],
                                               "bstrong": True, "alter_ch":alter_ch,
                                               "bweak": False,
                                               "bappog": False, 
                                               })
                            # debug
                            if IS_DEBUG: print(f" bar {bar_idx}: note_dicts[-2:] =", note_dicts[-2:])
                            # update 
                            j += 2
                            continue
                        else:
                            # 1 strong note 
                            note_dicts.append({"ch": ch, "pitch": pitch, "pos": pos, "dur": dur,
                                               "bstrong": True, "alter_ch":alter_ch,
                                               "bweak": False,
                                               "bappog": False, 
                                               })
                            # debug
                            if IS_DEBUG: print(f" bar {bar_idx}: note_dicts[-1] =", note_dicts[-1])
                            # update 
                            j += 1
                            continue
                    # 1 weak note 
                    if pos < 4:
                        ch = cur_ch1
                    else:
                        ch = cur_ch2 
                    note_dicts.append({"ch": ch, "pitch": pitch, "pos": pos, "dur": dur,
                                       "bstrong": False,
                                       "bweak": True,
                                       "bappog": False, 
                                       })
                    # debug
                    if IS_DEBUG: print(f" bar {bar_idx}: note_dicts[-1] =", note_dicts[-1])
                    # update 
                    j += 1
                    continue 
            # update
            bar_idx += 1
            i += 1
            bar_notes = []
            continue

        # skip token
        if t.startswith(("SEC_","KEY_","MODE_","TEMPO_","STRUCT_","GENDER_","<BOS>","<EOS>","<PAD>","<UNK>")):
            i += 1
            continue
        if t.startswith("CH_"):
            cur_ch1 = t[3:]
            t2 = tokens[i+1]
            if t2.startswith("CH_"):
                cur_ch2 = t2[3:]
            else:
                cur_ch2 = cur_ch1
            i += 2
            continue

        # note triplet: PITCH_y, POS_x, DUR_z
        if t.startswith("PITCH_"):
            # pitch
            pitch = int(t.split("_", 1)[1])
            # pos
            if i+1 >= len(tokens) or not tokens[i+1].startswith("POS_"):
                i += 1
                continue
            pos = int(tokens[i+1].split("_", 1)[1])
            # dur
            if i+2 >= len(tokens) or not tokens[i+2].startswith("DUR_"):
                i += 1
                continue
            dur = int(tokens[i+2].split("_", 1)[1])

            #print("    - note =", pitch, start, end)
            bar_notes.append([pitch, pos, dur])
            i += 3
            continue
        # update
        i += 1

    # Start statistic
    for nt in note_dicts:
        if nt["bstrong"]:
            # strong 
            chord_pitches = _chord_pitches(nt["ch"], key, mode)
            alter_ch = nt.get("alter_ch", None)
            if alter_ch != None:
                chord_pitches += _chord_pitches(alter_ch, key, mode)
            strong_total += 1
            if nt["pitch"] in chord_pitches:
                strong_ok += 1
            else:
                if IS_DEBUG: 
                    print (" > nt =", nt)
                    print("  chord_pitches=", nt["pitch"] in chord_pitches, chord_pitches)
        else:
            # weak/appog
            scale_pitches = _scale_pitches(nt["ch"], key, mode)
            weak_total += 1
            if nt["pitch"] in scale_pitches:
                weak_ok += 1
            else:
                if IS_DEBUG: 
                    print (" > nt =", nt)
                    print("  scale_pitches=", nt["pitch"] in scale_pitches, scale_pitches)
 
    if IS_DEBUG:
        print(" statis:", strong_ok, strong_total, weak_ok, weak_total)  
    return {
        "strong_ok": strong_ok,
        "strong_total": strong_total,
        "weak_ok": weak_ok, 
        "weak_total": weak_total, 
    }
 
def eval_E2_repeat(tokens):
    """ A2 repeat A1, B2 repeat B1 (if same chords) """
    # 1. extract VERSE1/VERSE2/CHORUS1/CHORUS2 token
    i = 0
    A1_bar_tokens = []
    A2_bar_tokens = []
    B1_bar_tokens = []
    B2_bar_tokens = []
    cur_sec = None
    while i < len(tokens):
        t = tokens[i]

        if t.startswith("SEC_"):
            if t in ("SEC_VERSE1", "SEC_VERSE2", "SEC_CHORUS1", "SEC_CHORUS2"):
                cur_sec = t
                if cur_sec == "SEC_VERSE1":
                    A1_bar_tokens.append([])
                elif cur_sec == "SEC_VERSE2":
                    A2_bar_tokens.append([])
                elif cur_sec == "SEC_CHORUS1":
                    B1_bar_tokens.append([])
                elif cur_sec == "SEC_CHORUS2":
                    B2_bar_tokens.append([])
            else:
                cur_sec = None 

        # skip token
        if t.startswith(("SEC_","KEY_","MODE_","TEMPO_","STRUCT_","GENDER_","CH_","BAR","<BOS>","<EOS>","<PAD>","<UNK>")):
            i += 1
            continue

        # note triplet: PITCH_y, POS_x, DUR_z
        if t.startswith("PITCH_"):
            # pitch
            pitch = int(t.split("_", 1)[1])
            # pos
            if i+1 >= len(tokens) or not tokens[i+1].startswith("POS_"):
                i += 1
                continue
            pos = int(tokens[i+1].split("_", 1)[1])
            # dur
            if i+2 >= len(tokens) or not tokens[i+2].startswith("DUR_"):
                i += 1
                continue
            dur = int(tokens[i+2].split("_", 1)[1])

            note = f"{pitch}_{pos}"

            if cur_sec == "SEC_VERSE1":
                A1_bar_tokens[-1].append(note)
            elif cur_sec == "SEC_VERSE2":
                A2_bar_tokens[-1].append(note)
            elif cur_sec == "SEC_CHORUS1":
                B1_bar_tokens[-1].append(note)
            elif cur_sec == "SEC_CHORUS2":
                B2_bar_tokens[-1].append(note) 

        # update
        i += 1

    # flatten tokens (remove final 1 bar because it may contain pickup of next section)
    A1_tokens = []
    A2_tokens = []
    B1_tokens = []
    B2_tokens = []
    for bar in A1_bar_tokens[:-1]:
        A1_tokens += bar
    for bar in A2_bar_tokens[:-1]:
        A2_tokens += bar
    for bar in B1_bar_tokens[:-1]:
        B1_tokens += bar
    for bar in B2_bar_tokens[:-1]:
        B2_tokens += bar

    def seq_similarity(a, b):
        # 2-gram Jaccard
        def ngrams(x, n=2):
            return set(tuple(x[i:i+n]) for i in range(len(x)-n+1))
        A, B = ngrams(a), ngrams(b)
        if not A and not B: 
            return 1.0
        return len(A & B) / max(1, len(A | B))

    v_sim, c_sim = 0, 0
    if len(A1_tokens) > 0 and len(A2_tokens) > 0:
        v_sim = seq_similarity(A1_tokens, A2_tokens)
        # print(" A1_tokens =", A1_tokens)
        # print(" A2_tokens =", A2_tokens)
        # print(" v_sim =", v_sim)

    if len(B1_tokens) > 0 and len(B2_tokens) > 0:
        c_sim = seq_similarity(B1_tokens, B2_tokens)
        # print(" B1_tokens =", B1_tokens)
        # print(" B2_tokens =", B2_tokens)
        # print(" c_sim =", c_sim) 
     
    return {"verse_repeat_sim": v_sim, 
            "chorus_repeat_sim": c_sim
            }
 
def eval_E3_diversity(tokens_list):
    # tokens_list: list of tokens, length = N
    n = len(tokens_list)    # n songs

    # 1. extract VERSE2/PRECHORUS/CHORUS pitch
    A_all_pitches_list = []
    C_all_pitches_list = []
    B_all_pitches_list = []
    A_strong_pitches_list = []
    C_strong_pitches_list = []
    B_strong_pitches_list = []
    for i in range(n): 
        cur_sec = ""
        j = 0
        A_all_pitches = []
        C_all_pitches = []
        B_all_pitches = []
        A_strong_pitches = []
        C_strong_pitches = []
        B_strong_pitches = []
        tokens = tokens_list[i]
        while j < len(tokens):
            t = tokens[j]

            if t.startswith("SEC_"):
                if t.startswith(("SEC_VERSE", "SEC_PRECHORUS", "SEC_CHORUS")):
                    cur_sec = t
                else:
                    cur_sec = "" 

            # note triplet: PITCH_y, POS_x, DUR_z
            if t.startswith("PITCH_"):
                # pitch
                pitch = int(t.split("_", 1)[1])
                # pos
                if j+1 >= len(tokens) or not tokens[j+1].startswith("POS_"):
                    j += 1
                    continue
 
                if cur_sec.startswith("SEC_VERSE"):
                    A_all_pitches.append(pitch)
                elif cur_sec.startswith("SEC_PRECHORUS"):
                    C_all_pitches.append(pitch)
                elif cur_sec.startswith("SEC_CHORUS"):
                    B_all_pitches.append(pitch)
            elif t.startswith("POS_"):
                # only consider strong pos
                pos = int(t.split("_", 1)[1])
            elif t.startswith("DUR_"):
                # 
                dur = int(t.split("_", 1)[1])
                # strong?
                is_strong = False 
                if pos == 0:
                    is_strong = True
                elif pos == 4:
                    is_strong = True
                elif pos == 3 and dur >= 2:
                    is_strong = True

                if is_strong:
                    if cur_sec.startswith("SEC_VERSE"):
                        A_strong_pitches.append(pitch)
                    elif cur_sec.startswith("SEC_PRECHORUS"):
                        C_strong_pitches.append(pitch)
                    elif cur_sec.startswith("SEC_CHORUS"):
                        B_strong_pitches.append(pitch)

            # update
            j += 1
        # save 
        A_all_pitches_list.append(A_all_pitches)
        C_all_pitches_list.append(C_all_pitches)
        B_all_pitches_list.append(B_all_pitches)
        A_strong_pitches_list.append(A_strong_pitches)
        C_strong_pitches_list.append(C_strong_pitches)
        B_strong_pitches_list.append(B_strong_pitches)

    def interval_ngrams(pitches, n=3):
        if len(pitches) < n + 1:
            return set()
        intervals = []
        for i in range(len(pitches)-1):
            d = pitches[i+1] - pitches[i]
            intervals.append(d)
        return set(tuple(intervals[i:i+n]) for i in range(len(intervals)-n+1))

    def jaccard(A, B):
        if not A and not B: 
            return 1.0 
        return len(A & B) / max(1, len(A | B))
 
    # 2. calculate diversity for pairs in A/C/B repectively
    total_sim_all = 0
    total_cnt_all = 0
    total_sim_strong = 0
    total_cnt_strong = 0
    for all_pitches in (A_all_pitches_list, C_all_pitches_list, B_all_pitches_list):
        if len(all_pitches) > 0:
            grams = [interval_ngrams(pitches) for pitches in all_pitches]

            assert len(grams) > 2, f"Error: unknown len(grams)={len(grams)}"
            
            all_sims = []
            for i, j in combinations(range(len(grams)), 2):
                all_sims.append(jaccard(grams[i], grams[j]))
            pair_sim_all = sum(all_sims) / len(all_sims)

            # save 
            total_sim_all += pair_sim_all
            total_cnt_all += 1
    for strong_pitches in (A_strong_pitches_list, C_strong_pitches_list, B_strong_pitches_list):
        if len(strong_pitches) > 0:
            grams = [interval_ngrams(pitches) for pitches in strong_pitches]

            assert len(grams) > 2, f"Error: unknown len(grams)={len(grams)}"
            
            strong_sims = []
            for i, j in combinations(range(len(grams)), 2):
                strong_sims.append(jaccard(grams[i], grams[j]))
            pair_sim_strong = sum(strong_sims) / len(strong_sims)

            # save 
            total_sim_strong += pair_sim_strong
            total_cnt_strong += 1

    # 3. final result 
    rtn_sim_all = total_sim_all / total_cnt_all
    rtn_sim_strong = total_sim_strong / total_cnt_strong

    return {"pair_sim_all": rtn_sim_all, 
            "diversity_all": 1.0 - rtn_sim_all,
            "pair_sim_strong": rtn_sim_strong, 
            "diversity_strong": 1.0 - rtn_sim_strong}

def eval_E4_cadence(tokens, key, mode):
    # final pitch is tonic
    final_pitch = 0
    final_chord = None
    curch1 = curch2 = None
    # loop each token
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # chord token
        if t.startswith("CH_"):
            cur_ch1 = t[3:]
            t2 = tokens[i+1]
            if t2.startswith("CH_"):
                cur_ch2 = t2[3:]
            else:
                cur_ch2 = cur_ch1
            i += 2
            continue

        # pitch token
        if t.startswith("PITCH_"):
            # pitch
            pitch = int(t.split("_", 1)[1])

            # final pitch
            final_pitch = pitch
            final_chord = cur_ch2

        # update
        i += 1

    root_pitches = _chord_pitches(final_chord, key, mode, only_root=True)

    final_hit = final_pitch in root_pitches 
    #print("key=", key, "final_pitch=", final_pitch, "chord=", final_chord, "hit=",final_hit)

    return {"cadence_hit": final_hit }


def eval_E5_register(tokens):
    # 1. extract VERSE2/PRECHORUS/CHORUS pitch
    cur_sec = ""
    j = 0
    A_pitches = []
    C_pitches = []
    B_pitches = []
    while j < len(tokens):
        t = tokens[j]

        if t.startswith("SEC_"):
            if t.startswith(("SEC_VERSE", "SEC_PRECHORUS", "SEC_CHORUS")):
                cur_sec = t
            else:
                cur_sec = "" 
 
        # note triplet: PITCH_y, POS_x, DUR_z
        if t.startswith("PITCH_"):
            # pitch
            pitch = int(t.split("_", 1)[1])

            if cur_sec.startswith("SEC_VERSE"):
                A_pitches.append(pitch)
            elif cur_sec.startswith("SEC_PRECHORUS"):
                C_pitches.append(pitch)
            elif cur_sec.startswith("SEC_CHORUS"):
                B_pitches.append(pitch)
        # update
        j += 1

    def _pitch_stats(pitches):
        arr = np.array(pitches, dtype=float)
        return {
            "n_notes": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "range_p90_p10": float(np.percentile(arr, 90) - np.percentile(arr, 10)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    rtn = {
        "mean_diff": None,
        "range_diff_90": None,
        "range_diff_10": None,
    }
    if len(A_pitches) > 0:
        rtn["A"] = _pitch_stats(A_pitches)
    if len(C_pitches) > 0:
        rtn["C"] = _pitch_stats(C_pitches)
    if len(B_pitches) > 0:
        rtn["B"] = _pitch_stats(B_pitches) 

    # Diff
    if "A" in rtn and "B" in rtn:
        rtn["mean_diff"] = rtn["B"]["mean"] - rtn["A"]["mean"]
        rtn["range_diff_90"] = rtn["B"]["p90"] - rtn["A"]["p90"]
        rtn["range_diff_10"] = rtn["B"]["p10"] - rtn["A"]["p10"]
    
    return rtn


def eval_E6_stepwise(tokens):
    # 1. extract VERSE2/PRECHORUS/CHORUS pitch
    cur_sec = ""
    j = 0
    A_pitches = []
    C_pitches = []
    B_pitches = []
    while j < len(tokens):
        t = tokens[j]

        if t.startswith("SEC_"):
            if t.startswith(("SEC_VERSE", "SEC_PRECHORUS", "SEC_CHORUS")):
                cur_sec = t
            else:
                cur_sec = "" 
 
        # note triplet: PITCH_y, POS_x, DUR_z
        if t.startswith("PITCH_"):
            # pitch
            pitch = int(t.split("_", 1)[1])

            if cur_sec.startswith("SEC_VERSE"):
                A_pitches.append(pitch)
            elif cur_sec.startswith("SEC_PRECHORUS"):
                C_pitches.append(pitch)
            elif cur_sec.startswith("SEC_CHORUS"):
                B_pitches.append(pitch)
        # update
        j += 1

    def _compute_interval(pitches):
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        n = len(intervals)

        if n == 0:
            return {
                "n_notes": len(pitches),
                "n_intervals": 0,
                "stepwise_rate": None,
                "smallleap_rate": None,
                "largeleap_rate": None,
                "extremeleap_rate": None,
                "mean_abs_interval": None,
                "direction_change_rate": None,
                "direction_change_rate2": None,
            }

        abs_int = [abs(x) for x in intervals]

        step = sum(1 for a in abs_int if a <= 2)
        small_leap = sum(1 for a in abs_int if 3 <= a <= 5)
        large_leap = sum(1 for a in abs_int if a >= 7)
        extreme = sum(1 for a in abs_int if a >= 12)

        mean_abs = sum(abs_int) / n

        # Direction change: count sign flips ignoring zeros
        def sgn(x):
            if x > 0: return 1
            if x < 0: return -1
            return 0

        signs = [sgn(x) for x in intervals]
        # compress zeros by carrying last nonzero sign forward (or drop zeros)
        nz = [s for s in signs if s != 0]
        
        dir_change = sum(1 for i in range(len(nz)-1) if nz[i+1] != nz[i])
        denom = len(nz) - 1

        return {
            "n_notes": len(pitches),
            "n_intervals":  n,
            "stepwise_rate": step / n,
            "smallleap_rate": small_leap / n,
            "largeleap_rate": large_leap / n,
            "extremeleap_rate": extreme / n,
            "mean_abs_interval": mean_abs,
            "direction_change_rate": dir_change / denom if denom > 0 else None,
            "direction_change_rate2": dir_change / (len(signs)-1) if len(signs) > 1 else None,
        }

    rtn = {
        "A": _compute_interval(A_pitches),
        "B": _compute_interval(B_pitches),
    }
    if len(C_pitches) > 0:
        rtn["C"] = _compute_interval(C_pitches)
    
    return rtn




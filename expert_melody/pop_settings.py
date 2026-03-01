#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pop_settings.py

Definitions and sampling utilities for global and per-song settings.

Responsibilities:
  - Define global conditions: tempo class, key/mode, vocal register, rhythm, pitch, etc.

Author:      <ludy>
Modified:    2026/01/22
Licence:     <MIT>
"""

from enum import IntEnum, Enum, unique
import numpy as np

##----------------------------------------------
# Enum: tempo + key + mode + section
##---------------------------------------------- 
@unique
class Tempo(Enum):
    SLOW = "SLOW"
    MID = "MID"
    FAST = "FAST"

@unique
class Key(IntEnum):
    C  = 0
    bD  = 1
    D  = 2
    bE   = 3
    E   = 4
    F   = 5
    bG   = 6
    G   = 7
    bA   = 8
    A   = 9
    bB   = 10
    B   = 11

@unique
class Mode(IntEnum):
    Major  = 0
    Minor  = 1

@unique
class SectionType(IntEnum):
    INTRO = 0
    VERSE = 1
    PRECHORUS = 2
    CHORUS = 3
    INTERLUDE1 = 4 
    INTERLUDE2 = 5 
    BRIDGE = 6
    OUTRO = 7

##-----------------------------
# 0. Setting: Global
##----------------------------- 
IS_DEBUG = False     # True / False

##-----------------------------
# 1. Setting: tempo 
##----------------------------- 
# Range of each kind of tempo (by exp)
TEMPO_RANGES = { 
    Tempo.SLOW: (70, 94),   # 25
    Tempo.MID: (95, 124),   # 30
    Tempo.FAST: (125, 150), # 25
}

##-----------------------------
# 2. Setting: Mode 
##----------------------------- 
# P(MODE|TEMPO)  (prior by exp)
P_MODE_TEMPO = {
    Tempo.SLOW:     {Mode.Major: 1, Mode.Minor: 0},
    Tempo.MID:      {Mode.Major: 1, Mode.Minor: 0},
    Tempo.FAST:     {Mode.Major: 1, Mode.Minor: 0},
}
# P_MODE_TEMPO = {  # for future use
#     Tempo.SLOW:     {Mode.Major: 0.4, Mode.Minor: 0.6},
#     Tempo.MID:      {Mode.Major: 0.8, Mode.Minor: 0.2},
#     Tempo.FAST:     {Mode.Major: 0.9, Mode.Minor: 0.1},
# }

##-----------------------------
# 3. Setting: Vocal register for different singers（MIDI）
##----------------------------- 
VOCAL_REGISTERS = {
    "female": [ (43, 64),  # G3–E5
                (42, 63),  # bG3–bE5
                (44, 65),  # bA3–F5
                (45, 66),  # A3–bG5
              ],
    "male": [   (36, 57),    # C3–A4
                (35, 56),    # B2–bA4
                (37, 58),    # bD2–bB4
                (38, 59),    # D2–B4
            ],
}

# offset of vocal register for different sections
VERSE_LOW_OFFSET = 0
VERSE_HIGH_OFFSET = -6
PRECH_LOW_OFFSET = 0
PRECH_HIGH_OFFSET = -6
CHORUS_LOW_OFFSET = +6
CHORUS_HIGH_OFFSET = 0

##-----------------------------
# 4. Setting: Song duration
##----------------------------- 
# only synthesize half song, shouldn't be too long!
MIN_SONG_SEC = 60  # 1min
MAX_SONG_SEC = 180  # 3min

##-----------------------------
# 5. Setting: Rhythm
##----------------------------- 
# Available syllable count under different bar-len / tempo class
RHYTHM_CHARS_RANGE = {
    Tempo.SLOW:  { 2: (5, 13),
                   4: (8, 18), },
    Tempo.MID:   { 2: (5, 13),
                   4: (8, 18), },
    Tempo.FAST:  { 2: (5, 13),
                   4: (8, 18), },
}

# trim notes that are too long in rhythm (<= 2nd half note)
TRIM_LONG_NOTE = True

##-----------------------------
# 6. Setting: Pitch transition P
##-----------------------------  
VERSE_PITCH_TRANS = {
   12: 0, 
   11: 0, 
    9: 0.1, 
    8: 0.2, 
    7: 0.2, 
    6: 0.1, 
    5: 0.5, 
    4: 0.5, 
    3: 0.5, 
    2: 0.3, 
    1: 0.3, 
    0: 0.1, 
   -1: 0.3, 
   -2: 0.3, 
   -3: 0.5, 
   -4: 0.5, 
   -5: 0.5, 
   -6: 0.1, 
   -7: 0.2, 
   -8: 0.2, 
   -9: 0.1, 
  -10: 0, 
  -11: 0, 
} 
PRECH_PITCH_TRANS = {
   12: 0, 
   11: 0, 
    9: 0.1, 
    8: 0.2, 
    7: 0.2, 
    6: 0.1, 
    5: 0.5, 
    4: 0.5, 
    3: 0.5, 
    2: 0.3, 
    1: 0.3, 
    0: 0.1, 
   -1: 0.3, 
   -2: 0.3, 
   -3: 0.5, 
   -4: 0.5, 
   -5: 0.5, 
   -6: 0.1, 
   -7: 0.2, 
   -8: 0.2, 
   -9: 0.1, 
  -10: 0, 
  -11: 0, 
} 
CHORUS_PITCH_TRANS = {
   12: 0, 
   11: 0, 
    9: 0.1, 
    8: 0.2, 
    7: 0.2, 
    6: 0.1, 
    5: 0.5, 
    4: 0.5, 
    3: 0.5, 
    2: 0.3, 
    1: 0.3, 
    0: 0.1, 
   -1: 0.3, 
   -2: 0.3, 
   -3: 0.5, 
   -4: 0.5, 
   -5: 0.5, 
   -6: 0.1, 
   -7: 0.2, 
   -8: 0.2, 
   -9: 0.1, 
  -10: 0, 
  -11: 0, 
} 



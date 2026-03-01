#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pop_chordprog.py

Chord progression templates and bar-wise chord assignment.

Responsibilities:
  - Maintain curated chord progression pools (by section, tempo class, mode).
  - Sample a progression template and expand it to bar-wise chords aligned to structure.
  - Provide chord symbols and chord tones for melody generation.

Author:      <ludy>
Modified:    2026/01/17
Licence:     <MIT>
"""

from expert_melody.pop_settings import Tempo, Key, SectionType, Mode
from expert_melody.pop_structure import STRUCTURE_TEMPLATES

import random

##-----------------------------
# Chord-progession
##-----------------------------
# Chord class
class Chord:
    def __init__(self, name):
        self.name = name
        QUALITY_DICT = {'': (0, 4, 7,),    #
                     'm':   (0, 3, 7,),    # m
                     'm7':  (0, 3, 7, 10, ),   # m7
                     '7':   (0, 4, 7, 10, ),   # 7
                     'M':   (0, 4, 8, ),   # M
                     'M7':  (0, 4, 7, 11, ),   # maj7
                     'aug': (0, 4, 8, ),   # aug
                     'dim': (0, 3, 6, ),   # dim
                     'dim7':(0, 3, 6, 9, ),    # dim7
                     'add9':(0, 4, 7, 2, ),   # add9      1,3,5 + 9 (no 7)
                     'madd9': (0, 3, 7, 2, ),   # madd9     1,b3,5 + 9 (no 7)
                     'Maj9':  (0, 4, 7, 2, ),   # Maj9  
                     '9':     (0, 4, 7, 2, ),   # 9  
                     'm9':    (0, 3, 7, 2, ),   # m9 
                     'sus2':  (0, 2, 7, ),   # sus2    1, 2, 5
                     'sus4':  (0, 5, 7, ),  # sus4     1, 4, 5
                     '7sus4': (0, 5, 7, 10, ),   # 7sus4
                     '6sus4': (0, 5, 7, 9, ),   # 6sus4
                     'Maj6':  (0, 4, 7, 9, ),     # Maj6
                     '6':     (0, 4, 7, 9, ),     # 6
                     'm6':    (0, 3, 7, 9, ),     # m6
                     '-5':    (0, 4, 6, ),    # -5
                     '+5':    (0, 4, 8, ),    # +5
                     'M7+5':  (0, 4, 8, 11, ),    # M7+5
                     'm-5':   (0, 3, 6, ),    # m-5
                     '7-5':   (0, 4, 6, 10, ),    # 7-5
                     '7+5':   (0, 4, 8, 10, ),    # 7+5
                     'M9':    (0, 4, 7, 2, ),   # M9    
                     'add2':  (0, 4, 7, 2, ),   # add2  
                     'MajAdd2': (0, 4, 7, 2, ),   # MajAdd2 
                     'm7b5':    (0, 3, 6, 10, ),    #m7b5
                     'm7-5':    (0, 3, 6, 10, ),    #m7-5
                     '46':      (7, 0, 4, ),    # 46
                     'Maj46':   (7, 0, 4, ),    # Maj46
                }
        SCALE_NATURE_MAJOR = [0, 2, 4, 5, 7, 9, 11]
        SCALE_NATURE_MINOR = [0, 2, 3, 5, 7, 8, 10]
        CHORD_SCALE = {'I': SCALE_NATURE_MAJOR,
                    'ii': SCALE_NATURE_MAJOR,
                    'iii': SCALE_NATURE_MAJOR,
                    'IV': SCALE_NATURE_MAJOR,
                    'V': SCALE_NATURE_MAJOR,
                    'vi': SCALE_NATURE_MAJOR,
                    'VII': [1, 3, 4, 6, 8, 9, 11],    # B Mixolydian: B-C#-D#-E-F#-G#-A → #1、#2、3、#4、#5、6、7
                    'VII7': [1, 3, 4, 6, 8, 9, 11],    # B Mixolydian: B-C#-D#-E-F#-G#-A → #1、#2、3、#4、#5、6、7
                    'vii': [0, 2, 4, 6, 7, 9, 10],    # Lydian
                    'vii7': [0, 2, 4, 6, 7, 9, 10],    # Lydian
                    'ii7': SCALE_NATURE_MAJOR,
                    'iii7': SCALE_NATURE_MAJOR,
                    'III':  [0, 2, 4, 5, 8, 9, 11],  
                    'III7': [0, 2, 4, 5, 8, 9, 11],
                    'vi7': SCALE_NATURE_MAJOR,
                    'Iadd9': [0, 2, 4, 5, 7, 9,   ],    #No 7th
                    'IIIadd9': [0,    4, 5, 7, 9, 11],    #No 7th
                    'IVadd9': [0, 2,    5, 7, 9, 11],    #No 7th
                    'Vadd9': [0, 2, 4,    7, 9, 11],    #No 7th
                    'VIadd9': [1, 2, 4, 6, 7,    11],    #No 7th
                    'I7': [0, 2, 4, 5, 7, 9, 10],   # C Mixolydian (C D E F G A Bb)
                    'IV7': [0, 2, 3, 5, 7, 9, 10],  # Dorian Mode：1、2、b3、4、5、6、b7
                    'V7': SCALE_NATURE_MAJOR,
                    'Vsus4': SCALE_NATURE_MAJOR,
                    'V7sus4': SCALE_NATURE_MAJOR,
                    'Isus4': SCALE_NATURE_MAJOR,
                    'IIsus4': SCALE_NATURE_MAJOR,
                    'IIIsus4': SCALE_NATURE_MAJOR,
                    'VIsus4': SCALE_NATURE_MAJOR,
                    'III7sus4': SCALE_NATURE_MAJOR,
                    'VI7sus4': SCALE_NATURE_MAJOR,
                    'IM7': SCALE_NATURE_MAJOR,
                    'IVM7': SCALE_NATURE_MAJOR,
                    'VM7': SCALE_NATURE_MAJOR,
                    'VI': [1, 2, 4, 5, 7, 9, 11],  # C Mixolydian：#1 2 3 4 5 6 7
                    'VI7': [1, 2, 4, 5, 7, 9, 11],  # C Mixolydian： #1 2 3 4 5 6 7
                    'II': [0, 2, 4, 6, 7, 9, 11],   # Lydian Mode：1、2、3、#4、5、6、7
                    'II7': [0, 2, 4, 6, 7, 9, 11],  # Lydian Mode：1、2、3、#4、5、6、7
                    'iv': [0, 2, 3, 5, 7, 8, 10],    # Aeolian
                    'iv6': [0, 2, 3, 5, 7, 8, 10],    # Aeolian
                    'iv7': [0, 2, 3, 5, 7, 8, 10],    # Aeolian
                    'v': [0, 2, 4, 5, 7, 9, 10],    # Mixolydian
                    'v7': [0, 2, 4, 5, 7, 9, 10],    # Mixolydian
                    'bv7b5': [0, 2, 4, 6, 7, 9, 11],    # Lydian Mode：1、2、3、#4、5、6、7
                    'bV': [0, 2, 4, 5, 7, 9, 10],    # Mixolydian Mode：1、2、3、4、5、6、b7
                    'bVII': [0, 2, 4, 5, 7, 9, 10],    # Mixolydian：1、2、3、4、5、6、b7
                    'bVIIM7': [0, 2, 4, 5, 7, 9, 10],    # Mixolydian
                    'Iaug': [0, 2, 4, 6, 8, 10],      # C Whole Tone: 1, 2, 3, #4, #5, #6
                    'Vaug': [1, 3, 5, 7, 9, 11],      # （G Whole Tone Scale）G-A-B-C#-D#-F → #1 #2 4 5 6 7 
                    'bVI': [0, 2, 3, 5, 7, 8, 10],    # Aeolian
                    'bVIM7': [0, 2, 3, 5, 7, 8, 10],    # Aeolian
                    'bIII': [0, 2, 3, 5, 7, 9, 10],    # bE Lydian
                    'bVIdim': [0, 2, 3, 5, 6, 8, 9, 11],    # (H/W) 8 tones
                    'ii7b5': [0, 2, 3, 5, 7, 8, 10],    # Aeolian
                    'iii7b5': [0, 2, 4, 5, 7, 9, 10],    # Mixolydian
                    'vii7b5': SCALE_NATURE_MAJOR,
                    'VIIsus4': [1,3,4,6,8,9,11],    # # B Mixolydian: B-C#-D#-E-F#-G#-A → #1、#2、3、#4、#5、6、7
                    'bIIdim': [0, 1, 3, 4, 6, 7, 9, 10],    # (W/H) 8 tones
                    'IVdim': [0, 2, 3, 5, 6, 8, 9, 11],    # (H/W) 8 tones
                    'bIIIdim': [0, 1, 3, 4, 6, 7, 9, 10],    # (W/H) 8 tones
                    'biii7': [0, 1, 3, 5, 6, 8, 10],    # Locrian Mode：1、b2、b3、4、b5、b6、b7
                    'bV': [0, 1, 3, 5, 6, 8, 10],    # Locrian Mode：1、b2、b3、4、b5、b6、b7
                    'bVIIdim': [0, 1, 3, 4, 6, 7, 9, 10],    # (W/H) 8个音
                    'bVIaug': [0, 2, 4, 6, 8, 10],     # C Whole Tone: 1, 2, 3, #4, #5, #6
                    'bVdim': [0, 2, 3, 5, 6, 8, 9, 11],    # (H/W) 8个音
                    'bIIaug': [1, 3, 5, 7, 9, 11, ],    # #C Whole Tone: C#-D#-F-G-A-B  → #1, #2, 4, 5, 6, 7
                }
        CHORD_DEGREE = {'I': [0+x for x in QUALITY_DICT['']],
                    'ii': [2+x for x in QUALITY_DICT['m']],
                    'iii': [4+x for x in QUALITY_DICT['m']],
                    'IV': [5+x for x in QUALITY_DICT['']],
                    'V': [7+x for x in QUALITY_DICT['']],
                    'vi': [9+x for x in QUALITY_DICT['m']],
                    'VII': [11+x for x in QUALITY_DICT['']],
                    'VII7': [11+x for x in QUALITY_DICT['7']],
                    'vii': [11+x for x in QUALITY_DICT['m']],
                    'vii7': [11+x for x in QUALITY_DICT['m7']],
                    'ii7': [2+x for x in QUALITY_DICT['m7']],
                    'iii7': [4+x for x in QUALITY_DICT['m7']],
                    'III': [4+x for x in QUALITY_DICT['']],
                    'III7': [4+x for x in QUALITY_DICT['7']],
                    'vi7': [9+x for x in QUALITY_DICT['m7']],
                    'Iadd9': [0+x for x in QUALITY_DICT['add9']],
                    'IIIadd9': [4+x for x in QUALITY_DICT['add9']],
                    'IVadd9': [5+x for x in QUALITY_DICT['add9']],
                    'Vadd9': [7+x for x in QUALITY_DICT['add9']],
                    'VIadd9': [9+x for x in QUALITY_DICT['add9']],
                    'I7': [0+x for x in QUALITY_DICT['7']],
                    'IV7': [5+x for x in QUALITY_DICT['7']],
                    'V7': [7+x for x in QUALITY_DICT['7']],
                    'Vsus4': [7+x for x in QUALITY_DICT['sus4']],
                    'V7sus4': [7+x for x in QUALITY_DICT['7sus4']],
                    'Isus4': [0+x for x in QUALITY_DICT['sus4']],
                    'IIsus4':  [2+x for x in QUALITY_DICT['sus4']],
                    'IIIsus4': [4+x for x in QUALITY_DICT['sus4']],
                    'VIsus4': [9+x for x in QUALITY_DICT['sus4']],
                    'III7sus4': [4+x for x in QUALITY_DICT['7sus4']],
                    'VI7sus4': [9+x for x in QUALITY_DICT['7sus4']],
                    'IM7': [0+x for x in QUALITY_DICT['M7']],
                    'IVM7': [5+x for x in QUALITY_DICT['M7']],
                    'VM7': [7+x for x in QUALITY_DICT['M7']], 
                    'VI': [9+x for x in QUALITY_DICT['']],
                    'VI7': [9+x for x in QUALITY_DICT['7']],
                    'II':  [2+x for x in QUALITY_DICT['']],
                    'II7': [2+x for x in QUALITY_DICT['7']],
                    'iv': [5+x for x in QUALITY_DICT['m']],
                    'iv6': [5+x for x in QUALITY_DICT['m6']],
                    'iv7': [5+x for x in QUALITY_DICT['m7']],
                    'v': [7+x for x in QUALITY_DICT['m']],
                    'v7': [7+x for x in QUALITY_DICT['m7']],
                    'bv7b5': [6+x for x in QUALITY_DICT['m7b5']],
                    'bV': [6+x for x in QUALITY_DICT['']],
                    'bVII': [10+x for x in QUALITY_DICT['']],
                    'bVIIM7': [10+x for x in QUALITY_DICT['M7']],
                    'Iaug': [0+x for x in QUALITY_DICT['aug']],
                    'Vaug': [7+x for x in QUALITY_DICT['aug']],
                    'bVI': [8+x for x in QUALITY_DICT['']],
                    'bVIM7': [8+x for x in QUALITY_DICT['M7']],
                    'bIII': [3+x for x in QUALITY_DICT['']],
                    'bVIdim': [8+x for x in QUALITY_DICT['dim']],
                    'ii7b5': [2+x for x in QUALITY_DICT['m7b5']],
                    'iii7b5': [4+x for x in QUALITY_DICT['m7b5']],
                    'vii7b5': [11+x for x in QUALITY_DICT['m7b5']],
                    'VIIsus4': [11+x for x in QUALITY_DICT['sus4']],
                    'bIIdim': [1+x for x in QUALITY_DICT['dim']],
                    'IVdim': [5+x for x in QUALITY_DICT['dim']],
                    'bIIIdim': [3+x for x in QUALITY_DICT['dim']],
                    'biii7': [3+x for x in QUALITY_DICT['m7']],
                    'bV': [6+x for x in QUALITY_DICT['']],
                    'bVIIdim': [10+x for x in QUALITY_DICT['dim']],
                    'bVIaug': [8+x for x in QUALITY_DICT['aug']],
                    'bVdim': [6+x for x in QUALITY_DICT['dim']],
                    'bIIaug': [1+x for x in QUALITY_DICT['aug']],
                }
        self.degrees = CHORD_DEGREE[name]
        self.scales = CHORD_SCALE[name]

    def root(self):
        return self.degrees[0]

    def __eq__(self, ch):
        return type(ch)==Chord and self.name==ch.name
    def __str__(self):
        return f"{self.name}"
    def __repr__(self):
        return f"{self.name}"

# Chords
I = Chord('I')
ii = Chord('ii')
iii = Chord('iii')
IV = Chord('IV')
V = Chord('V')
vi = Chord('vi')
VII = Chord('VII')
VII7 = Chord('VII7')
vii = Chord('vii') 
vii7 = Chord('vii7') 
ii7 = Chord('ii7')
iii7 = Chord('iii7')
III = Chord('III')
III7 = Chord('III7')
vi7 = Chord('vi7')
Iadd9 = Chord('Iadd9')
IIIadd9 = Chord('IIIadd9')
IVadd9 = Chord('IVadd9')
Vadd9 = Chord('Vadd9')
VIadd9 = Chord('VIadd9')
I7 = Chord('I7')
IV7 = Chord('IV7')
V7 = Chord('V7')
Vsus4 = Chord('Vsus4')
V7sus4 = Chord('V7sus4')
Isus4 = Chord('Isus4')
IIsus4 = Chord('IIsus4')
IIIsus4 = Chord('IIIsus4')
VIsus4 = Chord('VIsus4')
III7sus4 = Chord('III7sus4')
VI7sus4 = Chord('VI7sus4')
IM7 = Chord('IM7')
IVM7 = Chord('IVM7')
VM7 = Chord('VM7') 
VI = Chord('VI')
VI7 = Chord('VI7')
II = Chord('II')
II7 = Chord('II7')
iv = Chord('iv')
iv6 = Chord('iv6')
iv7 = Chord('iv7')
v = Chord('v')
v7 = Chord('v7')
bv7b5 = Chord('bv7b5')
bV = Chord('bV')
bVII = Chord('bVII')
bVIIM7 = Chord('bVIIM7')
Iaug = Chord('Iaug')
Vaug = Chord('Vaug')
bVI = Chord('bVI')
bVIM7 = Chord('bVIM7')
bIII = Chord('bIII')
bVIdim = Chord('bVIdim')
ii7b5 = Chord('ii7b5')
iii7b5 = Chord('iii7b5')
vii7b5 = Chord('vii7b5')
VIIsus4 = Chord('VIIsus4')
bIIdim = Chord('bIIdim')
IVdim = Chord('IVdim')
bIIIdim = Chord('bIIIdim')
biii7 = Chord('biii7')
bV = Chord('bV')
bVIIdim = Chord('bVIIdim')
bVIaug = Chord('bVIaug')
bVdim = Chord('bVdim')
bIIaug = Chord('bIIaug')

# Chord-progession templates for each kind of tempo (by exp)
# Intro
#CHORD_PROG_INTRO_TEMPLATES = {}
# Verse
CHORD_PROG_VERSE_TEMPLATES = {
    Tempo.SLOW: {
        Mode.Major: {
            8: [ [(I,vi), (IV,V), (I,vi), (IV,V), (I,vi), (IV,V), (I,vi), (IV,V)],
                 [(I,V), (vi,iii), (IV,I), (ii,V), (I,V), (vi,iii), (IV,I), (ii,V)],
                 [(I,V), (IV,V), (I,vi), (IV,V), (I,V), (IV,V), (I,vi), (IV,V)],
                 [(I,V), vi7, (IV,iii7), (ii,Vsus4), (I,V), (bVII,VI), (ii,V), I],
                 [I, III7, (vi,I), (ii7,V), iii7, vi7, (ii7,V), I],
                 [(I,vi), (ii,V7), (iii7,vi), (ii7,V7), (I,vi), (ii,V7), (iii7,vi), (ii7,V7)],
                 [(I,vi), iii7, (ii7,V), (iii7,VI), (IV,V), (iii7,vi7), ii7, V],
                 [(I,V), (IV,I), (vi,II7), (IV,V), (I,V), (IV,I), (vi,II7), (IV,V)],
                 [(vi,I), (IV,V), (vi,I), (IV,V), (vi,I), (IV,V), (vi,I), (IV,V)],
                 [(I,III), (vi,iii), (IV,I), (ii,V), (I,III), (vi,iii), (IV,I), (bIII,V)],
                 [I, vi, (IV,iii7), (ii7,V), I, vi, (IV,iii7), (ii7,V)],
                 [(I,iii), (IV,V), (I,iii), (IV,V), (vi,V), (IV,iii), (ii7,I), (bVII,V)],
                 [(I,IV), (iii,vi), (IV,vi), (ii,V), (I,IV), (I,IV), (I,IV), (V,I)],
                 [vi, iii, IV, V, vi, iii, IV, V],
                 [(I,III7), (vi7,I), (IV,I), (ii7,V), (I,III7), (vi7,I), (IV,I), (ii7,V)],
                 [(I,iii), (vi,I), (IV,V), I, (ii,iv), (iii,VI), (ii,iv), V],
                 [I, vi, (IV,V), I, I, vi, (IV,V), I],
                 [(I,IV), (V,I), (vi7,iii), (IV,V), (I,IV), (V,I), (vi7,II7), (V,I)],
                 [(I,vi), (IV,V), (I,vi), (IV,V), (I,vi), (IV,V), (I,vi), (IV,V)],
                 [(I,iii), (IV,V), (I,iii), (IV,V), (I,iii), (IV,V), (I,iii), (IV,V)],
                 ],
            16: [[I, V, vi, I, (IV,V), (iii7,vi7), ii7, (Vsus4,V),  I, V, vi, I, (IV,V), (iii7,vi7), (ii7,V), I],
                 [(I,vi), iii7, (ii7,V), (iii7,VI), (IV,V), (iii7,vi7), ii7, V,  (I,vi), iii7, (ii7,V), (iii7,VI), (IV,V), (iii7,vi7), (ii7,iv), (Vsus4,V)],
                 [(I,ii7), (iii7,vi7), (ii7,V7), I, (IV,V7), (iii7,vi7), (ii7,II7), (IV,V),  (I,ii7), (iii7,vi7), (ii7,V7), I, (IV,V7), (iii7,vi7), (ii7,V), I],
                 [I, (V,vi), I, (IV,V), III7, vi, (IV,ii), V,  I, (V,vi), I, (IV,V), III7, vi, (IV,V), I],
                 [I, iii7, vi7, iii7, (vi,iv), I, ii, V,  I, iii7, vi7, iii7, (vi,iv), I, ii, V],
                 [I, IV, V, vi, (IV,V), (I,vi), IV, V,    I, IV, V, vi, (IV,V), (I,vi), (IV,V), I],
                 [vi, iii, IV, V, vi, iii, IV, (V,III),  vi, iii, IV, V, vi, iii, IV, V],
                 [I, V, vi, iii7, IV, I, IV, V,   I, V, vi, iii7, IV, I, (IV,V), I ],
                 [(I,V), (vi,I), (IV,iii7), (ii7,V7), (I,V), (v,VI7), (ii7,V7), I,  (I,V), (vi,I), (IV,iii7), (ii7,V7), (I,V), (v,VI7), (ii7,V7), I],
                ],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
    Tempo.MID: {
        Mode.Major: {
            8: [ [I, vi, IV, V, I, vi, IV, V],
                 [I, iii, IV, V, I, V, IV, V],
                 [(IV,I), (ii,vi), (IV,V), vi, (IV,I), (ii,vi), (IV,V), I],
                 [I, iii7, vi, iii7, (ii7,V7), (I,vi), (ii7,V), I],
                 [I, V, vi, IV, I, V, IV, I],
                 [I, iii, ii, V7, I, iii, ii, V7],
                 [(ii,V7), I, (vi,ii), (IV,V), (ii,V7), I, (vi,ii), (V,I)],
                 [(ii7,V), (I,vi7), (ii7,V), (I,vi7), (ii7,iv6), (iii7,VI7), (ii7,bIIIdim), III],
                 [vi, iii, IV, V, vi, iii, IV, V], 
                 [I, vi, ii, V, iii, vi, (IV,V), I],
                 [(I,ii7), (iii7,vi), (IV,ii), V, (I,ii7), (iii7,vi), (IV,ii), V],
                 [I, V, v, ii7, IV, iii7, ii7, V],
                 [I, (IV,V), (iii7,vi), (ii7,V), I, (IV,V), (iii7,vi7), (ii7,V)],
                 [I, (IV,I), (ii7,V), I, I, (IV,I), (ii7,V), I],
                 [I, V, vi, iii, IV, I, ii7, V],
                 [(I,V), (vi,V), (IV,I), (ii,V), (IV,III7), (vi,II), (ii,V), I],
                 [(I,V), (vi,I), (IV,V), I, (I,V), (vi,I), (IV,V), I], 
                 [(I,V), vi, (ii,IV), V, (iii,vi), (ii,iv), (I,vi), (ii,V)], 
                 [I, (IV,V), I, (IV,V), I, (IV,V), (vi7,IV), (V,I)], 
                 [(I,IV), (V,I), (vi7,iii), (IV,V), (I,IV), (V,I), (vi7,II7), (V,I)], 
                 [(IV,V), vi, (IV,V), vi, (IV,V), vi, (IV,V), vi],
                 [I, vi, IV, V, iii7, vi, IV, V], 
                 [I, iii, IV, V, I, iii, IV, V,], 
                 ],
            16: [[I, V, vi, iii, IV, I, ii, V,  I, V, vi, iii, IV, I, ii, V], 
                 [I,III, vi,iii, IV,I, ii,V, I,III, vi,iii, IV,I, bIII,V],  
                 [I, iii, vi, iii7, (ii7,V), (iii7,vi), II, V,  I, iii, vi, iii7, (ii7,V), (iii7,vi), (II,V), I], 
                 [vi, iii, IV, V, vi, iii, IV, (V,III),  vi, iii, IV, V, vi, iii, IV, V],
                ],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
    Tempo.FAST: {
        Mode.Major: {
            8: [ [I, vi, IV, V, I, vi, IV, V],
                 [I, iii, IV, V, I, V, IV, V],
                 [(ii,V7), I, (vi,ii), (IV,V), (ii,V7), I, (vi,ii), (V,I)],
                 [I, iii, vi, (IV,V), I, iii, vi, (IV,V)],
                 [I, iii, vi7, iii7, (IV,V), (iii7,vi), ii7, V],
                 [I, (IV,V), I, (IV,V), I, (IV,V), (vi7,IV), (V,I)],
                 [I, vi, IV, V, I, vi, IV, V], 
                 [I, vi, IV, V, iii7, vi, IV, V],
                ],
            16: [[IV,V, iii,vi, IV,V, iii,vi, IV,V, iii,vi, IV, IV, V, V], 
                 [I, V, vi, iii, IV, I, ii, V,  I, V, vi, iii, IV, I, ii, V],
                 [I, V, vi, I, ii, IV, iii, V,   I, V, bVII, VI, ii, V, I, I],
                 [I, iii, IV, iii, (IV,V), (iii7,vi7), (IV,bVdim), V,  I, iii, IV, iii, (IV,V), (iii7,vi7), (IV,bVdim), V],
                 [I, III7, vi, iii7, (ii7,V7), (iii,VI), ii, V7, I, III7, vi, iii7, (ii7,V7), (iii,VI), ii, V], 
                 [I,V, vi,I, IV,V, I,V,   I,V, vi,I, IV,V, I,I],
                 [I, V, vi7, III, IV, I, ii7, V,  I, V, vi7, III, IV, I, ii7, V, ],
                 [I, V, vi, iii, IV, I, V, V,  I, V, vi, IV, ii, V, I, I],
                 [I, ii7, iii, vi, IV, ii7, V7, I,   I, ii7, iii, vi, IV, ii7, V7, I],
                ],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
}
# 预副歌
CHORD_PROG_PRECHORUS_TEMPLATES = {
    Tempo.SLOW: {
        Mode.Major: {
            8: [ [(IV,I), (ii,V), (iii,vi), (ii,V), (IV,I), (ii,V), (iii,vi), (ii,V)],
                 [(IV,V), (iii,vi), (ii,iii), (IV,V), (IV,V), (iii,vi), (ii,iii), (IV,V)],
                 [(ii,iii), (IV,V), (IV,I), (ii,V), (ii,iii), (IV,V), (IV,I), (ii,V)],
                 [IV, iii, V, VI7, (ii,I), (IV,I), (ii,II), V], 
                 [(ii7,V), I, (ii7,III7), (vi,I), IV, I, ii, V],
                 [(IV,V), vi7, (IV,V), vi7, (IV,V), vi7, IV, III7],
                 [ii7, V, I, vi, ii7, bVIIM7, IV, V],
                 [(vi,I), (IVM7,IVdim), (iii7,vi), (ii,V), (vi,I), (IVM7,IVdim), (iii7,vi), (ii,V)], 
                 [IV, iii7, (ii7,V7), I, (IV,iv7), (iii7,vi7), (ii7,II7), V],
                 [IV, I, IV, (vii7b5,III), (vi,I), (IV,I), II, V],
                 [(IV,V), I, (IV,V), (I,V), (IV,V), I, (IV,V), I],
                 [vi, iii, (ii,V), I, III7, vi, (IV,bv7b5), V],
                 [(IV,V), (iii7,vi7), (IV,ii7), (V,I), (IV,V), (III7,vi7), (ii7,II7), (IV,V)],
                 ],
            16: [],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
    Tempo.MID: {
        Mode.Major: {
            8: [ [IV, V, iii7, vi, ii, I, IV, V],
                 [IV, V, iii7, vi, ii, I, IV, V],
                 [IV, V, III, vi, IV, I, iv, V7],
                 [IV, iv, I, vi, ii, I, IV, V],
                 [I, IV, ii, V, III, vi, ii, V],
                 [(vi7,ii7), (V7,I), IVM7, V, (vi7,ii7), (V7,I), (IVM7,bVII), V],
                 [(ii,III), (vi7,I), (II7,V), I, (ii,III), (vi7,I), II7, V],
                 [iii7, vi7, IV, I, iii7, vi7, IV, V],
                 [IV, V, iii7, vi7, IV, iii7, (ii7,iii7), (IV,V)],
                 [IV, V, iii7, vi7, IV, iii7, ii7, V],
                 [(IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), II, V],
                 [III7, vi7, III7, (vi,I), IV, I, IV, V],
                 [iii7, vi, ii, V, iii7, vi, ii, V],
                 [vi7, iii7, vi7, iii7, vi7, iii7, (IV,ii7), V7],
                 [ii, V, iii, vi, IV, iii, ii, V],
                 [vi, iii7, ii7, I, IV, (III,vi), ii7, III7],
                 [iii, vi, iii, vi, ii7, iii7, IV, V],
                 [iii7, vi7, ii7, V7, iii7, vi7, ii7, V],
                 [IV, iii7, IV, iii7, IV, iii7, IV, V],
                 [iii, vi, iii, vi, IV, I, II, V],
                 [IV, iv, I, vi, II7, V, vi, V],
                 [III7, vi, III7, vi, IV, iv, II7, V], 
                 [iii, vi, iii, vi, iii, vi, ii, V],
                 ],
            16: [],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
    Tempo.FAST: {
        Mode.Major: {
            8: [ [IV, iii, ii, V, IV, iii, ii, V], 
                 [IV, V, iii, vi7, IV, V, (ii,II), V],
                 [IV, iv, I, vi, ii, I, IV, V],
                 [III7, vi7, III7, (vi,I), IV, I, IV, V],
                 [iii, vi, iii, vi, ii7, iii7, IV, V],
                 [iii7, vi7, ii7, V7, iii7, vi7, ii7, V], 
                 [IV, I, V, vi, ii, iii, IV, V],
                 [iii, vi7, ii, V, iii, vi7, ii, V],
                 [IV, V, III7, vi7, IV, I, ii7, V],
                 [IV, iii7, IV, iii7, IV, iii7, IV, V],
                 [iii, vi, iii, vi, IV, I, II, V],
                 [IV, iv, I, vi, II7, V, vi, V], 
                 [III7, vi, III7, vi, IV, iv, II7, V],
                 [iii, vi, iii, vi, iii, vi, ii, V],
                 ],
            16: [],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
}
# 副歌
CHORD_PROG_CHORUS_TEMPLATES = {
    Tempo.SLOW: {
        Mode.Major: {
            8: [  [(IV,V), (iii,vi), IV, V, (IV,V), (iii,vi), (IV,V), I],
                  [(IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), (II,V), I],
                  [(I,V), (vi,I), (IV,I), (ii,V), (I,V), (vi,I), (IV,bVII), I],
                  [IV, I, (ii,III), (vi,I), IV, I, (ii,V), I],
                  [(IV,V), I, (IV,V), vi, (IV,V), (I,vi), (IV,V), I],
                  [(I,IV), (V,I), (IV,II7), V, (I,IV), (V,I), (IV,V), I], 
                  [(I,V), (vi,I), (IV,I), (ii,V), (I,V), (bVII,VI), (ii,V), I],
                  [(I,V), (IV,I), (vi,V), (I,V), (I,V), (IV,I), (vi,V), I],
                  [(IV,V), (iii7,vi7), (ii7,III7), vi7, (IV,V), (III7,vi7), (ii7,V), I],
               ],
            16: [ [I, (IV,III7), vi7, (IV,V), I, (IV,III7), (vi,I), (ii,V),  I, (IV,III7), vi7, (IV,V), I, (IV,III7), (ii,V), I],
                  [I, III7, vi, I, (IV,V), (vi7,VI), (ii7,iii7), (ii7b5,V),  I, III7, vi, I, (IV,V), (vi7,VI), (ii7b5,V), I],
                  [I, V, vi, iii, IV, (iii,vi), ii7, V,  I, V, vi, iii, IV, (iii,vi), (ii7,V), I],
                  [IV, (iii,vi), IV, (ii,V7), (IV,V), (III7,vi), IV, (bVI,V),  IV, (iii,vi), IV, (ii,V7), (IV,V), (III7,vi), (IV,bVI), I],
                  [(IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), II, V,  (IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), (II,V), I],
                  [(I,V), (vi,I), (IV,I), (ii,V), (I,V), (vi,I), (IV,I), (ii,V),  (I,V), (vi,I), (IV,I), (ii,V), (I,V), (vi,I), (IV,bVII), I],
                  [IV, I, (ii,III), (vi,I), IV, I, ii, V,   IV, I, (ii,III), (vi,I), IV, I, (ii,V), I],
                  [(I,iii), vi, (IV,iii7), ii7, III7, vi, IV, V,   (I,iii), vi, (IV,iii7), ii7, III7, vi, (IV,V), vi],
                  [(I,V), vi, (IV,ii), (IIIsus4,III), IV, I, ii, V,   (I,V), vi, (IV,ii), (IIIsus4,III), IV, I, (ii,V), vi],
                  [(IV,V), (iii,vi), (ii,V), I, (IV,V), (iii,vi), ii, V,  (IV,V), (iii,vi), (ii,V), I, (IV,V), (iii,vi), (ii,V), I],
                  [(vi,iii), (IV,V), (vi,V), (IV,V), (vi,iii), (IV,V), (vi,V), (IV,V),   (vi,iii), (IV,V), (vi,V), (IV,V), (vi,iii), (IV,V), (vi,V), I],
                ],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
    Tempo.MID: {
        Mode.Major: {
            8: [  [I, V, IV, V, I, V, (IV,V), I],
                  [IV, V, iii, vi, IV, I, (IV,V), I],
                  [I, vi, IV, V, I, vi, (IV,V), I], 
                  [I, (IV,III7), vi7, (IV,V), I, (IV,III7), (ii,V), I],
                  [(IV,V7), (iii7,vi7), (IV,V7), (iii7,vi7), (IV,V7), (iii7,vi7), (ii7,V7), I],
                  [I, iii7, IV, V7, I, iii7, (ii,V7), I],
                  [I, vi, ii, V7, I, vi, (IV,iv), I],
                  [(ii,V), (iii7,vi), (ii7,V7), (I,VI7), (ii,V), (iii7,VI), (ii7,V7), I],
                  [IV, iii7, (ii7,V), I, IVM7, iii7, (ii7,V), I],
                  [(I,V), vi7, IV, V, (I,V), vi7, (IV,V), I],
                  [(IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), (II,V), I], 
                  [IV, I, (ii,III), (vi,I), IV, I, (ii,V), I], 
                  [(IV,V), I, (IV,V), vi, (IV,V), (I,vi), (IV,V), I],
                  [(I,V), (ii7,vi7), (IV,V), I, (I,V), (ii7,vi7), (IV,V), I],
                  [vi, IV, V, I, vi, IV, V, I],
               ],
            16: [ [I, IV, V, vi, ii7, I, IV, V,  I, IV, V, vi, ii7, I, (IV,V), I],
                  [(IV,V7), (iii7,vi7), (IV,V7), (iii7,vi7), (IV,V7), (iii7,vi7), ii7, V7,  (IV,V7), (iii7,vi7), (IV,V7), (iii7,vi7), (IV,V7), (iii7,vi7), (ii7,V7), I],
                  [I, V, vi, iii, IV, I, bVII, V,  I, V, vi, iii, IV, I, (bVII,V), I],
                  [I, iii7, IV, V7, I, iii7, (ii,bVII), V7,  I, iii7, IV, V7, I, iii7, (ii,V7), I],
                  [I, vi, ii, V7, I, vi, IV, iv,  I, vi, ii, V7, I, vi, (IV,iv), I],
                  [(I,V), vi7, IV, V, (I,V), vi7, IV, V,  (I,V), vi7, IV, V, (I,V), vi7, (IV,V), I],
                  [(IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), II, V,  (IV,V), (iii,vi), (ii7,V), (iii,vi), (IV,V), (III,vi), (II,V), I], 
                  [I, III7, vi, (IV,V), vi, II7, IV, V,   I, III7, vi, (IV,V), vi, II7, (IV,V), I],
                  [I, iii7, vi7, II7, IV, V7, IV, V7,  I, iii7, vi7, II7, IV, V7, (IV,V), I],
                  [IV, I, (ii,III), (vi,I), IV, I, ii, V,   IV, I, (ii,III), (vi,I), IV, I, (ii,V), I],
                  [I, iii7, vi, iii, IV, ii, (IV,II7), V,   I, iii7, vi, iii, IV, ii, (IV,V), I],
                  [IV, V, iii7, vi, ii7, iii7, IV, V,  IV, V, III, vi, ii7, V, I, I],
                  [I, V, vi7, III, IV, I, ii7, V,  I, V, vi7, III, IV, I, (ii7,V), I], 
                  [I, V, vi, iii, IV, I, II, V,   I, V, vi, iii, IV, I, (IV,V), I],
                  [I, V, vi, iii, IV, I, II7, V,  I, V, vi, iii, IV, I, (II7,V), I], 
                  [vi,iii, IV,V, vi,V, IV,V,    vi,iii, IV,V, vi,V, (IV,V), I], 
                ],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
    Tempo.FAST: {
        Mode.Major: {
            8: [  [(I,V), vi, IV, V, (I,V), vi, (IV,V), I],
                  [(I,V), (vi7,I), IV, V, (I,V), (vi7,I), (IV,V), I],
                  [vi, IV, V, I, vi, IV, V, I],
                  [vi7,IV, V,I, vi7,IV, V,I],
               ],
            16: [ [I, V, iii, vi, IV, I, bVII, V,  I, V, iii, vi, IV, I, (IV,V), I],
                  [I, V, vi, iii, (IV,V), (iii,vi), ii7, V,   I, V, vi, iii, (IV,V), (iii,vi), (ii7,V), I],
                  [I, vi, ii, V7, I, vi, IV, iv,  I, vi, ii, V7, I, vi, (IV,iv), I],
                  [(I,V), vi, IV, V, (I,V), vi, IV, V,  (I,V), vi, IV, V, (I,V), vi, (IV,V), I],
                  [(I,V), (vi7,I), IV, V, (I,V), (vi7,I), IV, V,  (I,V), (vi7,I), IV, V, (I,V), (vi7,I), (IV,V), I],
                  [IV,V, vi,I, IV,V, vi,iii, IV,V, vi,I, IV,V, I,I], 
                  [I, iii7, vi, V, IV, iii, ii7, V,  I, iii7, vi, V, IV, iii, (ii7,V), I],
                  [I, V, vi, iii, IV, I, ii7, V, I, V, vi, iii, IV, I, (ii7,V), I],
                  [I, V, vi, iii, IV, I, IV, V, I, V, vi, iii, IV, I, (IV,V), I], 
                  [I, iii, vi, I, (IV,V), (I,vi), ii, V,  I, iii, vi, I, (IV,V), (I,vi), (ii,V), I],
                  [IV, V, iii7, vi, ii7, iii7, IV, V,  IV, V, III, vi, ii7, V, I, I],
                  [I,V, vi7,vi7, IV,(iii,vi7), II7,V,   I,V, vi7,vi7, IV,(iii,vi7), (ii7,V), I],
                  [I,V, vi,iii, IV,(iii,vi7), ii7,V,   I,V, vi,iii, IV,(iii,vi7), (ii7,V),I], 
                  [IV,V, iii,vi, IV,V, I,I,   IV,V, (iii,III),vi, IV,V, I,I], 
                  [I, V, vi7, III, IV, I, ii7, V,  I, V, vi7, III, IV, I, (ii7,V), I],
                  [I, V, vi, iii, (IV,V), (iii,vi), IV, V,   I, V, vi, iii, (IV,V), (iii,vi), (IV,V),I],
                  [I, iii7, vi7, (v7,I7), (IV,V), (iii7,vi7), ii7, V,   I, iii7, vi7, (v7,I7), (IV,V), (iii7,vi7), (ii7,V), I],
                  [I, vi, IV, V, III7, vi7, ii7, V,  I, vi, IV, V, III7, vi7, (ii7,V), I],
                  [IV, V, iii, vi, IV, V, iii, vi,   IV, V, iii, vi, IV, I, (IV,V), I],
                  [I, V, vi, iii, IV, I, II, V,   I, V, vi, iii, IV, I, (IV,V), I],
                  [I, vi, IV, (II,V), I, vi, IV, V,   I, vi, IV, (II,V), I, vi, (IV,V), I], 
                  [I, V, vi, iii, IV, I, II7, V,  I, V, vi, iii, IV, I, (II7,V), I], 
                  [I, vi, IV, V, I, vi, IV, V,   I, vi, IV, V, I, vi, (IV,V), I], 
                ],
            },
        Mode.Minor: {
            8: [],
            16: [],
            },
    },
} 
# Outro
#CHORD_PROG_OUTRO_TEMPLATES = {}

class ChordprogGenerator:
    """
        For a given structure (list of section) / tempo_class / mode, 
        generate chord-progression for each kind of section
    """ 
    def __init__(self, structure, tempo_class, mode, seed, _log=print):
        self.structure = structure
        self.tempo_class = tempo_class
        self.mode = mode
        self.rand = random.Random(seed)
        self._log = _log

    def gen(self):
        """
        Returns:
          [  [ch1, ch2 ... ], # chord progression of SectionType1
             [ch1, ch2 ... ], # chord progression of SectionType2
             [ch1, ch2 ... ], # chord progression of SectionType3
            ...
          ]
        """ 
        # section info
        n_struct = len(self.structure)
        sec_chord_prog_seq = [None for k in range(n_struct)]  # for return 
        chorus_chord_prog = []  # only used in this method for intro/outro

        # ---------------- Loop 1st time ----------------
        # only deal with vocal part(verse/pre-chorus/chorus)
        # because chords of intro/outro may COPY vocal part.
        i = 0
        while i < n_struct:
            # section info
            sec = self.structure[i]
            stype = sec.section_type
            nbar = sec.num_bar
            next_sec = self.structure[i+1] if i+1 < n_struct else None

            assert nbar in (8,16), f"Error: nbar in (8,16), but got {nbar}."

            # check section type
            if stype in (SectionType.VERSE, SectionType.PRECHORUS, \
                         SectionType.CHORUS):
                # Vocal part (verse, pre-chorus, chorus)
                if next_sec and sec == next_sec:
                    # 2 same continuous sections
                    tar_chord_progs = self._choose_chord_prog(stype, nbar, nsec=2)
                    # save chord progression
                    sec_chord_prog_seq[i] = tar_chord_progs[0]
                    sec_chord_prog_seq[i+1] = tar_chord_progs[1]
                    # update 
                    i += 1
                else:
                    # 1 section
                    # pick chord progression(random)
                    tar_chord_progs = self._choose_chord_prog(stype, nbar, nsec=1)

                    # save chord progression
                    sec_chord_prog_seq[i] = tar_chord_progs[0] 

                # save chorus-chord-progression
                if stype == SectionType.CHORUS:
                    chorus_chord_prog = sec_chord_prog_seq[i]
            # update 
            i += 1

        #---------------- Loop 2nd time ----------------
        # only deal with instrument part(intro/outro)
        # because chords of intro/outro may COPY vocal part.
        for i, sec in enumerate(self.structure):
            # section info
            stype = sec.section_type
            nbar = sec.num_bar
            r = self.rand.random()

            # INTRO & OUTRO
            if stype in (SectionType.INTRO, SectionType.OUTRO):
                # Intro/Outro may copy Chorus
                nbar_chorus = len(chorus_chord_prog)
                if nbar == nbar_chorus:
                    # same num_bar, copy chorus-chords 
                    sec_chord_prog_seq[i] = chorus_chord_prog
                elif nbar==8 and nbar_chorus==16:
                    # half of chorus-chords 
                    if r < 0.5 and stype != SectionType.OUTRO:
                        sec_chord_prog_seq[i] = chorus_chord_prog[:8]
                    else:
                        sec_chord_prog_seq[i] = chorus_chord_prog[8:]
                elif nbar==16 and nbar_chorus==8:
                    # double chorus-chords 
                    sec_chord_prog_seq[i] = chorus_chord_prog + chorus_chord_prog
                else:
                    raise NotImplementedError(f"Error: shouldn't happen! nbar={nbar} vs nbar_chorus={nbar_chorus}.")

        # return
        return sec_chord_prog_seq

    def gen_by_chords(self, chordprog_list):
        """ Add intro/outro chord progression 
        Returns:
          [  [ch1, ch2 ... ], # chord progression of SectionType1
             [ch1, ch2 ... ], # chord progression of SectionType2
             [ch1, ch2 ... ], # chord progression of SectionType3
            ...
          ]
        """
        sec_chord_prog_seq = []

        # str -> Chord
        for sec_chords in chordprog_list:
            chs = []
            for x in sec_chords:
                if '(' in x or "[" in x:
                    x12 = x.strip('(').strip('[').strip(')').strip(']')
                    x12 = x12.split(',')
                    chs.append([Chord(x12[0].strip()), Chord(x12[1].strip())])
                elif type(x) in (tuple, list):
                    chs.append([Chord(y) for y in x])
                else:
                    chs.append(Chord(x))

            sec_chord_prog_seq.append(chs)

        # add Intro/Outro with ChorusChords
        if len(sec_chord_prog_seq[-1]) == 8:
            sec_chord_prog_seq = [sec_chord_prog_seq[-1],] + sec_chord_prog_seq
            sec_chord_prog_seq = sec_chord_prog_seq + [sec_chord_prog_seq[-1],]
        else:
            sec_chord_prog_seq = [sec_chord_prog_seq[-1][-8:],] + sec_chord_prog_seq
            sec_chord_prog_seq = sec_chord_prog_seq + [sec_chord_prog_seq[-1][-8:],]

        # return
        return sec_chord_prog_seq


    def _choose_chord_prog(self, sec_type, nbar, nsec=1):
        # random        
        assert nbar in (8, 16), f"Error: nbar not in (8,16), but got {nbar}"
        assert nsec in (1, 2), f"Error: nsec not in (1,2), but got {nsec}"
        if sec_type == SectionType.INTRO:
            #candis = CHORD_PROG_INTRO_TEMPLATES[self.tempo_class][self.mode][nbar]
            raise NotImplementedError("TODO")
        elif sec_type == SectionType.VERSE:
            candis = CHORD_PROG_VERSE_TEMPLATES[self.tempo_class][self.mode]
        elif sec_type == SectionType.PRECHORUS:
            candis = CHORD_PROG_PRECHORUS_TEMPLATES[self.tempo_class][self.mode]
        elif sec_type == SectionType.CHORUS:
            candis = CHORD_PROG_CHORUS_TEMPLATES[self.tempo_class][self.mode]
        elif sec_type == SectionType.OUTRO:
            #candis = CHORD_PROG_OUTRO_TEMPLATES[self.tempo_class][self.mode][nbar]
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError(f"FIXME: unknown sec_type == {sec_type}")

        # choose
        if nbar == 16:
            # For 16-bar section, use an 16-bar template
            sel_chord_prog = self.rand.choice(candis[16])
            return [sel_chord_prog for n in range(nsec)]

        elif nbar == 8:
            if nsec == 1:
                # For 8-bar section, use an 8-bar template
                sel_chord_prog = self.rand.choice(candis[8])
                return [sel_chord_prog,]
            elif nsec == 2: 
                # For 2 8-bar sections, use an 16-bar template or 2 8-bar template
                sel_chord_prog = self.rand.choice(candis[8] + candis[16])
                if len(sel_chord_prog) == 8:
                    return [sel_chord_prog, sel_chord_prog]
                elif len(sel_chord_prog) == 16:
                    return [sel_chord_prog[:8], sel_chord_prog[8:]]
                else:
                    raise NotImplementedError(f"Error: unknown len(sel_chord_prog) = {len(sel_chord_prog)}")
        
# Check
for tmplt_name, ch_tmplts in [("verse_chord_tmplts", CHORD_PROG_VERSE_TEMPLATES),
                              ("pre-ch_chord_tmplts", CHORD_PROG_PRECHORUS_TEMPLATES),
                              ("chorus_chord_tmplts", CHORD_PROG_CHORUS_TEMPLATES) ]:
    print(f" > Checking {tmplt_name}...")
    assert Tempo.SLOW in ch_tmplts, "Error: Tempo.SLOW not found."
    assert Tempo.MID in ch_tmplts, "Error: Tempo.MID not found."
    assert Tempo.FAST in ch_tmplts, "Error: Tempo.FAST not found."
    assert Mode.Major in ch_tmplts[Tempo.SLOW], "Error: Mode.Major not found in Tempo.SLOW."
    assert Mode.Minor in ch_tmplts[Tempo.SLOW], "Error: Mode.Minor not found in Tempo.SLOW."
    assert Mode.Major in ch_tmplts[Tempo.MID], "Error: Mode.Major not found in Tempo.MID."
    assert Mode.Minor in ch_tmplts[Tempo.MID], "Error: Mode.Minor not found in Tempo.MID."
    assert Mode.Major in ch_tmplts[Tempo.FAST], "Error: Mode.Major not found in Tempo.FAST."
    assert Mode.Minor in ch_tmplts[Tempo.FAST], "Error: Mode.Minor not found in Tempo.FAST."
    assert 8 in ch_tmplts[Tempo.SLOW][Mode.Major], "Error: 8 not found in Tempo.SLOW & Mode.Major."
    assert 16 in ch_tmplts[Tempo.SLOW][Mode.Major], "Error: 16 not found in Tempo.SLOW & Mode.Major."
    assert 8 in ch_tmplts[Tempo.SLOW][Mode.Minor], "Error: 8 not found in Tempo.SLOW & Mode.Minor."
    assert 16 in ch_tmplts[Tempo.SLOW][Mode.Minor], "Error: 16 not found in Tempo.SLOW & Mode.Minor."
    assert 8 in ch_tmplts[Tempo.MID][Mode.Major], "Error: 8 not found in Tempo.MID & Mode.Major."
    assert 16 in ch_tmplts[Tempo.MID][Mode.Major], "Error: 16 not found in Tempo.MID & Mode.Major."
    assert 8 in ch_tmplts[Tempo.MID][Mode.Minor], "Error: 8 not found in Tempo.MID & Mode.Minor."
    assert 16 in ch_tmplts[Tempo.MID][Mode.Minor], "Error: 16 not found in Tempo.MID & Mode.Minor."
    assert 8 in ch_tmplts[Tempo.FAST][Mode.Major], "Error: 8 not found in Tempo.FAST & Mode.Major."
    assert 16 in ch_tmplts[Tempo.FAST][Mode.Major], "Error: 16 not found in Tempo.FAST & Mode.Major."
    assert 8 in ch_tmplts[Tempo.FAST][Mode.Minor], "Error: 8 not found in Tempo.FAST & Mode.Minor."
    assert 16 in ch_tmplts[Tempo.FAST][Mode.Minor], "Error: 16 not found in Tempo.FAST & Mode.Minor."
    for t in (Tempo.SLOW, Tempo.MID, Tempo.FAST):
        for m in (Mode.Major, Mode.Minor):
            for l in (8, 16):
                assert type(ch_tmplts[t][m][l])==list, f"Error: template[{t.name}][{m.name}][{l}] is not list."
                for j, ch_list in enumerate(ch_tmplts[t][m][l]):
                    assert len(ch_list)==l, f"Error: template[{t.name}][{m.name}][{l}][{j}] length is not {l}, but got {len(ch_list)}."
                    for k, ch in enumerate(ch_list):
                        if type(ch) == Chord:
                            pass
                        elif type(ch) in (tuple, list):
                            for ach in ch:
                                assert type(ach)==Chord, f"Error: unknown chord '{ch}' in template[{t.name}][{m.name}][{l}][{j}][{k}]"
                        else:
                            raise NotImplementedError(f"Error: unknown chord '{ch}' in template[{t.name}][{m.name}][{l}][{j}][{k}]")
# Augment: 8+8bar -> 16bar
AUGMENT_CHORD_PROG = True
if AUGMENT_CHORD_PROG:
    for tmplt_name, ch_tmplts in [("verse_chord_tmplts", CHORD_PROG_VERSE_TEMPLATES),
                                  ("pre-ch_chord_tmplts", CHORD_PROG_PRECHORUS_TEMPLATES),
                                  ("chorus_chord_tmplts", CHORD_PROG_CHORUS_TEMPLATES) ]:
        print(f"     > Augment: {tmplt_name}...")
        for t in (Tempo.SLOW, Tempo.MID, Tempo.FAST):
            for m in (Mode.Major, Mode.Minor):
                candi8 = ch_tmplts[t][m][8]
                candi16 = ch_tmplts[t][m][16]
                for chs8 in candi8:
                    b_exist = False
                    for chs16 in candi16:
                        if chs8 in (chs16[:8], chs16[8:]):
                            b_exist = True
                            break 
                    if not b_exist:
                        ch_tmplts[t][m][16].append(chs8+chs8)

# summary all chords 
SUMMARY_ALL_CHORDS = True
if SUMMARY_ALL_CHORDS:
    all_chords = []
    for tmplt_name, ch_tmplts in [("verse_chord_tmplts", CHORD_PROG_VERSE_TEMPLATES),
                                  ("pre-ch_chord_tmplts", CHORD_PROG_PRECHORUS_TEMPLATES),
                                  ("chorus_chord_tmplts", CHORD_PROG_CHORUS_TEMPLATES) ]:
        for t in (Tempo.SLOW, Tempo.MID, Tempo.FAST):
            for m in (Mode.Major, Mode.Minor ):
                candi8 = ch_tmplts[t][m][8]
                candi16 = ch_tmplts[t][m][16]
                for chs in candi8 + candi16:
                    for ch in chs:
                        if type(ch) in (list, tuple):
                            for x in ch:
                                if x not in all_chords:
                                    all_chords.append(x)
                        else:
                            if ch not in all_chords:
                                all_chords.append(ch) 
    print(f"     > All used chord: ", [x.name for x in all_chords])



    
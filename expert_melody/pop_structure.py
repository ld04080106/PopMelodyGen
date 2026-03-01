#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pop_structure.py

Song-level structure generation for pop melodies.

Responsibilities:
  - Sample structure templates (Intro–A–[C]–B–Outro).
  - Convert templates into bar-level section boundaries.
  - Expose a compact structure representation used downstream by harmony/rhythm/melody.

Typical functions:
  - sample_structure(cfg, rng) -> Structure
  - structure_to_bar_map(structure) -> list[SectionTag]

Author:      <ludy>
Modified:    2026/01/16
Licence:     <MIT>
"""

from expert_melody.pop_settings import Tempo, Key, Mode, SectionType
from expert_melody.pop_settings import MAX_SONG_SEC, MIN_SONG_SEC, IS_DEBUG

from dataclasses import dataclass
from enum import Enum 
import random

##-----------------------------
# Structure
##-----------------------------
""" Currently,
(1) Only the following is available:
- Intro
- Verse (as A)
- Prechorus (as C)
- Chorus (as B)
- Outro
(2) Rules:
- MUST appear: Intro、A、B、Outro
- A MUST be before B
- C is optional, MUST between A & B, MUST not be longer than A/B
- A/B can repeat itself. (A→AA、B→BB)
(3) Note:
A8/B8 are excluded because pop songs typically repeat verses and choruses, 
and a single 8-bar section is too short to establish the expected repetition. 
Similarly, 16-bar Pre-chorus (C16) is not used as the pre-chorus is normally a short transitional section.
"""

@dataclass
class ASection:
    section_type: SectionType
    num_bar: int

    def __repr__(self):
        return f"[{self.section_type.name}-{self.num_bar}]"
    def __eq__(self, asec):
        return self.section_type == asec.section_type

sec_intro_8 = ASection(SectionType.INTRO, 8)
sec_A_8 = ASection(SectionType.VERSE, 8)
sec_A_16 = ASection(SectionType.VERSE, 16)
sec_C_8 = ASection(SectionType.PRECHORUS, 8)
sec_B_8 = ASection(SectionType.CHORUS, 8)
sec_B_16 = ASection(SectionType.CHORUS, 16)
sec_outro_8 = ASection(SectionType.OUTRO, 8)

# Structure templates for each kind of tempo (by exp)
STRUCTURE_TEMPLATES = { 
    Tempo.SLOW: [
        # A - B
        ( sec_intro_8, sec_A_16, sec_B_16, sec_outro_8 ), #A16B16
        # A - C - B
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_16, sec_outro_8 ),
        # A - A - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_16, sec_outro_8 ),
        # A - A - C - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_16, sec_outro_8 ),
        # A - B - B
        ( sec_intro_8, sec_A_16, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - C - B - B
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - A - B - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_16, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - A - C - B - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
    ],
    Tempo.MID: [
        # A - B
        ( sec_intro_8, sec_A_16, sec_B_16, sec_outro_8 ),
        # A - C - B
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_16, sec_outro_8 ),
        # A - A - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_16, sec_outro_8 ),
        # A - A - C - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_16, sec_outro_8 ),
        # A - B - B
        ( sec_intro_8, sec_A_16, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - C - B - B
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - A - B - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_16, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - A - C - B - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
    ],
    Tempo.FAST: [
        # A - B
        ( sec_intro_8, sec_A_16, sec_B_16, sec_outro_8 ),
        # A - C - B
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_16, sec_outro_8 ),
        # A - A - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_16, sec_outro_8 ),
        # A - A - C - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_16, sec_outro_8 ),
        # A - B - B
        ( sec_intro_8, sec_A_16, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - C - B - B
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - A - B - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_8, sec_A_8, sec_B_16, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_B_16, sec_B_16, sec_outro_8 ),
        # A - A - C - B - B
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_8, sec_B_8, sec_outro_8 ),
        ( sec_intro_8, sec_A_8, sec_A_8, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
        ( sec_intro_8, sec_A_16, sec_A_16, sec_C_8, sec_B_16, sec_B_16, sec_outro_8 ),
    ],
}

class StructureGenerator:
    """
        For a given tempo_class & bpm, choose a structure template
        The duration(seconds) of song should be in range.
    """ 
    def __init__(self, tempo_class, bpm, seed, _log=print):
        self.tempo_class = tempo_class
        self.bpm = bpm
        self.rand = random.Random(seed)
        self._log = _log

    def gen(self):
        """
            Returns: structure(list of section)
        """ 
        # find all candidates
        struct_candidates = []

        # candidate's duration MUST be in range (Shouldn't be too long!)
        all_structs = STRUCTURE_TEMPLATES[self.tempo_class]
        for struct in all_structs:
            # calculate seconds 
            dur_sec = self.calc_song_duration_seconds(struct)

            # if IS_DEBUG:
            #     self._log(f"    > Structure: dur_sec={dur_sec} ([{MIN_SONG_SEC}s, {MAX_SONG_SEC}s)")

            # check duration
            if MIN_SONG_SEC <= dur_sec <= MAX_SONG_SEC:
                # in range 
                struct_candidates.append(struct)

        if IS_DEBUG:
            self._log(f"    > Structure: Candidates of structure={len(struct_candidates)} ([{MIN_SONG_SEC}s, {MAX_SONG_SEC}s)")

        # choose one
        tar_structure = self.rand.choice(struct_candidates)

        if IS_DEBUG:
            self._log(f"    > Structure: Choose structure={tar_structure}")

        return tar_structure

    def gen_by_str(self, struct_str):
        """
            Returns: structure(list of section)
        """
        tar_structure = [sec_intro_8,]

        while len(struct_str) > 0:
            if struct_str.startswith("A8"):
                tar_structure.append(sec_A_8)
                struct_str = struct_str[2:]
            elif struct_str.startswith("A16"):
                tar_structure.append(sec_A_16)
                struct_str = struct_str[3:]
            elif struct_str.startswith("C8"):
                tar_structure.append(sec_C_8)
                struct_str = struct_str[2:]
            elif struct_str.startswith("C16"):
                tar_structure.append(sec_C_16)
                struct_str = struct_str[3:]
            elif struct_str.startswith("B8"):
                tar_structure.append(sec_B_8)
                struct_str = struct_str[2:]
            elif struct_str.startswith("B16"):
                tar_structure.append(sec_B_16)
                struct_str = struct_str[3:]
            else:
                raise ValueError(f"Error: unknown struct_str = {struct_str}")

        tar_structure.append(sec_outro_8)

        return tar_structure


    def calc_song_duration_seconds(self, structure, beats_per_bar=4):
        """ 
            Calculate song length by structure and bpm
            Default time signature=4/4
        """
        total_bars = sum(sec.num_bar for sec in structure) 
        total_beats = total_bars * beats_per_bar
        duration_sec = total_beats * 60.0 / self.bpm 
        return duration_sec


    
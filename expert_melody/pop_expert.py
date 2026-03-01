#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pop_expert.py

Entry point for the rule-based expert melody synthesizer.

This module orchestrates the full synthesis pipeline:
(global settings) -> (structure) -> (Harmony) -> (rhythm) -> (pitch) -> (MIDI+metadata).

Public API:
  - ExpertMelodyGenerator: main class for generating one song/sample.

Inputs:
  - musical conditions
  - seeds
Outputs:
  - MIDI file (.mid) and metadata JSON (.json) 

Author:      <ludy>
Modified:    2026/01/22
Licence:     <MIT>
"""

from expert_melody.pop_settings import Tempo, SectionType, Key, Mode
from expert_melody.pop_settings import TEMPO_RANGES, P_MODE_TEMPO, VOCAL_REGISTERS, IS_DEBUG
from expert_melody.pop_structure import StructureGenerator
from expert_melody.pop_chordprog import ChordprogGenerator
from expert_melody.pop_rhythm import RhythmGenerator
from expert_melody.pop_melody import MelodyGenerator

import random

class ExpertMelodyGenerator:
    """
    Rule-based melody expert generator:
    - picks bpm, key, mode, vocal register, structure, chord-progression
    - for each section
        - generates rhythm patterns (via RhythmGenerator)
        - samples melody degrees & pitches under harmony / scale constraints (via MelodyGenerator)
        - returns Melody + metadata.
    """
    def __init__(self, tempo_class, gender, 
                       seed_global, seed_rhythm, seed_pitch, 
                       key=None, mode=None, struct=None, 
                       chordprog_list=None, _log=print):
        self.tempo_class = tempo_class
        self.gender = gender
        self.seed_global = seed_global
        self.seed_rhythm = seed_rhythm
        self.seed_pitch = seed_pitch
        self.key = key
        self.mode = mode 
        self.struct = struct 
        self.chordprog_list = chordprog_list 
        self._log = _log

        self.rand_global = random.Random(seed_global)
        #self.rand_rhythm = random.Random(seed_rhythm)
        #self.rand_pitch = random.Random(seed_pitch)

    def generate_song(self, n_result_per_cond=1): 
        ## ---------------------------------------------------
        ## 1. Random tempo(bpm)
        bpm_min, bpm_max = TEMPO_RANGES[self.tempo_class]
        tar_bpm = self.rand_global.randint(bpm_min, bpm_max)

        if IS_DEBUG:
            self._log(f"    > Bpm: choose {tar_bpm} from [{bpm_min}, {bpm_max}] (TEMPO={self.tempo_class.name})")

        ## ---------------------------------------------------
        ## 2. Random key 
        tonics = [Key.C, Key.bD, Key.D, Key.bE, Key.E, Key.F, Key.bG, Key.G, Key.bA, Key.A, Key.bB, Key.B]
        if self.key is None:
            tar_key = self.rand_global.choice(tonics)
        else:
            tar_key = tonics[self.key]

        if IS_DEBUG:
            self._log(f"    > Key: choose {tar_key.name} from 12 keys.")

        ## ---------------------------------------------------
        ## 3. Random mode
        prior = P_MODE_TEMPO[self.tempo_class]
        if self.mode is None:
            r = self.rand_global.random()
            tar_mode = Mode.Major if r < prior[Mode.Major] else Mode.Minor
        else:
            if self.mode == 0:
                tar_mode = Mode.Major
            elif self.mode == 1:
                tar_mode = Mode.Minor 
            else:
                raise NotImplementedError(f"Error: unknown self.mode = {self.mode}")

        if IS_DEBUG:
            self._log(f"    > Mode: choose {tar_mode.name} by P={prior}")

        ## ---------------------------------------------------
        ## 4. Vocal register (MIDI pitch) 
        vregister_candidates = VOCAL_REGISTERS[self.gender]
        vocal_register = self.rand_global.choice(vregister_candidates)

        if IS_DEBUG:
            self._log(f"    > VocalRegister: choose {vocal_register} (Gender={self.gender})")

        ## ---------------------------------------------------
        ## 5. Structure (List of section)
        # create generator
        sg = StructureGenerator(self.tempo_class, tar_bpm, self.seed_global, self._log)
        # generate
        if self.struct is None:
            tar_structure = sg.gen()
        else:
            tar_structure = sg.gen_by_str(self.struct)

        ## ---------------------------------------------------
        ## 6. Quantize (smallest unit/shortest duration)
        # if self.tempo_class == Tempo.SLOW:
        #     # Slow quantization = 16th Note
        #     quant_unit = 0.25
        # else:
        #     # Normal quantization = 8th Note
        #     quant_unit = 0.5
        quant_unit = 0.5    # 0.5(8th Note) only.

        if IS_DEBUG:
            self._log(f"    > Quant: quant_unit={quant_unit} (Tempo={self.tempo_class.name})")

        ## ---------------------------------------------------
        ## 7. Random chord-progression for each kind of section
        # create generator
        cpg = ChordprogGenerator(tar_structure, self.tempo_class, 
                                 tar_mode, self.seed_global, self._log)
        # generate
        if self.chordprog_list is None:
            chord_prog_seq = cpg.gen()
        else:
            chord_prog_seq = cpg.gen_by_chords(self.chordprog_list)

        if IS_DEBUG:
            self._log(f"    > ChordProgress: chord_prog_seq={chord_prog_seq}")

        ## ---------------------------------------------------
        # generate rhythm
        MAX_RETRY = 5
        tried = 0
        while MAX_RETRY > 0:
            try:
                ## 8. Random rhythm patterns for each section
                rg = RhythmGenerator(tar_structure, self.tempo_class, quant_unit, chord_prog_seq, 
                                     self.seed_rhythm, self._log)
                rhythm_seq = rg.gen()
                MAX_RETRY = -1
            except:
                MAX_RETRY -= 1
                tried += 1
                self.seed_rhythm += tried*10000
                import traceback
                #traceback.print_exc()
                # Errors may happen when allocating syllables or sampling patterns due to various reasons.
                # For example, if syllables for each phase is sampled too much, there may be not enough spaces for allocating because quant_unit is fixed to 0.5. 
                # Just ignore and retry.

        ## ---------------------------------------------------
        # generate pitch
        MAX_RETRY = 5
        tried = 0
        while MAX_RETRY > 0:
            try:
                ## 9. Random melody for each section
                mg = MelodyGenerator(tar_structure, chord_prog_seq, rhythm_seq,
                                     self.tempo_class, tar_key, tar_mode, 
                                     vocal_register, quant_unit, self.seed_pitch, self._log)
                melody_seq, song_meta = mg.gen()

                if IS_DEBUG:
                    self._log(f"    ---------------------- Ouput Melody ------------------------")
                    for i, note_seq in enumerate(melody_seq):
                        self._log(f"     > Section {i+1} : {note_seq}")
                        self._log("    ----------------------------------------------")
                        # check pitch for debug
                        if note_seq != None:
                            for x in note_seq.notes:
                                if x.pitch < vocal_register[0]-2 or x.pitch > vocal_register[1]+2:
                                    raise ValueError(f"Error: pitch({x.pitch}) out of range({vocal_register})")
                    self._log(f"     > song_meta : {song_meta}")
                MAX_RETRY = -1
            except:
                MAX_RETRY -= 1 
                tried += 1
                self.seed_pitch += tried*10000
                import traceback
                #traceback.print_exc()
        # Add meta
        song_meta["bpm"] = tar_bpm
        song_meta["gender"] = self.gender
        song_meta["seed_global"] = self.seed_global
        song_meta["seed_rhythm"] = self.seed_rhythm
        song_meta["seed_pitch"] = self.seed_pitch

        # return
        return melody_seq, song_meta


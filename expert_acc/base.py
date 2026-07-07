#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expert_acc/base.py

Base pattern for accompaniment

Author:      <ludy>
Modified:    2026/06/22
Licence:     <MIT>
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List
from expert_melody.pop_melody import Scale
from expert_melody.pop_chordprog import Chord
from expert_melody.pop_settings import Tempo, SectionType, Key, Mode
from expert_acc import InstrGroupType, INSTRUMENT_INFO

LEN_4NOTE = 48
LEN_2NOTE = 96

##----------------------------------------------
# Class
##---------------------------------------------- 
@dataclass
class NoteEvent:
    start: int   # 64th note(triple): 64th=3,32th=6,16th=12,8th=24,4th=48,2nd=96,whole=192
    dur: int     # 64th note(triple): 64th=3,32th=6,16th=12,8th=24,4th=48,2nd=96,whole=192
    pitch: int   # GM drum pitch
    vel: int

    def copy(self):
        return NoteEvent(start=self.start, dur=self.dur, pitch=self.pitch, vel=self.vel)

    def __repr__(self):
        return f"[{self.pitch}, {self.start}+{self.dur}, v={self.vel}]"

@dataclass
class BarPattern:
    notes: List[NoteEvent]

    def gen_bar(self, ch, key, mode, instr_group, rng_low, rng_high, cache_pitch):
        # Generate a bar 
        # method by instrument-group
        if instr_group == InstrGroupType.Drum:
            tar_notes = self._gen_drum_bar()
            cache_pitch = None
        elif instr_group == InstrGroupType.Bass:
            tar_notes, cache_pitch = self._gen_bass_bar(ch, key, mode, rng_low, rng_high, pre_root_pitch=cache_pitch)
        else:
            tar_notes, cache_pitch = self._gen_acc_bar(ch, key, mode, rng_low, rng_high, pre_chord_pitches=cache_pitch)
       
        return tar_notes, cache_pitch

    def _gen_drum_bar(self):
        # copy notes 
        drum_notes = []
        for nt in self.notes:
            drum_notes.append(nt.copy())
        return drum_notes 

    def _gen_bass_bar(self, ch, key, mode, rng_low, rng_high, pre_root_pitch=0):
        # as root note
        if '(' in ch or '[' in ch:
            ch1, ch2 = ch[1:-1].split(', ')
            root1_pitches, chord1_pitches = self._chord_pitches(ch1, key, mode, rng_low, rng_high)
            root2_pitches, chord2_pitches = self._chord_pitches(ch2, key, mode, rng_low, rng_high)
        elif type(ch) in (list, tuple):
            ch1, ch2 = ch[0], ch[1] 
            root1_pitches, chord1_pitches = self._chord_pitches(ch1, key, mode, rng_low, rng_high)
            root2_pitches, chord2_pitches = self._chord_pitches(ch2, key, mode, rng_low, rng_high)
        else:
            root_pitches, chord_pitches = self._chord_pitches(ch, key, mode, rng_low, rng_high)
            root1_pitches = root2_pitches = root_pitches
            chord1_pitches = chord2_pitches = chord_pitches
 
        # find nearest pitch with 'pre_root_pitch' among 'root_pitches'
        pre_root1_pitch = 0 if pre_root_pitch==None else pre_root_pitch
        min_dis = 9999
        tar_root1_pitch = root1_pitches[0]
        for p in root1_pitches:
            dis = abs(p - pre_root1_pitch)
            if dis < min_dis:
                min_dis = dis
                tar_root1_pitch = p
        pre_root2_pitch = tar_root1_pitch
        min_dis = 9999
        tar_root2_pitch = root2_pitches[0]
        for p in root2_pitches:
            dis = abs(p - pre_root2_pitch)
            if dis < min_dis:
                min_dis = dis
                tar_root2_pitch = p
        #print(" root_pitches, chord_pitches =", root_pitches, chord_pitches, "tar_root_pitch =", tar_root_pitch)
        # index of root_pitch in chord_pitch
        idx_root1 = chord1_pitches.index(tar_root1_pitch)
        idx_root2 = chord2_pitches.index(tar_root2_pitch)

        # pattern_pitch -> chord_pitch
        bass_notes = []
        
        for nt in self.notes:
            # read note info
            in_pitch = nt.pitch
            st = nt.start
            dur = nt.dur

            # 前2拍 or 后2拍
            if st < LEN_2NOTE:
                tar_root_pitch = tar_root1_pitch
                chord_pitches = chord1_pitches
                idx_root = idx_root1
            else:
                tar_root_pitch = tar_root2_pitch
                chord_pitches = chord2_pitches
                idx_root = idx_root2

            # in chord 
            # BassPattern中允许的音高
            # : 36=root, 40=3rd, 43=5th, 48=root+8th
            # : 31=5th, 28=3th
            np = len(chord_pitches)
            if in_pitch==36:
                tar_pitch = tar_root_pitch
            elif in_pitch==40:
                tar_pitch = chord_pitches[idx_root+1] if idx_root+1 < np else tar_root_pitch
            elif in_pitch==43:
                tar_pitch = chord_pitches[idx_root+2] if idx_root+2 < np else tar_root_pitch
            elif in_pitch==48:
                tar_pitch = chord_pitches[idx_root+3] if idx_root+3 < np else tar_root_pitch
            elif in_pitch==31:
                tar_pitch = chord_pitches[idx_root-1] if idx_root-1 >= 0 else tar_root_pitch
            elif in_pitch==28:
                tar_pitch = chord_pitches[idx_root-2] if idx_root-2 >= 0 else tar_root_pitch
            else:
                raise NotImplementedError(f"Error: unknown in_pitch={in_pitch}")

            tar_note = nt.copy()
            tar_note.pitch = tar_pitch

            # save
            bass_notes.append(tar_note)
        # update 
        pre_root_pitch = tar_root2_pitch

        # return
        return bass_notes, pre_root_pitch

    def _gen_acc_bar(self, ch, key, mode, rng_low, rng_high, pre_chord_pitches=[]):
        # as root note
        if '(' in ch or '[' in ch:
            ch1, ch2 = ch[1:-1].split(', ')
            root1_pitches, chord1_pitches = self._chord_pitches(ch1, key, mode, rng_low, rng_high)
            root2_pitches, chord2_pitches = self._chord_pitches(ch2, key, mode, rng_low, rng_high)
        elif type(ch) in (list, tuple):
            ch1, ch2 = ch[0], ch[1] 
            root1_pitches, chord1_pitches = self._chord_pitches(ch1, key, mode, rng_low, rng_high)
            root2_pitches, chord2_pitches = self._chord_pitches(ch2, key, mode, rng_low, rng_high)
        else:
            root_pitches, chord_pitches = self._chord_pitches(ch, key, mode, rng_low, rng_high)
            root1_pitches = root2_pitches = root_pitches
            chord1_pitches = chord2_pitches = chord_pitches
 
        # statistic
        in_pitches = []
        for nt in self.notes:
            if nt.pitch not in in_pitches:
                in_pitches.append(nt.pitch)
            in_pitches.sort()

        # MAP pattern_pitch -> chord_pitch 
        np_in = len(in_pitches)
        np_tar1 = len(chord1_pitches)
        np_tar2 = len(chord2_pitches)
        
        # ensure number of tar pitch >= in pitch
        #print(" in_pitches, chord1_pitches =", in_pitches, chord1_pitches, "ch=", ch)
        assert np_in <= np_tar1, f"!!Error: np_in <= np_tar1 (but got {np_in} vs {np_tar1})"
        assert np_in <= np_tar2, f"!!Error: np_in <= np_tar1 (but got {np_in} vs {np_tar2})"

        # delete pitch from target, untill tar pitch == in pitch
        pre_chord1_pitches = [] if pre_chord_pitches==None else pre_chord_pitches
        avg_in = np.average(in_pitches)
        while np_tar1 > np_in:
            # if avg_in low, delete high; vise versa
            avg_tar = np.average(chord1_pitches)
            if avg_in <= avg_tar:
                # delete hifh pitch
                for k in range(np_tar1-1, -1, -1):
                    if chord1_pitches[k] not in pre_chord1_pitches\
                       or (chord1_pitches[k]-12 in chord1_pitches or chord1_pitches[k]+12 in chord1_pitches):
                        # keep occured (exclude±12)
                        chord1_pitches = chord1_pitches[:k] + chord1_pitches[k+1:]
                        break
            else:
                # delete low pitch
                for k in range(np_tar1):
                    if chord1_pitches[k] not in pre_chord1_pitches\
                       or (chord1_pitches[k]-12 in chord1_pitches or chord1_pitches[k]+12 in chord1_pitches):
                        # keep occured (exclude±12)
                        chord1_pitches = chord1_pitches[:k] + chord1_pitches[k+1:]
                        break
            if len(chord1_pitches) == np_tar1:
                # if failed to delete, randomly delete one
                k = random.choice(chord1_pitches)
                chord1_pitches.remove(k)
                #raise ValueError("Error: faile to delete (len(tar_right_tones) == tar_num_right_tone) ({})".format(tar_right_tones))
            # update
            np_tar1 = len(chord1_pitches)

        pre_chord2_pitches = chord1_pitches
        avg_in = np.average(in_pitches)
        while np_tar2 > np_in:
            # if avg_in low, delete high; vise versa 
            avg_tar = np.average(chord2_pitches)
            if avg_in <= avg_tar:
                # delete hifh pitch
                for k in range(np_tar2-1, -1, -1):
                    if chord2_pitches[k] not in pre_chord2_pitches\
                       or (chord2_pitches[k]-12 in chord2_pitches or chord2_pitches[k]+12 in chord2_pitches):
                        # keep occured (exclude±12)
                        chord2_pitches = chord2_pitches[:k] + chord2_pitches[k+1:]
                        break
            else:
                # delete low pitch
                for k in range(np_tar2):
                    if chord2_pitches[k] not in pre_chord2_pitches\
                       or (chord2_pitches[k]-12 in chord2_pitches or chord2_pitches[k]+12 in chord2_pitches):
                        # keep occured (exclude±12)
                        chord2_pitches = chord2_pitches[:k] + chord2_pitches[k+1:]
                        break
            if len(chord2_pitches) == np_tar2:
                # if failed to delete, randomly delete one
                k = random.choice(chord2_pitches)
                chord2_pitches.remove(k)
                #raise ValueError("Error: faile to delete (len(tar_right_tones) == tar_num_right_tone) ({})".format(tar_right_tones))
            # update
            np_tar2 = len(chord2_pitches)
  
        # generate notes
        acc_notes = []
        for nt in self.notes:
            tar_note = nt.copy() 
            st = tar_note.start
            dur = tar_note.dur

            # chord pitches
            if st < LEN_2NOTE:
                chord_pitches = chord1_pitches
            else:
                chord_pitches = chord2_pitches

            idx = in_pitches.index(nt.pitch)
            tar_note.pitch = chord_pitches[idx]

            # save note
            acc_notes.append(tar_note)

        # return
        return acc_notes, chord2_pitches

    def _chord_pitches(self, ch, key, mode, rng_low, rng_high):
        """ Return scale degrees for chord """

        scale = Scale(key, mode)

        root = key.value if type(key)==Key else key
        if mode in (Mode.Minor, Mode.Minor.value):
            root += 3 

        ch = Chord(ch)

        chord_pitches = [] 
        root_pitches = [] 
        #for p in scale.pitches:
        #    if rng_low <= p <= rng_high: 
        for p in range(rng_low, rng_high): 
            if (p - root)%12 in [x%12 for x in ch.degrees]:
                chord_pitches.append(p)
            if (p - root)%12 == ch.degrees[0]:
                root_pitches.append(p)
        # return
        return root_pitches, chord_pitches

class SectionPattern:
    def __init__(self, bar_whole_patterns, bar_half_patterns):
        self.bar_whole_patterns = bar_whole_patterns    # List[BarPattern]
        self.bar_half_patterns = bar_half_patterns    # List[BarPattern]
        self.rng_low = 0
        self.rng_high = 128

    def __len__(self):
        return len(self.bar_whole_patterns)

    def init_range(self, instr_group, instr_no):
        # range by instrument & pattern
        # range of pattern
        rng_low, rng_high = 128, 0
        for bar_pat in self.bar_whole_patterns + self.bar_half_patterns:
            for nt in bar_pat.notes:
                p = nt.pitch
                if p > rng_high:
                    rng_high = p
                if p < rng_low:
                    rng_low = p
        # range of instrument
        if instr_group == InstrGroupType.Drum:
            MIN_RNG = 0
        elif instr_group == InstrGroupType.Bass:
            MIN_RNG = 14
        else:
            MIN_RNG = 22
        if MIN_RNG > 0:
            instr_low, instr_high = self._range_of_instrument(instr_group, instr_no)
            instr_mid = instr_low + int((instr_high - instr_low)/2)
            while rng_high - rng_low < MIN_RNG:
                # range by instrument
                rng_mid = rng_low + int((rng_high - rng_low)/2)
                if rng_mid < instr_mid:
                    rng_high += 1 
                elif rng_mid > instr_mid:
                    rng_low -= 1
                else:
                    rng_low -= 1
        # return
        self.rng_low = rng_low
        self.rng_high = rng_high

    # get pitch range of instrument
    def _range_of_instrument(self, instr_group, instr_no):
        # instrumental register 
        if str(instr_no+1) in INSTRUMENT_INFO:
            info = INSTRUMENT_INFO[str(instr_no+1)]
            rng_low, rng_high = info.rng
            assert rng_high > 0, f"Error: unknown range={info.rng} for instr-no={instr_no}" 
        else:
            raise NotImplementedError(f"Error: unknown range for instr-group={instr_group}, no={instr_no}")
            
        # return
        return rng_low, rng_high

@dataclass
class TrackPattern:
    def __init__(self, instr_group, instr_no, 
                       Intro, A, C, B, Outro):
        self.instr_group = instr_group
        self.instr_no = instr_no

        def _init_section_pattern(sec_bars):
            bar_whole_pats = []
            bar_half_pats = []
            if type(sec_bars) == list:
                for bar_notes in sec_bars:
                    events = []
                    for nt in bar_notes:
                        nt_event = NoteEvent(start=nt[0], dur=nt[1], pitch=nt[2], vel=nt[3])
                        # save note 
                        events.append(nt_event)
                    bar_whole_pats.append(BarPattern(events))
                bar_half_pats = bar_whole_pats
            else:
                for bar_notes in sec_bars["whole"]:
                    events = []
                    for nt in bar_notes:
                        nt_event = NoteEvent(start=nt[0], dur=nt[1], pitch=nt[2], vel=nt[3])
                        # save note 
                        events.append(nt_event)
                    bar_whole_pats.append(BarPattern(events))
                for bar_notes in sec_bars["half"]:
                    events = []
                    for nt in bar_notes:
                        nt_event = NoteEvent(start=nt[0], dur=nt[1], pitch=nt[2], vel=nt[3])
                        # save note 
                        events.append(nt_event)
                    bar_half_pats.append(BarPattern(events))

            # new SectionPattern
            sec = SectionPattern(bar_whole_pats, bar_half_pats)
            sec.init_range(instr_group, instr_no)
            return sec

        self.Intro = _init_section_pattern(Intro)   # SectionPattern
        self.A = _init_section_pattern(A)           # SectionPattern
        self.C = _init_section_pattern(C)           # SectionPattern
        self.B = _init_section_pattern(B)           # SectionPattern
        self.Outro = _init_section_pattern(Outro)   # SectionPattern 


class AccompStyle:
    def __init__(self, style_id, tracks):
        self.style_id = style_id
        self.tracks = tracks

    def gen(self, bpm, key, mode, structure, _log=print):
        
        # generate each track
        track_dicts = []
        for i, trk in enumerate(self.tracks):
            instr_group = trk.instr_group
            instr_no = trk.instr_no
            #_log(f"         > No {i+1}th track: instr_group={instr_group.name}")

            track_dicts.append({"instr_group": instr_group,
                               "instr_no": instr_no,
                                "sec_notes": [],
                               })
            # loop each section
            for sec_dict in structure:
                # read section info
                sec_idx = sec_dict["section_index"]
                st_bar = sec_dict["bar_index"]
                sec_type = sec_dict["type"]
                sec_nbar = sec_dict["num_bars"]
                sec_chords = sec_dict["chords"]

                # pattern for generating
                if sec_type == "INTRO":
                    trk_pat = trk.Intro 
                elif sec_type.startswith("VERSE"):
                    trk_pat = trk.A
                elif sec_type.startswith("PRECHORUS"):
                    trk_pat = trk.C
                elif sec_type.startswith("CHORUS"):
                    trk_pat = trk.B
                elif sec_type == "OUTRO":
                    trk_pat = trk.Outro 
                else:
                    raise NotImplementedError(f"Error: unknown sec_type={sec_type}")
                npat = len(trk_pat)
                rng_low = trk_pat.rng_low
                rng_high = trk_pat.rng_high

                #_log(f"             > {sec_type}: nbar={sec_nbar}, range={rng_low}~{rng_high}")

                # generate each bar 
                cache_pitch = None
                sec_notes = []
                for j in range(sec_nbar):
                    # chord of bar (1 or 2)
                    ch = sec_chords[j]

                    # pattern by number of chord 
                    if '(' in ch or '[' in ch:
                        bar_pat = trk_pat.bar_half_patterns[j%npat]
                    elif type(ch) in (list, tuple):
                        bar_pat = trk_pat.bar_half_patterns[j%npat]
                    else:
                        bar_pat = trk_pat.bar_whole_patterns[j%npat] 
                    # gen
                    bar_notes, cache_pitch = bar_pat.gen_bar(ch, key, mode, instr_group, rng_low, rng_high, cache_pitch)
                    
                    # save 
                    sec_notes.append(bar_notes)

                # save notes 
                track_dicts[-1]["sec_notes"].extend(sec_notes)
        # return 
        return track_dicts


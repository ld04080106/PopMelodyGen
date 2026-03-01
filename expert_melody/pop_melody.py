#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pop_melody.py

Melody pitch generation under tonal and chord constraints.

Responsibilities:
  - Assign pitches to rhythmic onsets conditioned on key/mode and bar-wise chords.
  - Enforce vocal register, step/leap ratios, and cadence rules.
  - Output a note sequence suitable for MIDI writing and tokenization.

Typical functions:
  - generate_melody(cfg, settings, structure, chords, rhythm, rng) -> list[Note]
"""

from expert_melody.pop_settings import Tempo, SectionType, Key, Mode
from expert_melody.pop_settings import TEMPO_RANGES, IS_DEBUG, TRIM_LONG_NOTE
from expert_melody.pop_settings import VERSE_PITCH_TRANS, PRECH_PITCH_TRANS, CHORUS_PITCH_TRANS
from expert_melody.pop_settings import VERSE_LOW_OFFSET, VERSE_HIGH_OFFSET, CHORUS_LOW_OFFSET, CHORUS_HIGH_OFFSET
from expert_melody.pop_settings import PRECH_LOW_OFFSET, PRECH_HIGH_OFFSET

from expert_melody.pop_chordprog import I, ii, iii, IV, V, vi, VII, vii
from expert_melody.pop_chordprog import Chord

from dataclasses import dataclass
from typing import List
import random
 
# =========================
# NOTE 
# ========================= 
@dataclass
class NoteEvent:
    pitch: int
    bar_idx: int
    start_beat: float
    dur_beat: float
    velocity: int = 124

    def end_beat(self):
        return self.start_beat + self.dur_beat
    def __repr__(self):
        return f"[{self.pitch}, {self.start_beat}+{self.dur_beat}, v={self.velocity}]"

@dataclass
class Melody:
    notes: List[NoteEvent]
    beats_per_bar = 4

    def add(self, note):
        self.notes.append(note)

    @property
    def length_beats(self):
        if not self.notes or len(self.notes) == 0:
            return 0.0

        final_nt = self.notes[-1]
        end = final_nt.bar_idx * 4 + final_nt.end_beat()
        return end

    def __len__(self):
        return len(self.notes)
    def __repr__(self):
        return f"{self.notes}"

 
# Scale
class Scale:
    SEMITONES_MAJOR = [0, 2, 4, 5, 7, 9, 11]
    SEMITONES_MINOR = [0, 2, 3, 5, 7, 8, 10]
 
    def __init__(self, key, mode):
        semitones = self.SEMITONES_MAJOR if mode in (Mode.Major, Mode.Major.value) else self.SEMITONES_MINOR
        key_value = key.value if type(key) == Key else key

        self.pitches = []
        for i in range(-12, 128, 12):
            for j in semitones:
                mn = i + key_value + j
                if 0 <= mn < 128 and mn not in self.pitches:
                    self.pitches.append(mn)
 

class MelodyGenerator:
    """
    For a given section (structure, chord_prog, rhythm, tempo_class, key/mode),
    - Sample 4 phrase syllable counts
    - Split each phrase into upbeat/full/unfull/ending portions (bars & syllables)
    - Choose bar-level rhythm patterns for each bar.
    """

    def __init__(self, structure, chord_prog_seq, rhythm_seq, 
                       tempo_class, key, mode, vocal_range, 
                       quant_unit, seed, _log=print):
        self.structure = structure
        self.chord_prog_seq = chord_prog_seq
        self.rhythm_seq = rhythm_seq
        self.tempo_class = tempo_class
        self.key = key
        self.mode = mode
        #self.scale = Scale(key, mode)
        self.pitch_low, self.pitch_high = vocal_range
        self.quant_unit = quant_unit
        self.rand = random.Random(seed)
        self._log = _log

    def _chord_pitches(self, ch, pitch_low, pitch_high, only_root=False, is_final_strong=False):
        """ Return degrees for chord """
        root = self.key.value 
        if self.mode == Mode.Minor:
            root += 3

        # chord tones
        degrees = ch.degrees
        if only_root:
            degrees = degrees[:1]

        # remove 4/7
        if is_final_strong:
            if len(degrees) > 1 and 5 in degrees:
                degrees.remove(5)
            if len(degrees) > 1 and 11 in degrees:
                degrees.remove(11)

        scales = [(root+x)%12 for x in degrees]
        #print(" ch =", ch, "degrees =", degrees, "is_final_strong =", is_final_strong)

        # loop pitch 0~127
        chord_pitches = []
        for p in range(128):
            if p%12 in scales and pitch_low <= p <= pitch_high:
                chord_pitches.append(p)
        # return
        return chord_pitches

    def _scale_pitches(self, ch, pitch_low, pitch_high):
        """ Return scales for chord """
        root = self.key.value 
        if self.mode == Mode.Minor:
            root += 3

        scales_pitches = [] 
        for p in range(128):
            if p%12 in [(root+x)%12 for x in ch.scales] and pitch_low <= p <= pitch_high:
                scales_pitches.append(p)
        # return
        return scales_pitches

    def _sample_pitch(self, prev_pitch, candi_pitches, p_trans=VERSE_PITCH_TRANS, ref_pitch=None):
        # sample pitch
        if prev_pitch == 0:
            prev_pitch = self.pitch_low + int((self.pitch_high - self.pitch_low)/2)
 
        # P(pitch)
        p_pitch = [0 for x in range(len(candi_pitches))]
        for k in p_trans.keys():
            if prev_pitch+k in candi_pitches:
                idx = candi_pitches.index(prev_pitch+k)
                p_pitch[idx] = p_trans[k]

        # use reference first
        if ref_pitch != None and ref_pitch > 0:
            if ref_pitch in candi_pitches:
                # ref in candidate
                sel_pitch = ref_pitch
            else:
                # find nearest
                self.rand.shuffle(candi_pitches)
                min_diff = 9999
                nearest_pitch = candi_pitches[0]
                for x in candi_pitches:
                    diff = abs(x - ref_pitch)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_pitch = x
                sel_pitch = nearest_pitch
        else:
            if len(p_pitch) == 1:
                p_pitch[0] = 1.0
            sel_pitch = self.rand.choices(candi_pitches, weights=p_pitch)[0]
            #self._log(" > DEBUG: rand pitch is " + str(sel_pitch) + " from " + str(candi_pitches) + " w=" + str(p_pitch))

        return sel_pitch

    def _sample_key_root_pitch(self):
        # key root pitch
        root = self.key.value
        for p in range(self.pitch_low, self.pitch_high):
            if p%12 == root:
                return p 
        raise NotImplementedError(f"Error: not found root={root}")

    def _sample_chord_root_pitch(self, chord_root):
        # chord root pitch
        root = self.key.value 
        if self.mode == Mode.Minor:
            root += 3
        chord_root = (chord_root + root)%12

        for p in range(self.pitch_low, self.pitch_high):
            if p%12 == chord_root:
                return p 
        raise NotImplementedError(f"Error: not found root={chord_root}")

    def _sample_noscale_pitch(self, prev_pitch, 
                                    pitch_low, pitch_high, p_trans=VERSE_PITCH_TRANS):
        # sample pitch
        if prev_pitch == 0:
            prev_pitch = pitch_low + int((pitch_high - pitch_low)/2)

        # pitch in register
        allow_pitch = []
        for p in range(128):
            if pitch_low <= p <= pitch_high:
                allow_pitch.append(p)

        # P(pitch)
        p_pitch = [0 for x in range(len(allow_pitch))]
        for k in p_trans.keys():
            if prev_pitch+k in allow_pitch:
                idx = allow_pitch.index(prev_pitch+k)
                p_pitch[idx] = p_trans[k]

        sel_pitch = self.rand.choices(allow_pitch, weights=p_pitch)[0]

        return sel_pitch

    def _sample_scale_pitch(self, prev_pitch, 
                            pitch_low, pitch_high, p_trans=VERSE_PITCH_TRANS):
        # sample pitch
        if prev_pitch == 0:
            prev_pitch = pitch_low + int((pitch_high - pitch_low)/2)

        # pitch in scale
        candi_pitches = self.scale.pitches

        allow_pitch = []
        for p in candi_pitches:
            if pitch_low <= p <= pitch_high:
                allow_pitch.append(p)

        # P(pitch)
        p_pitch = [0 for x in range(len(allow_pitch))]
        for k in p_trans.keys():
            if prev_pitch+k in allow_pitch:
                idx = allow_pitch.index(prev_pitch+k)
                p_pitch[idx] = p_trans[k]

        sel_pitch = self.rand.choices(allow_pitch, weights=p_pitch)[0]

        return sel_pitch

    def gen(self, force_key_root=False,     # for DEBUG
                  force_chord_root=False,   # for DEBUG
                  force_rand_noscale_pitch=False,   # for DEBUG
                  force_rand_scale_pitch=False,   # for DEBUG
                  limit_pitch_trans=False,  # for DEBUG
                  ): 
        # section info
        melody_seq = []    # for return
        section_melody_plans = {}   # for temp use

        # ---------------- Loop each section ----------------
        # Only deal with vocal part(verse/pre-chorus/chorus)
        # ending chords of previous section (default is Key+Mode)
        if self.mode == Mode.Major:
            pre_bar_chord = I
        elif self.mode == Mode.Minor:
            pre_bar_chord = vi
        else:
            raise NotImplementedError(f"Error: unknown mode = {self.mode}")

        # global
        global_bar_idx = 0 
        section_metas = [] 
 
        # loop each section
        nsec = len(self.structure)
        for i, sec in enumerate(self.structure):
            # section info
            stype = sec.section_type
            num_bar = sec.num_bar

            next_stype = self.structure[i+1].section_type if i+1 < nsec else None

            chords = self.chord_prog_seq[i]
            nch = len(chords)

            # check section type
            if stype in (SectionType.VERSE, SectionType.PRECHORUS, SectionType.CHORUS):
                # VERSE / PRECHORUS / CHORUS
                if IS_DEBUG:
                    self._log(f"    > Melody: {i}th section - {stype.name} ------------------")
                r1 = self.rand.random()
                r2 = self.rand.random()
                r3 = self.rand.random()
                r4 = self.rand.random()

                # rhythm info
                phrase_nchars = self.rhythm_seq[i]["phrase_nchars"]
                rhythm_bars = self.rhythm_seq[i]["bar_patterns"]

                # ref
                ref_melody_plan = None
                if stype in section_melody_plans:
                    # has ref
                    ref_melody_plan = section_melody_plans[stype]
                else:
                    # no ref
                    is_chord_repeat12 = False 
                    is_chord_repeat13 = False 
                    is_chord_repeat24 = False 
                    if num_bar == 8:
                        is_chord_repeat12 = chords[0]==chords[2] and r2 > 0.9
                        is_chord_repeat13 = chords[0]==chords[4] and r2 > 0.9
                        is_chord_repeat24 = chords[2]==chords[6] and r2 > 0.9 
                    elif num_bar == 16:
                        is_chord_repeat12 = chords[0]==chords[4] and chords[1]==chords[5] and r2 > 0.9
                        is_chord_repeat13 = chords[0]==chords[8] and chords[1]==chords[9] and r2 > 0.9
                        is_chord_repeat24 = chords[4]==chords[12] and chords[5]==chords[13] and r2 > 0.9
                #print("is_chord_repeat12 =", is_chord_repeat12)
                #print("is_chord_repeat13 =", is_chord_repeat13)
                #print("is_chord_repeat24 =", is_chord_repeat24)

                # vocal register / P_trans
                if stype == SectionType.VERSE:
                    pitch_low = self.pitch_low + VERSE_LOW_OFFSET
                    pitch_high = self.pitch_high + VERSE_HIGH_OFFSET
                    p_trans = VERSE_PITCH_TRANS
                    st_pitch = pitch_low + int((pitch_high-pitch_low)/2)
                elif stype == SectionType.PRECHORUS:
                    pitch_low = self.pitch_low + PRECH_LOW_OFFSET
                    pitch_high = self.pitch_high + PRECH_HIGH_OFFSET
                    p_trans = PRECH_PITCH_TRANS
                    st_pitch = pitch_low + int((pitch_high-pitch_low)/2)
                elif stype == SectionType.CHORUS:
                    pitch_low = self.pitch_low + CHORUS_LOW_OFFSET
                    pitch_high = self.pitch_high + CHORUS_HIGH_OFFSET
                    p_trans = CHORUS_PITCH_TRANS
                    st_pitch = pitch_low + int((pitch_high-pitch_low)/2)
                else:
                    raise NotImplementedError("Error: unknown sec_type={stype}")

                # 1. 1st loop: identify each note's attribute 
                if IS_DEBUG:
                    self._log(f"    > ======== Step 1. Extracting notes ======== ")
 
                note_dicts = []
                for j, bar_ch in enumerate(chords): 
                    if IS_DEBUG:
                        self._log(f"        > No {j+1} bar: chords = {bar_ch}")

                    # current rhythm
                    pats = rhythm_bars[j]

                    # deal with rhythm patterns
                    for pat in pats:
                        pat_type = pat.pattern_type
                        if pat_type == "upbeat":
                            # pickup: use previous bar's chord
                            cur_bar_idx = j - 1
                            if type(pre_bar_chord) == Chord:
                                cur_ch1 = cur_ch2 = pre_bar_chord 
                            elif type(pre_bar_chord) in (list, tuple):
                                cur_ch1 = pre_bar_chord[0]
                                cur_ch2 = pre_bar_chord[1] if len(pre_bar_chord) > 0 else cur_ch1 
                            else:
                                raise NotImplementedError(f"Error: unknown type(pre_bar_chord) == {type(pre_bar_chord)}")
                        else:
                            # full/unfull/ending: use current bar's chord
                            cur_bar_idx = j
                            if type(bar_ch) == Chord:
                                cur_ch1 = cur_ch2 = bar_ch 
                            elif type(bar_ch) in (list, tuple):
                                cur_ch1 = bar_ch[0]
                                cur_ch2 = bar_ch[1] if len(bar_ch) > 0 else cur_ch1
                            else:
                                raise NotImplementedError(f"Error: unknown type(bar_ch) == {type(bar_ch)}")

                        if IS_DEBUG:
                            self._log(f"          > Rhythm type: {pat.pattern_type}") 

                        # identifying strong/weak positions
                        nnote = len(pat)
                        k = 0
                        in_bar_pos = 0 
                        while k < nnote:
                            # note info
                            dur = pat.durations[k]
                            is_rest = pat.is_rests[k]
                            
                            # if is rest
                            if is_rest: 
                                # update
                                in_bar_pos += dur 
                                k += 1
                                continue

                            # if is strong
                            is_strong = False 
                            is_final_strong = False
                            append_ch = None
                            if in_bar_pos == 0:
                                is_strong = True
                                ch = cur_ch1
                            elif in_bar_pos == 2.0:
                                is_strong = True
                                ch = cur_ch2 
                            elif in_bar_pos == 1.5 and dur >= 1.0:
                                is_strong = True
                                ch = cur_ch2 
                            elif in_bar_pos == 0.5 and dur >= 1.0 and \
                                 (k>0 and pat.is_rests[k-1]):
                                is_strong = True
                                ch = cur_ch1 
                            elif in_bar_pos == 1.0 and dur >= 1.0 and \
                                 (k>0 and pat.is_rests[k-1]):
                                is_strong = True
                                ch = cur_ch2 
                            elif in_bar_pos == 1.0 and dur >= 1.0 and pat.is_rests[0]:
                                is_strong = True
                                ch = cur_ch1 
                            # ending's final note is strong
                            if pat_type == "ending" and \
                               (k==nnote-1 or pat.is_rests[k+1]):
                                is_strong = True
                                is_final_strong = True   # no fa/xi
                                ch = cur_ch2
                                #print(" is_final_strong -> k=", k, "nnote =", nnote)
                            # one note's dur > 2
                            if cur_ch1 != cur_ch2:
                                if pat_type == "ending" and in_bar_pos <= 1.0 and \
                                   (k==nnote-1 or pat.is_rests[k+1]) \
                                   or in_bar_pos==1.0 and dur>= 2.0:
                                    ch = cur_ch2
                                    append_ch = cur_ch1

                            # if is weak(appoggiaturas)
                            if is_strong:
                                next_dur = pat.durations[k+1] if k+1 < nnote and not pat.is_rests[k+1] else 0
                                next_next_dur = pat.durations[k+2] if k+2 < nnote and not pat.is_rests[k+2] else 0
                                if in_bar_pos==0 and dur==0.5 and next_dur==0.5 and next_next_dur > 0.5 \
                                   or in_bar_pos==2 and dur==0.5 and next_dur==0.5 and next_next_dur > 0.5:
                                    # 2 appoggiaturas
                                    note_dicts.append({"pitch": -1, 
                                                       "dur": dur, "ch": ch, "append_ch":None,
                                                       "bar_idx": cur_bar_idx, "st": in_bar_pos,
                                                       "brest": False, "pat_type": pat_type,
                                                       "bstrong": False, "bstrong_final": False,
                                                       "bweak": False, 
                                                       "bappog": True, 
                                                       })
                                    note_dicts.append({"pitch": -1, 
                                                       "dur": next_dur, "ch": ch, "append_ch":None,
                                                       "bar_idx": cur_bar_idx, "st": in_bar_pos+dur,
                                                       "brest": False, "pat_type": pat_type,
                                                       "bstrong": False, "bstrong_final": False,
                                                       "bweak": False,
                                                       "bappog": True, 
                                                       })
                                    # ending's final note is strong
                                    if pat_type == "ending" and (k+2==nnote-1 or pat.is_rests[k+3]): 
                                        is_final_strong = True
                                    # one note's dur > 2
                                    if cur_ch1 != cur_ch2:
                                        if pat_type == "ending" and in_bar_pos+dur+next_dur <= 1.0 and \
                                           (k+2==nnote-1 or pat.is_rests[k+3]) \
                                           or in_bar_pos+dur+next_dur==1.0 and next_next_dur>= 2.0:
                                            ch = cur_ch2
                                            append_ch = cur_ch1
                                    note_dicts.append({"pitch": -1, 
                                                       "dur": next_next_dur, "ch": ch, "append_ch":append_ch,
                                                       "bar_idx": cur_bar_idx, "st": in_bar_pos+dur+next_dur,
                                                       "brest": False, "pat_type": pat_type,
                                                       "bstrong": True, "bstrong_final": is_final_strong,
                                                       "bweak": False,
                                                       "bappog": False, 
                                                       })
                                    # update
                                    in_bar_pos += (dur+next_dur+next_next_dur)
                                    k += 3
                                    continue
                                elif in_bar_pos==0 and dur==0.5 and next_dur>0.5 \
                                     or in_bar_pos==2 and dur==0.5 and next_dur > 0.5 \
                                     or in_bar_pos==0 and dur==1 and next_dur > 1 \
                                     or in_bar_pos==0 and dur==1 and next_dur >= 1 and k+2==nnote:
                                    # 1 appoggiatura
                                    note_dicts.append({"pitch": -1, 
                                                       "dur": dur, "ch": ch, "append_ch":None,
                                                       "bar_idx": cur_bar_idx, "st": in_bar_pos,
                                                       "brest": False, "pat_type": pat_type,
                                                       "bstrong": False, "bstrong_final": False,
                                                       "bweak": False,
                                                       "bappog": True, 
                                                       })
                                    if pat_type == "ending" and (k+1==nnote-1 or pat.is_rests[k+2]): 
                                        is_final_strong = True  
                                        #print(" is_final_strong -> k+1=", k+1, "nnote =", nnote)
                                    # one note's dur > 2
                                    if cur_ch1 != cur_ch2:
                                        if pat_type == "ending" and in_bar_pos+dur <= 1.0 and \
                                           (k+1==nnote-1 or pat.is_rests[k+2]) \
                                           or in_bar_pos+dur==1.0 and next_dur>= 2.0:
                                            ch = cur_ch2
                                            append_ch = cur_ch1
                                    note_dicts.append({"pitch": -1, 
                                                       "dur": next_dur, "ch": ch, "append_ch":append_ch,
                                                       "bar_idx": cur_bar_idx, "st": in_bar_pos+dur,
                                                       "brest": False, "pat_type": pat_type,
                                                       "bstrong": True, "bstrong_final": is_final_strong,
                                                       "bweak": False,
                                                       "bappog": False, 
                                                       })
                                    in_bar_pos += (dur+next_dur)
                                    k += 2
                                    continue

                            # save strong
                            if is_strong:
                                # save
                                note_dicts.append({"pitch": -1, 
                                                   "dur": dur, "ch": ch, "append_ch":append_ch,
                                                   "bar_idx": cur_bar_idx, "st": in_bar_pos,
                                                   "brest": False, "pat_type": pat_type,
                                                   "bstrong": True, "bstrong_final": is_final_strong,
                                                   "bweak": False,
                                                   "bappog": False, 
                                                   })
                                # update
                                in_bar_pos += dur 
                                k += 1
                                continue

                            # save weak
                            if in_bar_pos <= 2.0:
                                ch = cur_ch1
                            else:
                                ch = cur_ch2
                            note_dicts.append({"pitch": -1, 
                                               "dur": dur, "ch": ch, "append_ch":None,
                                               "bar_idx": cur_bar_idx, "st": in_bar_pos,
                                               "brest": False, "pat_type": pat_type,
                                               "bstrong": False, "bstrong_final": False,
                                               "bweak": True,
                                               "bappog": False, 
                                               })
                            # update
                            in_bar_pos += dur 
                            k += 1
                 
                    #update
                    pre_bar_chord = bar_ch
                if IS_DEBUG:
                    self._log(f"          > note_dicts = {len(note_dicts)}") 
                    #for x in note_dicts:
                    #    self._log(f"          > {x}") 
 
                # 2. 2nd loop: sample pitches for strong notes
                if IS_DEBUG:
                    self._log(f"    > ======== Step 2. Determine pitch (Strong) ======== ")

                def _find_ref_pitch(ref_note_dicts, bar_idx, st):
                    # find pitch at specified position(bar-beat) in ref
                    for x in ref_note_dicts:
                        if x["bar_idx"] == bar_idx and x["st"] <= st < x["st"]+x["dur"]:
                            #print(" Found ref pitch =", x["pitch"]) 
                            return x["pitch"]
                    #print(" Found no pitch :", bar_idx, st) 
                    return None
                def _ref_or_not(tar_nt, ref_melody_plan, note_dicts):
                    # find ref 
                    bar_idx = tar_nt["bar_idx"]
                    st = tar_nt["st"]
                    pat_type = tar_nt["pat_type"]

                    ref_pitch = None
                    if ref_melody_plan != None:
                        # use ref first
                        ref_pitch = _find_ref_pitch(ref_melody_plan, bar_idx, st) 
                    else:
                        if is_chord_repeat12:
                            # phrase2 repeat phrase1
                            if num_bar==8:
                                if bar_idx==1 and pat_type=="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, -1, st)
                                elif bar_idx==2 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 0, st)
                            elif num_bar==16:
                                if bar_idx==1 and pat_type=="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, -1, st)
                                elif bar_idx==2 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 0, st)
                                elif bar_idx==3 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 1, st)
                        if is_chord_repeat13:
                            # phrase3 repeat phrase1
                            if num_bar==8:
                                if bar_idx==3 and pat_type=="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, -1, st)
                                elif bar_idx==4 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 0, st)
                            elif num_bar==16:
                                if bar_idx==7 and pat_type=="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, -1, st)
                                elif bar_idx==8 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 0, st)
                                elif bar_idx==9 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 1, st)
                        if is_chord_repeat24:
                            # phrase4 repeat phrase2
                            if num_bar==8:
                                if bar_idx==5 and pat_type=="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 1, st)
                                elif bar_idx==6 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 2, st)
                            elif num_bar==16:
                                if bar_idx==11 and pat_type=="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 3, st)
                                elif bar_idx==12 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 4, st)
                                elif bar_idx==13 and pat_type!="upbeat":
                                    ref_pitch = _find_ref_pitch(note_dicts, 5, st)
                    return ref_pitch

                prev_pitch = st_pitch
                nnote = len(note_dicts)
                for j, nt in enumerate(note_dicts):
                    # Deal with strong note
                    if nt["bstrong"]:
                        # chord
                        ch = nt["ch"]
                        bar_idx = nt["bar_idx"]
                        st = nt["st"]
                        dur = nt["dur"]
                        pat_type = nt["pat_type"]
                        is_final_strong = nt["bstrong_final"]
                        append_ch = nt["append_ch"]

                        if IS_DEBUG:
                            self._log(f"        > No {j+1} strong note: ch = {ch}, is_final_strong={is_final_strong}, append_ch={append_ch}, st={st}, dur={dur}.")
                        
                        # use ref first
                        ref_pitch = _ref_or_not(nt, ref_melody_plan, note_dicts)

                        # chord tones
                        chord_pitches = []
                        # deal with final note
                        #if stype==SectionType.CHORUS and stype != next_stype and j+1 == nnote:
                        if stype==SectionType.CHORUS and j+1 == nnote:
                            # Final note of chorus 
                            chord_pitches = self._chord_pitches(ch, pitch_low, pitch_high, only_root=True, is_final_strong=is_final_strong)
                        else:
                            # chord tones
                            if append_ch != None:
                                # one note 2 chords
                                candi_pitches1 = self._chord_pitches(ch, pitch_low, pitch_high, is_final_strong=is_final_strong)
                                candi_pitches2 = self._chord_pitches(append_ch, pitch_low, pitch_high, is_final_strong=is_final_strong)
                                chord_pitches = list(set(candi_pitches1) & set(candi_pitches2))
                            if len(chord_pitches) == 0:
                                # ignore append chord
                                chord_pitches = self._chord_pitches(ch, pitch_low, pitch_high, is_final_strong=is_final_strong)
                        
                        if IS_DEBUG:
                            self._log(f"        >      chord_pitches = {chord_pitches}.")
                            if ref_pitch:
                                self._log(f"        >      ref_pitch = {ref_pitch}.")
                        
                        # sample by P_trans
                        chosen_pitch = self._sample_pitch(prev_pitch, chord_pitches, p_trans, ref_pitch)

                        if IS_DEBUG:
                            self._log(f"        >      choose pitch = {chosen_pitch}.")
                         
                        # save pitch
                        note_dicts[j]["pitch"] = chosen_pitch

                        # update
                        prev_pitch = chosen_pitch

                # 3. 3rd loop: sample pitches for weak notes
                if IS_DEBUG:
                    self._log(f"    > ======== Step 3. Determine pitch (Appgo & weak) ======== ")
                prev_pitch = -1
                nnote = len(note_dicts)
                j = 0
                while j < nnote:
                    # weak
                    nt = note_dicts[j]
                    next_nt = note_dicts[j+1] if j+1 < nnote else None
                    next_next_nt = note_dicts[j+2] if j+2 < nnote else None
                    
                    ch = nt["ch"]
                    bar_idx = nt["bar_idx"]
                    st = nt["st"]

                    ref_pitch = _ref_or_not(nt, ref_melody_plan, note_dicts)

                    # use ref
                    if nt["bappog"]:
                        # appoggiatura
                        # number of appoggiatura
                        if next_nt and next_nt["bappog"] and next_next_nt and next_next_nt["bappog"]:
                            raise NotImplementedError("Error: fail to deal with 3 continuous appogs.")
                        elif next_nt and next_nt["bappog"]:
                            # 2 appoggiatura notes
                            next_strong_pitch = next_next_nt["pitch"]
                            # ref
                            ref_pitches = None
                            if ref_pitch != None:
                                # use ref
                                ref_pitch2 = _ref_or_not(next_nt, ref_melody_plan, note_dicts)
                                ref_pitches = [ref_pitch, ref_pitch2]
                                if IS_DEBUG:
                                    self._log(f"        >      ref_pitches = {ref_pitches}.")
                            max_interval = 5
                            #print(f"   > [{j+1}] Appog → 2个, prev_p=", prev_pitch, "next_strong_pitch=", next_strong_pitch, "ch=", ch)
                            # gen
                            #print(" prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches, max_interval =", prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches, max_interval)pitches = self._gen_2_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches, max_interval)
                            pitches = self._gen_2_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches, max_interval)
                            if prev_pitch < 0:
                                # prevent appog pitch out of range
                                if pitches[0] < pitch_low or pitches[0] > pitch_high or pitches[1] < pitch_low or pitches[1] > pitch_high:
                                    pitches = self._gen_2_weak(128, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches, max_interval)
                            # save notes
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1] 
                            #print(f"   > Appog → {pitches[0]}   in range {pitch_low}~{pitch_high}")
                            j += 2
                            continue
                        else:
                            # 1 appoggiatura note
                            next_strong_pitch = next_nt["pitch"]
                            if ref_pitch != None:
                                if IS_DEBUG:
                                    self._log(f"        >      ref_pitch = {ref_pitch}.")
                            max_interval = 3
                            #print(f"   > [{j+1}] Appog → 1个, prev_p=", prev_pitch, "next_strong_pitch=", next_strong_pitch, "ch=", ch)
                            # gen
                            pitch = self._gen_1_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitch, max_interval)
                            if prev_pitch < 0:
                                if pitch < pitch_low or pitch > pitch_high:
                                    pitch = self._gen_1_weak(128, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitch, max_interval)
                            # save notes
                            note_dicts[j]["pitch"] = pitch
                            #print(f"   > Appog → {pitch}  in range {pitch_low}~{pitch_high}")
                            j += 1
                            continue
                    elif nt["bweak"]:
                        # Weak
                        # number of weak notes
                        nweak = 1
                        widx = j+1
                        next_strong_pitch = prev_pitch
                        ref_pitches = [ref_pitch, ] if ref_pitch != None else None
                        while widx < len(note_dicts):
                            if note_dicts[widx]["bweak"]:
                                nweak += 1
                                if ref_pitch != None:
                                    p = _ref_or_not(note_dicts[widx], ref_melody_plan, note_dicts)
                                    if p is None:
                                        ref_pitches == None
                                        break
                                    else:
                                        ref_pitches.append(p)
                            elif note_dicts[widx]["bappog"] or note_dicts[widx]["brest"]:
                                pass
                            else:
                                next_strong_pitch = note_dicts[widx]["pitch"]
                                break
                            # update
                            widx += 1 
                        if ref_melody_plan != None:
                            if IS_DEBUG:
                                self._log(f"        >      ref_pitches = {ref_pitches}, nweak = {nweak}.")
                        if nweak == 1:
                            # 1 weak note 
                            #print(f"   > [{j+1}] weak → 1, prev_p=", prev_pitch, "next_strong_pitch=", next_strong_pitch, "ch=", ch)
                            pitch = self._gen_1_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitch)
                            if prev_pitch < 0:
                                if pitch < pitch_low or pitch > pitch_high:
                                    pitch = self._gen_1_weak(128, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitch)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitch
                            #print(f"   > Result → {pitch}")
                            j += 1
                            continue
                        elif nweak == 2:
                            # 2 weak notes
                            #print(f"   > [{j+1}] weak → 2, prev_p=", prev_pitch, "next_strong_pitch=", next_strong_pitch, "ch=", ch) 
                            pitches = self._gen_2_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            if prev_pitch < 0:
                                # prevent appog pitch out of range
                                if pitches[0] < pitch_low or pitches[0] > pitch_high or pitches[1] < pitch_low or pitches[1] > pitch_high:
                                    pitches = self._gen_2_weak(128, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}")
                            j += 2
                            continue
                        elif nweak == 3:
                            # 3 weak notes 
                            #print(f"   > [{j+1}] Weak → 3") 
                            pitches = self._gen_3_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            note_dicts[j+2]["pitch"] = pitches[2]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}, {pitches[2]}")
                            j += 3 
                        elif nweak == 4:
                            # 4 weak notes 
                            #print(f"   > [{j+1}] Weak → 4") 
                            pitches = self._gen_4_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            note_dicts[j+2]["pitch"] = pitches[2]
                            note_dicts[j+3]["pitch"] = pitches[3]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}, {pitches[2]}, {pitches[3]}")
                            j += 4 
                        elif nweak == 5:
                            # 5 weak notes 
                            #print(f"   > [{j+1}] Weak → 5") 
                            pitches = self._gen_5_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            note_dicts[j+2]["pitch"] = pitches[2]
                            note_dicts[j+3]["pitch"] = pitches[3]
                            note_dicts[j+4]["pitch"] = pitches[4]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}, {pitches[2]}, {pitches[3]}, {pitches[4]}")
                            j += 5 
                        elif nweak == 6:
                            # 6 weak notes 
                            #print(f"   > [{j+1}] Weak → 6") 
                            pitches = self._gen_6_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            note_dicts[j+2]["pitch"] = pitches[2]
                            note_dicts[j+3]["pitch"] = pitches[3]
                            note_dicts[j+4]["pitch"] = pitches[4]
                            note_dicts[j+5]["pitch"] = pitches[5]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}, {pitches[2]}, {pitches[3]}, {pitches[4]}, {pitches[5]}")
                            j += 6
                        elif nweak == 7:
                            # 7 weak notes 
                            #print(f"   > [{j+1}] Weak → 7") 
                            pitches = self._gen_7_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            note_dicts[j+2]["pitch"] = pitches[2]
                            note_dicts[j+3]["pitch"] = pitches[3]
                            note_dicts[j+4]["pitch"] = pitches[4]
                            note_dicts[j+5]["pitch"] = pitches[5]
                            note_dicts[j+6]["pitch"] = pitches[6]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}, {pitches[2]}, {pitches[3]}, {pitches[4]}, {pitches[5]}, {pitches[6]}")
                            j += 7
                        elif nweak == 8:
                            # 8 weak notes 
                            #print(f"   > [{j+1}] Weak → 8") 
                            pitches = self._gen_8_weak(prev_pitch, next_strong_pitch, ch, pitch_low, pitch_high, ref_pitches)
                            # save pitch 
                            note_dicts[j]["pitch"] = pitches[0]
                            note_dicts[j+1]["pitch"] = pitches[1]
                            note_dicts[j+2]["pitch"] = pitches[2]
                            note_dicts[j+3]["pitch"] = pitches[3]
                            note_dicts[j+4]["pitch"] = pitches[4]
                            note_dicts[j+5]["pitch"] = pitches[5]
                            note_dicts[j+6]["pitch"] = pitches[6]
                            note_dicts[j+7]["pitch"] = pitches[7]
                            #print(f"   > Result → {pitches[0]}, {pitches[1]}, {pitches[2]}, {pitches[3]}, {pitches[4]}, {pitches[5]}, {pitches[6]}, {pitches[7]}")
                            j += 8
                        else:
                            # N weak notes 
                            raise NotImplementedError(f"Error: unknown nweak={nweak}, TODO...") 
                    elif nt["bstrong"]:
                        # strong
                        # update 
                        prev_pitch = note_dicts[j]["pitch"]
                        j += 1
                    else:
                        # rest 
                        j += 1

                # 4. save notes
                # create melody instance
                note_seq = Melody([])
                nnote = len(note_dicts)
                for k, nt in enumerate(note_dicts):
                    # create Note
                    if not nt["brest"]:
                        bar = nt["bar_idx"]
                        dur = nt["dur"]
                        start_beat = nt["st"]
                        pitch = nt["pitch"]
                        vel = 100 if nt["bstrong"] else 92

                        # Extend final note's duration of ending(+0.5)
                        if (k+1==nnote and start_beat+dur<=3.5) \
                           or (k+1<nnote and note_dicts[k+1]["bar_idx"] > bar and start_beat+dur<=3.5) \
                           or (k+1<nnote and start_beat+dur+0.5 < note_dicts[k+1]["st"]):
                            dur += 0.5
                            if start_beat+dur==4.0:
                                dur -= 0.25

                        # create note
                        note = NoteEvent(pitch, bar, start_beat, dur, vel)

                        # save note
                        note_seq.add(note)

                # Deal with final note's duration of previous section
                if len(note_dicts) > 0 and note_dicts[0]["bar_idx"] < 0:
                    # only when current section starts with [Pickup]
                    st = note_dicts[0]["st"]
                    if len(melody_seq) > 1 and len(melody_seq[-1].notes) > 0:
                        preend = melody_seq[-1].notes[-1].end_beat()
                        if preend > st:
                            melody_seq[-1].notes[-1].dur_beat -= (preend-st)
 
                # save melody
                melody_seq.append(note_seq)

                # save plan
                if stype not in section_melody_plans:
                    section_melody_plans[stype] = note_dicts
                # else:
                #     raise '111'
            else:
                # Intro / Outro
                pitch_low = self.pitch_low
                pitch_high = self.pitch_high
                phrase_nchars = []
                pre_bar_chord = chords[-1]

                melody_seq.append(None)
             
            # section meta
            sec_meta = {
                "section_index": i,
                "bar_index": global_bar_idx,
                "type": stype.name,
                "num_bars": nch,
                "pitch_low": pitch_low,
                "pitch_high": pitch_high,
                "phrase_nchars": phrase_nchars,
                "chords": [str(ch) for ch in chords],
            }
            section_metas.append(sec_meta)
  
            # update  
            global_bar_idx += nch
 
        # song meta
        song_meta = {
            "tempo_class": self.tempo_class.name,
            "key": self.key.value,
            "mode": self.mode.value,
            "quant_unit": self.quant_unit,
            "structure": section_metas,
        }

        # Trim long note : <= 2.0(half note)
        if TRIM_LONG_NOTE:
            # 
            MAX_DUR = 2.0
            nsec = len(melody_seq)
            for i, melody in enumerate(melody_seq):
                if i == 0 or i == nsec-1:
                    # ignore first/final section
                    continue 
                elif len(melody) > 0:
                    # loop each note and trim
                    for j, nt in enumerate(melody.notes):
                        if nt.dur_beat > MAX_DUR:
                            nt.dur_beat = MAX_DUR 
 
        # return
        return melody_seq, song_meta
 
    ####################################################################
    ## Private 
    ####################################################################
    def _get_scale_degree_from_pitch(self, ch, p):
        """ Return scales for chord """
        root = self.key.value 
        if self.mode == Mode.Minor:
            root += 3
        p12 = (p - root)%12
        #print( " ch =", ch, root, p, p12, ch.scales)
        if p12 in ch.scales:
            d = ch.scales.index(p12)+1
            d = min(7, d) # temp solution
            return d
        else:
            return None
    def _is_in_range(self, low, high, p):
        return low <= p <= high
    def _nearest_scale_pitch_down(self, ch, p):
        for offset in range(-1,-12,-1):
            cand = p+offset
            if self._get_scale_degree_from_pitch(ch, cand) is not None:
                return cand
        return p
    def _nearest_scale_pitch_up(self, ch, p):
        for offset in range(1, 12):
            cand =p+offset
            if self._get_scale_degree_from_pitch(ch, cand) is not None:
                return cand
        return p
    def _degree_to_near_pitch(self, d, lastp):
        root = self.key.value 
        if self.mode == Mode.Minor:
            root += 3
        p = d + root
        while lastp - p > 6: p += 12
        while p - lastp > 6: p -= 12
        return p
    def _choose_chord_pitch_near_down(self, lastp, ch):
        cands = []
        for d in ch.degrees:
            p = self._degree_to_near_pitch(d, lastp)
            if p < lastp:
                cands.append(p)
        if not cands:
            return -1
        return min(cands, key=lambda p: abs(p - lastp))
    def _choose_chord_pitch_near_up(self, lastp, ch):
        cands = []
        for d in ch.degrees:
            p = self._degree_to_near_pitch(d, lastp)
            if p > lastp:
                cands.append(p)
        if not cands:
            return -1
        return min(cands, key=lambda p: abs(p - lastp))

    def _gen_8_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches):
        """ sample 8 weak notes """
        # 1+7
        # 7
        mn7 = self._gen_7_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[1:] if ref_pitches != None else None) 
        # 1 
        mn1 = self._gen_1_weak(lastmn, mn7[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None) 
        # return
        mns = [mn1,] + mn7
        return mns

    def _gen_7_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches):
        """ sample 7 weak notes """
        # 1+6
        # 6
        mn6 = self._gen_6_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[1:] if ref_pitches != None else None) 
        # 1 
        mn1 = self._gen_1_weak(lastmn, mn6[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None) 
        # return
        mns = [mn1,] + mn6
        return mns
        
    def _gen_6_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches):
        """ sample 6 weak notes """
        # 1+5
        # 5
        mn5 = self._gen_5_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[1:] if ref_pitches != None else None) 
        # 1 
        mn1 = self._gen_1_weak(lastmn, mn5[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None) 
        # return
        mns = [mn1,] + mn5
        return mns

    def _gen_5_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches):
        """ sample 5 weak notes """
        # 1+4、2+3、3+2、4+1
        k = self.rand.choice([0,1,2,3])
        if k == 0:
            # 1+4 
            # 4
            mn4 = self._gen_4_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[1:] if ref_pitches != None else None) 
            # 1 
            mn1 = self._gen_1_weak(lastmn, mn4[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None) 
            # return
            mns = [mn1,] + mn4
        elif k == 1:
            # 2+3
            # 3
            mn3 = self._gen_3_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[2:] if ref_pitches != None else None) 
            # 2
            mn2 = self._gen_2_weak(lastmn, mn3[0], ch, plow, phigh, ref_pitches[:2] if ref_pitches != None else None) 
            # return
            mns = mn2 + mn3
        elif k == 2:
            # 3+2
            # 2
            mn2 = self._gen_2_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[3:] if ref_pitches != None else None)
            # 3
            mn3 = self._gen_3_weak(lastmn, mn2[0], ch, plow, phigh, ref_pitches[:3] if ref_pitches != None else None)
            # return
            mns = mn3 + mn2
        else:
            # 4+1
            # 1
            mn1 = self._gen_1_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[4] if ref_pitches != None else None) 
            # 4
            mn4 = self._gen_3_weak(lastmn, mn1, ch, plow, phigh, ref_pitches[:4] if ref_pitches != None else None) 
            # return
            mns = mn4 + [mn1,]
        return mns

    def _gen_4_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches):
        """ sample 4 weak notes """
        # 2+2、1+3、3+1
        k = self.rand.choice([0,1,2])
        if k == 0:
            # 2+2 
            # 2
            mn2 = self._gen_2_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[2:] if ref_pitches != None else None) 
            # 2 
            mn1 = self._gen_2_weak(lastmn, mn2[0], ch, plow, phigh, ref_pitches[:2] if ref_pitches != None else None) 
            # return
            mns = mn1 + mn2
        elif k == 1:
            # 1+3
            # 3
            mn3 = self._gen_3_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[1:] if ref_pitches != None else None) 
            # 1
            mn1 = self._gen_1_weak(lastmn, mn3[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None) 
            # return
            mns = [mn1,] + mn3
        else:
            # 3+1
            # 1
            mn1 = self._gen_1_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[3] if ref_pitches != None else None) 
            # 3
            mn3 = self._gen_3_weak(lastmn, mn1, ch, plow, phigh, ref_pitches[:3] if ref_pitches != None else None) 
            # return
            mns = mn3 + [mn1,]
        return mns

    def _gen_3_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches):
        """ sample 3 weak notes """
        # 2+1 or 1+2
        k = self.rand.choice([0,1])
        if k == 0:
            # 2+1 
            # 1
            mn1 = self._gen_1_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[2] if ref_pitches != None else None) 
            # 2 
            mn2 = self._gen_2_weak(lastmn, mn1, ch, plow, phigh, ref_pitches[:2] if ref_pitches != None else None) 
            if lastmn < 0:
                # prevent appog pitch out of range
                if mn2[0] < plow or mn2[0] > phigh or mn2[1] < plow or mn2[1] > phigh:
                    mn2 = self._gen_2_weak(128, mn1, ch, plow, phigh, ref_pitches[:2] if ref_pitches != None else None) 
            # return
            mns = mn2 + [mn1,]
        else:
            # 1+2
            # 2
            mn2 = self._gen_2_weak(lastmn, nextmn, ch, plow, phigh, ref_pitches[1:] if ref_pitches != None else None) 
            # 1
            mn1 = self._gen_1_weak(lastmn, mn2[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None) 
            if lastmn < 0:
                # prevent appog pitch out of range
                if mn1 < plow:
                    mn1 = self._gen_1_weak(128, mn2[0], ch, plow, phigh, ref_pitches[0] if ref_pitches != None else None)
            # return
            mns = [mn1,] + mn2
        return mns

    def _gen_1_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitch=None, max_interval=12):
        """ sample 1 weak note """ 
        candis = []
        nextd = self._get_scale_degree_from_pitch(ch, nextmn) # next degree[1~7]
        if nextd == None:
            nextmn += 1
            nextd = self._get_scale_degree_from_pitch(ch, nextmn)
        if nextd == None:
            nextmn -= 1
            nextd = self._get_scale_degree_from_pitch(ch, nextmn)
        if lastmn >= 0 and lastmn <= 127:
            lastd = self._get_scale_degree_from_pitch(ch, lastmn) # last degree[1~7]
            if lastd == None:
                lastmn += 1
                lastd = self._get_scale_degree_from_pitch(ch, lastmn)
            if lastd == None:
                lastmn -= 1
                lastd = self._get_scale_degree_from_pitch(ch, lastmn)
        elif lastmn > 127:
            lastd = 128
        else:
            lastd = -1 
        if nextmn > lastmn:
            #           nextmn
            #         ↗ 
            # lastmn
            if nextd == 1:
                if lastd == 7:
                    # 7 → 7/1/2/ch↓ → 1
                    mn2 = self._nearest_scale_pitch_up(ch, nextmn) 
                    candis += [lastmn, nextmn,] # 7/1
                    if self._is_in_range(plow, phigh, mn2): candis += [mn2,] #2
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)
                elif lastd == 6:
                    # 6 → 6/7/1/2/ch↓ → 1
                    mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                    mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += [lastmn, mn7, nextmn] # 6/7/1 
                    if self._is_in_range(plow, phigh, mn2): candis += [mn2,] #2
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)
                else:
                    # 1/2/3/4/5 → 6/1/ch↓ → 1
                    mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                    mn6 = self._nearest_scale_pitch_down(ch, mn7)
                    candis += [mn6, nextmn,] # 6/1
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
            elif nextd == 2:
                if lastd == 1:
                    # 1 → 1/2/3/ch↓ → 2
                    mn3 = self._nearest_scale_pitch_up(ch, nextmn) 
                    candis += [lastmn, nextmn,] # 1/2
                    if self._is_in_range(plow, phigh, mn3): candis += [mn3,] #3 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)
                else:
                    # 3/4/5/6/7 → 1/2/ch↓ → 2   |boost 1
                    mn1 = self._nearest_scale_pitch_down(ch, nextmn) 
                    candis += 5*[mn1,] #1 
                    candis += [nextmn,] #2
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
            elif nextd == 3:
                if lastd == 2:
                    # 2 → 2/3/4/ch↓/ch↑ → 3
                    mn4 = self._nearest_scale_pitch_up(ch, nextmn) 
                    candis += [lastmn, nextmn,] # 2/3
                    if self._is_in_range(plow, phigh, mn4): candis += [mn4,]  #4
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)
                    mnch2 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch2):
                        candis += [mnch2,] #(↑chord)
                elif lastd == 1:
                    # 1 → 2/1/3 → 3   |boost2
                    mn2 = self._nearest_scale_pitch_down(ch, nextmn) 
                    candis += 5*[mn2,] #2 
                    candis += [lastmn,] #1
                    candis += [nextmn,] #3 
                elif lastd == 7:
                    # 7 → 2/3/ch↓ → 3   |boost2
                    mn2 = self._nearest_scale_pitch_down(ch, nextmn) 
                    candis += 5*[mn2,] #2 
                    candis += [nextmn,] #3
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)
                else:
                    # 4/5/6 → 2/3/ch↓ → 3   |boost2
                    mn2 = self._nearest_scale_pitch_down(ch, nextmn) 
                    candis += 5*[mn2,] #2
                    candis += [nextmn,] #3
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)
            elif nextd == 4:
                if lastd == 3:
                    # 3 → 3/4/5/ch↓ → 4
                    mn5 = self._nearest_scale_pitch_up(ch, nextmn)  
                    candis += [lastmn, nextmn,] # 3/4 
                    if self._is_in_range(plow, phigh, mn5): candis += [mn5,] # 5
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
                elif lastd == 2:
                    # 2 → 2/3/4→ 4   |boost3
                    mn3 = self._nearest_scale_pitch_down(ch, nextmn)  
                    candis += [lastmn,] #2
                    candis += 5*[mn3,] #3
                    candis += [nextmn,] #4 
                else:
                    # 4/5/6/7/1 → 3/4/ch↓ → 4   |boost3
                    mn3 = self._nearest_scale_pitch_down(ch, nextmn)  
                    candis += 5*[mn3,] #3  
                    candis += [nextmn,] #4 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)  
            elif nextd == 5:
                if lastd == 4:
                    # 4 → 4/5/6/ch↓ → 5
                    mn6 = self._nearest_scale_pitch_up(ch, nextmn)  
                    candis += [lastmn, nextmn,] # 4/5
                    if self._is_in_range(plow, phigh, mn6): candis += [mn6,] #6
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
                elif lastd == 3:
                    # 3 → 3/4/5 → 5
                    mn4 = self._nearest_scale_pitch_down(ch, nextmn)  
                    candis += [lastmn,] #3
                    candis += [mn4,] #4 
                    candis += [nextmn,] #5 
                else:
                    # 4/5/6/7/1/2 → 3/5/ch↓ → 5   |boost3
                    mn4 = self._nearest_scale_pitch_down(ch, nextmn)
                    mn3 = self._nearest_scale_pitch_down(ch, mn4)
                    candis += 5*[mn3,] #3
                    candis += [nextmn,] #5 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)  
            elif nextd == 6:
                if lastd == 5:
                    # 5 → 5/6/ch↓ → 6
                    candis += [lastmn, nextmn,] # 5/6
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
                elif lastd == 4:
                    # 4 → 5/6 → 6  |boost5
                    mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += 5*[mn5,] #5
                    candis += [nextmn,] #6 
                else:
                    # 6/7/1/2/3 → 5/6/ch↓ → 6   |boost5
                    mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += 5*[mn5,] #5
                    candis += [nextmn,] #6 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord)  
            elif nextd == 7:
                if lastd == 6:
                    # 6 → 6/7/ch↓ → 7
                    candis += [lastmn, nextmn,] # 6/7
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
                elif lastd == 5:
                    # 5 → 6/7/ch↓ → 7  |boost6
                    mn6 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += 5*[mn6,] #6
                    candis += [nextmn,] #7 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
                else:
                    # 7/1/2/3/4 → 6/7/ch↓ → 7   |boost6
                    mn6 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += 5*[mn6,] #6 
                    candis += [nextmn,] #7 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↓chord) 
            else:
                raise NotImplementedError(f"Error: unknown next_d={nextd}, nextmn={nextmn}")
        elif nextmn < lastmn:
            #  lastmn
            #         ↘ 
            #            nextmn 
            if nextd == 1:
                if lastd == 2:
                    # 2 → 2/1/7/ch↑/ch↓ → 1 
                    mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += [lastmn, nextmn,] # 2/1
                    if self._is_in_range(plow, phigh, mn7): candis += [mn7,]  #7
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                    mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch2):
                        candis += [mnch2,] #(↓chord)
                elif lastd == 3:
                    # 3 → 2/1/ch↑ → 1    | boost2
                    mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn2,] #2 
                    candis += [nextmn,] #1
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                else:
                    # 1/7/6/5/4 → 2/1/ch↑ → 1    | boost2
                    mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn2,] #2 
                    candis += [nextmn,] #1
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            elif nextd == 2:
                if lastd == 3:
                    # 3 → 3/2/1/ch↑ → 2 
                    mn1 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += [lastmn, nextmn,] # 3/2 
                    if self._is_in_range(plow, phigh, mn1): candis += [mn1,]  # 1
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord) 
                elif lastd == 4:
                    # 4 → 3/2/ch↑ → 2    | boost3
                    mn3 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn3,] #3 
                    candis += [nextmn,] #2
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                else:
                    # 2/1/7/6/5 → 3/2/ch↑ → 2    | boost3
                    mn3 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn3,] #3 
                    candis += [nextmn,] #2
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            elif nextd == 3:
                if lastd == 4:
                    # 4 → 4/3/2/ch↑ → 3 
                    mn2 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += [lastmn, nextmn,] # 4/3
                    if self._is_in_range(plow, phigh, mn2): candis += [mn2,]  # 2
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord) 
                elif lastd == 5:
                    # 5 → 4/3/ch↑ → 3    | boost4
                    mn4 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn4,] #4 
                    candis += [nextmn,] #3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                else:
                    # 3/2/1/7/6 → 3/ch↑ → 3  
                    candis += [nextmn,] #3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            elif nextd == 4:
                if lastd == 5:
                    # 5 → 5/4/3/ch↑ → 4
                    mn3 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += [lastmn, nextmn,] # 5/4 
                    if self._is_in_range(plow, phigh, mn3): candis += [mn3,]  # 3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord) 
                elif lastd == 6:
                    # 6 → 5/4/ch↑ → 4    | boost5
                    mn5 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn5,] #5
                    candis += [nextmn,] #4
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                else:
                    # 4/3/2/1/7 → 5/4/ch↑ → 4    | boost5
                    mn5 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn5,] #5 
                    candis += [nextmn,] #4
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            elif nextd == 5:
                if lastd == 6:
                    # 6 → 6/5/ch↑ → 5
                    candis += [lastmn, nextmn,] # 6/5 
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord) 
                elif lastd == 7:
                    # 7 → 6/5/ch↑ → 5    | boost6
                    mn6 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn6,] #6
                    candis += [nextmn,] #5
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                else:
                    # 5/4/3/2/1 → 6/5/ch↑ → 5    | boost6
                    mn6 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += 5*[mn6,] #6
                    candis += [nextmn,] #5
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            elif nextd == 6:
                if lastd == 7:
                    # 7 → 7/6/5/ch↑ → 6
                    mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                    candis += [lastmn, nextmn,] # 7/6
                    if self._is_in_range(plow, phigh, mn5): candis += [mn5,]  # 5
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord) 
                elif lastd == 1:
                    # 1 → 7/1/6/ch↑ → 6 
                    mn7 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += [mn7, lastmn, nextmn,] # 7/1/6
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
                else:
                    # 5/4/3/2 → 1/6/ch↑ → 6
                    mn7 = self._nearest_scale_pitch_up(ch, nextmn)
                    mn1 = self._nearest_scale_pitch_up(ch, mn7)
                    candis += [mn1, nextmn, ] # 1/6
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            elif nextd == 7:
                if lastd == 1:
                    # 1 → 1/7/ch↑ → 7
                    candis += [lastmn, nextmn,] # 1/7 
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord) 
                else:
                    # 6/5/4/3/2 → 1/7/ch↑ → 7
                    mn1 = self._nearest_scale_pitch_up(ch, nextmn)
                    candis += [mn1, nextmn,] # 1/7
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis += [mnch1,] #(↑chord)
            else:
                raise NotImplementedError(f"Error: unknown next_d={nextd}, nextmn={nextmn}")
        else:
            # lastmn == nextmn
            if nextd == lastd == 1:
                # 1 → 6/2/ch↑/ch↓ → 1 
                mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, mn7)
                if self._is_in_range(plow, phigh, mn2): candis += [mn2,]  # 2
                if self._is_in_range(plow, phigh, mn6): candis += [mn6,]  # 6
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord)
            elif nextd == lastd == 2:
                # 2 → 1/3/ch↑/ch↓ → 2 
                mn3 = self._nearest_scale_pitch_up(ch, nextmn)
                mn1 = self._nearest_scale_pitch_down(ch, nextmn)
                if self._is_in_range(plow, phigh, mn1): candis += [mn1,]  # 1
                if self._is_in_range(plow, phigh, mn3): candis += [mn3,]  # 3
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord)
            elif nextd == lastd == 3:
                # 3 → 2/4/ch↑/ch↓ → 3
                mn4 = self._nearest_scale_pitch_up(ch, nextmn)
                mn2 = self._nearest_scale_pitch_down(ch, nextmn)
                if self._is_in_range(plow, phigh, mn2): candis += [mn2,]  # 2
                if self._is_in_range(plow, phigh, mn4): candis += [mn4,]  # 4
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord)
            elif nextd == lastd == 4:
                # 4 → 3/5/ch↑/ch↓ → 4
                mn5 = self._nearest_scale_pitch_up(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, nextmn)
                if self._is_in_range(plow, phigh, mn3): candis += [mn3,]  # 3
                if self._is_in_range(plow, phigh, mn5): candis += [mn5,]  # 5
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord)
            elif nextd == lastd == 5:
                # 5 → 4/6/ch↑/ch↓ → 5
                mn6 = self._nearest_scale_pitch_up(ch, nextmn)
                mn4 = self._nearest_scale_pitch_down(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, mn4)
                if self._is_in_range(plow, phigh, mn3): candis += [mn3,]  # 3
                if self._is_in_range(plow, phigh, mn4): candis += [mn4,]  # 4
                if self._is_in_range(plow, phigh, mn6): candis += [mn6,]  # 6
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord)
            elif nextd == lastd == 6:
                # 6 → 5/1/ch↑/ch↓ → 6
                mn7 = self._nearest_scale_pitch_up(ch, nextmn)
                mn1 = self._nearest_scale_pitch_up(ch, mn7)
                mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                if self._is_in_range(plow, phigh, mn1): candis += [mn1,]  # 1
                if self._is_in_range(plow, phigh, mn5): candis += [mn5,]  # 5
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord)
            elif nextd == lastd == 7:
                # 7 → 6/1/ch↑/ch↓ → 7
                mn1 = self._nearest_scale_pitch_up(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, nextmn)
                if self._is_in_range(plow, phigh, mn1): candis += [mn1,]  # 1
                if self._is_in_range(plow, phigh, mn6): candis += [mn6,]  # 6
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis += [mnch1,] #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch2):
                    candis += [mnch2,] #(↓chord) 
            else:
                raise NotImplementedError(f"Error: unknown next_d={nextd}, lastd={lastd}")
        # --------- post ---------
        # filter by interval
        filter_candis = []
        for x in candis:
            up = nextmn+max_interval
            down = nextmn-max_interval
            if down <= x <= up:
                filter_candis.append(x) 
        if len(filter_candis) > 0:
            candis = filter_candis
        # use ref if possible
        if ref_pitch != None and ref_pitch > 0:
            if ref_pitch in candis:
                # in candidate
                sel_pitch = ref_pitch
            else:
                # find nearest
                self.rand.shuffle(candis)
                min_diff = 9999
                nearest_pitch = candis[0]
                for x in candis:
                    diff = abs(x - ref_pitch)
                    if diff < min_diff:
                        min_diff = diff
                        nearest_pitch = x
                sel_pitch = nearest_pitch
        else:
            # sample
            sel_pitch = self.rand.choice(candis) 
        # return 
        return sel_pitch

    def _gen_2_weak(self, lastmn, nextmn, ch, plow, phigh, ref_pitches=None, max_interval=12):
        """ sample 2 weak notes """  
        candis = []
        nextd = self._get_scale_degree_from_pitch(ch, nextmn) # next degree[1~7]
        if nextd == None:
            nextmn += 1
            nextd = self._get_scale_degree_from_pitch(ch, nextmn)
        if nextd == None:
            nextmn -= 1
            nextd = self._get_scale_degree_from_pitch(ch, nextmn)
        if lastmn >= 0 and lastmn <= 127:
            lastd = self._get_scale_degree_from_pitch(ch, lastmn) # last degree[1~7]
            if lastd == None:
                lastmn += 1
                lastd = self._get_scale_degree_from_pitch(ch, lastmn)
            if lastd == None:
                lastmn -= 1
                lastd = self._get_scale_degree_from_pitch(ch, lastmn)
        elif lastmn > 127:
            lastd = 128
        else:
            lastd = -1 
        if nextmn > lastmn:
            #           nextmn
            #         ↗ 
            # lastmn
            if nextd == 1:
                mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, mn7)
                mn5 = self._nearest_scale_pitch_down(ch, mn6)
                if lastd == 7:
                    # 7 → 6-7/7-2/7-1/1-2/2-2/2-1/1-ch↑ → 1
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, lastmn]) # 6-7
                    if self._is_in_range(plow, phigh, mn2): candis.append([lastmn, mn2]) # 7-2
                    candis.append([lastmn, nextmn]) # 7-1
                    if self._is_in_range(plow, phigh, mn2): candis.append([nextmn, mn2]) # 1-2
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn2]) # 2-2
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn7, nextmn]) # 2-1
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 1-ch↑
                elif lastd == 6:
                    # 6 → 5-6/6-6/6-7/7-7/7-1/7-2/1-2/1-ch↑ → 1
                    if self._is_in_range(plow, phigh, mn5): candis.append([mn5, lastmn]) # 5-6
                    candis.append([lastmn, lastmn]) # 6-6
                    candis.append([lastmn, mn7]) # 6-7
                    candis.append([mn7, mn7]) # 7-7
                    candis.append([mn7, nextmn]) # 7-1
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn7, mn2]) # 7-2
                    #candis.append([nextmn, mn7]) # 1-7
                    if self._is_in_range(plow, phigh, mn2): candis.append([nextmn, mn2]) # 1-2
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 1-ch↑
                elif lastd == 5:
                    # 5 → 5-6/6-6/6-7/6-1/6-2/1-2 → 1   |boost6-7
                    candis.append([lastmn, mn6])  # 5-6
                    candis.append([mn6, mn6])  # 6-6
                    candis.append([mn6, mn7])
                    candis.append([mn6, mn7])
                    candis.append([mn6, mn7])
                    candis.append([mn6, mn7]) # 6-7
                    candis.append([mn6, nextmn]) # 6-1
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn6, mn2]) # 6-2
                    #candis.append([nextmn, mn7]) # 1-7
                    if self._is_in_range(plow, phigh, mn2): candis.append([nextmn, mn2]) # 1-2
                elif lastd == 4:
                    # 4 → 6-7/5-6 → 1 
                    candis.append([mn6, mn7]) # 6-7
                    candis.append([mn5, mn6])
                    candis.append([mn5, mn6]) # 5-6 
                else:
                    # 1/2/3 → 6-7/5-6/1-2/ch↓-ch↓  → 1
                    candis.append([mn6, mn7]) # 6-7
                    if self._is_in_range(plow, phigh, mn2): candis.append([nextmn, mn2]) # 1-2
                    #candis.append([nextmn, mn7]) # 1-7
                    if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn6]) # 5-6 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    mnch2 = self._choose_chord_pitch_near_down(mnch1, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↓-ch↓
            elif nextd == 2:
                mn3 = self._nearest_scale_pitch_up(ch, nextmn)
                mn1 = self._nearest_scale_pitch_down(ch, nextmn)
                mn7 = self._nearest_scale_pitch_down(ch, mn1)
                mn6 = self._nearest_scale_pitch_down(ch, mn7)
                if lastd == 1:
                    # 1 → 1-1/1-2/1-3/2-1/2-2/2-3/3-1/3-2/3-3/2-ch↑ → 2
                    candis.append([lastmn, lastmn]) # 1-1
                    candis.append([lastmn, nextmn]) # 1-2
                    if self._is_in_range(plow, phigh, mn3): candis.append([lastmn, mn3]) # 1-3
                    candis.append([nextmn, lastmn]) # 2-1
                    candis.append([nextmn, nextmn]) # 2-2
                    if self._is_in_range(plow, phigh, mn3): candis.append([nextmn, mn3]) # 2-3
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, lastmn]) # 3-1
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, nextmn]) # 3-2
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn3]) # 3-3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 2-ch↑
                elif lastd == 7:
                    # 7 → 7-1/1-1/1-2/1-3/2-1/1-ch↑ → 2 
                    candis.append([lastmn, mn1]) # 7-1 
                    candis.append([mn1, mn1]) # 1-1
                    candis.append([mn1, nextmn]) # 1-2
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn1, mn3]) # 1-3
                    candis.append([nextmn, mn1]) # 2-1
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn1, mnch1]) # 1-ch↑
                elif lastd == 6:
                    # 6 → 6-1/7-1/1-1/1-2/1-3 → 2   |boost7-1
                    candis.append([lastmn, mn1]) # 6-1
                    candis.append([mn7, mn1])
                    candis.append([mn7, mn1])
                    candis.append([mn7, mn1]) # 7-1
                    candis.append([mn1, mn1]) # 1-1
                    candis.append([mn1, nextmn]) # 1-2
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn1, mn3]) # 1-3
                else:
                    # 2/3/4/5 → 6-1/1-1/1-2/1-3/ch↓-1/ch↓-ch↓ → 2
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn1]) # 6-1
                    candis.append([mn1, mn1]) # 1-1
                    candis.append([mn1, nextmn]) # 1-2
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn1, mn3]) # 1-3
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    mnch2 = self._choose_chord_pitch_near_down(mnch1, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mnch1, mn1]) # ch↓-1
                        if self._is_in_range(plow, phigh, mnch2):
                            candis.append([mnch2, mnch1]) # ch↓-ch↓
            elif nextd == 3:
                mn4 = self._nearest_scale_pitch_up(ch, nextmn)
                mn2 = self._nearest_scale_pitch_down(ch, nextmn)
                mn1 = self._nearest_scale_pitch_down(ch, mn2)
                mn7 = self._nearest_scale_pitch_down(ch, mn1)
                if lastd == 2:
                    # 2 → 1-2/2-2/2-3/2-4/3-2/3-3/3-4/4-2/3-ch↑/3-ch↓ → 3
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, lastmn]) # 1-2
                    candis.append([mn2, mn2]) # 2-2
                    candis.append([mn2, nextmn]) # 2-3
                    candis.append([lastmn, mn4]) # 2-4
                    candis.append([nextmn, mn2]) # 3-2
                    candis.append([nextmn, nextmn]) # 3-3
                    if self._is_in_range(plow, phigh, mn4): candis.append([nextmn, mn4]) # 3-4
                    if self._is_in_range(plow, phigh, mn4): candis.append([mn4, lastmn]) # 4-2
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 3-ch↑
                    mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch2):
                        candis.append([nextmn, mnch2]) # 3-ch↓
                elif lastd == 1:
                    # 1 → 1-2/2-2/2-3/2-4/3-2/3-4/2-ch↑ → 3
                    candis.append([lastmn, mn2]) # 1-2 
                    candis.append([mn2, mn2]) # 2-2
                    candis.append([mn2, nextmn]) # 2-3
                    if self._is_in_range(plow, phigh, mn4): candis.append([mn2, mn4]) # 2-4
                    candis.append([nextmn, mn2]) # 3-2
                    if self._is_in_range(plow, phigh, mn4): candis.append([nextmn, mn4]) # 3-4
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn2, mnch1]) # 2-ch↑
                elif lastd == 7:
                    # 7 → 1-2/2-4 → 3   |boost1-2
                    candis.append([mn1, mn2])
                    candis.append([mn1, mn2])
                    candis.append([mn1, mn2])
                    candis.append([mn1, mn2]) # 1-2
                    if self._is_in_range(plow, phigh, mn4): candis.append([mn2, mn4]) # 2-4
                else:
                    # 3/4/5/6 → 1-2/2-4 → 3     |boost1-2
                    candis.append([mn1, mn2])
                    candis.append([mn1, mn2])
                    candis.append([mn1, mn2])
                    candis.append([mn1, mn2]) # 1-2
                    if self._is_in_range(plow, phigh, mn4): candis.append([mn2, mn4]) # 2-4 
            elif nextd == 4:
                mn5 = self._nearest_scale_pitch_up(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, nextmn)
                mn2 = self._nearest_scale_pitch_down(ch, mn3)
                mn1 = self._nearest_scale_pitch_down(ch, mn2)
                if lastd == 3:
                    # 3 → 2-3/3-3/3-4/4-3/4-5/3-ch↑ → 4
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, lastmn]) # 2-3
                    candis.append([lastmn, lastmn]) # 3-3
                    candis.append([lastmn, nextmn]) # 3-4
                    candis.append([nextmn, lastmn]) # 4-3
                    if self._is_in_range(plow, phigh, mn5): candis.append([nextmn, mn5]) # 4-5
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn3, mnch1]) # 3-ch↑
                elif lastd == 2:
                    # 2 → 2-3/3-3/4-3/3-ch↑ → 4
                    candis.append([lastmn, mn3]) # 2-3
                    candis.append([mn3, mn3]) # 3-3
                    candis.append([nextmn, mn3]) # 4-3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn3, mnch1]) # 3-ch↑
                elif lastd == 1:
                    # 1 → 2-3 → 4
                    candis.append([mn2, mn3]) # 2-3 
                else:
                    # 3/4/5/6/7 → 2-3 → 4 
                    candis.append([mn2, mn3]) # 2-3 
            elif nextd == 5:
                mn6 = self._nearest_scale_pitch_up(ch, nextmn)
                mn4 = self._nearest_scale_pitch_down(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, mn4)
                mn2 = self._nearest_scale_pitch_down(ch, mn3)
                if lastd == 4:
                    # 4 → 3-4/4-6/5-5/5-6/6-5/6-6/ch↑-5 → 5 
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, lastmn]) # 3-4
                    #if self._is_in_range(plow, phigh, mn6): candis.append([lastmn, mn6]) # 4-6
                    candis.append([nextmn, nextmn]) # 5-6
                    if self._is_in_range(plow, phigh, mn6): candis.append([nextmn, mn6]) # 5-6
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, nextmn]) # 6-5
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn6]) # 6-6
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mnch1, nextmn]) # ch↑-5
                elif lastd == 3:
                    # 3 → 2-3/3-3/3-4/3-5/4-4/4-5/4-6/5-6/4-ch↑ → 5
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, lastmn]) # 2-3
                    candis.append([lastmn, lastmn]) # 3-3
                    candis.append([lastmn, mn4]) # 3-4
                    candis.append([lastmn, nextmn]) # 3-5
                    candis.append([mn4, mn4]) # 4-5
                    candis.append([mn4, nextmn]) # 4-5
                    #if self._is_in_range(plow, phigh, mn6): candis.append([mn4, mn6]) # 4-6
                    if self._is_in_range(plow, phigh, mn6): candis.append([nextmn, mn6]) # 5-6
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn4, mnch1]) # 4-ch↑
                elif lastd == 2:
                    # 2 → 2-3/3-3/3-4/3-5/3-6/5-6 → 5   |boost3-4
                    candis.append([lastmn, mn3]) # 2-3 
                    candis.append([mn3, mn3]) # 3-3 
                    candis.append([mn3, mn4])
                    candis.append([mn3, mn4])
                    candis.append([mn3, mn4])
                    candis.append([mn3, mn4]) # 3-4 
                    candis.append([mn3, nextmn]) # 3-5
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn3, mn6]) # 3-6
                    if self._is_in_range(plow, phigh, mn6): candis.append([nextmn, mn6]) # 5-6
                else:
                    # 5/6/7/1 → 2-3/3-4/3-6/5-6/ch↓-ch↓ → 5   |boost3-4
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn3]) # 2-3 
                    candis.append([mn3, mn4])
                    candis.append([mn3, mn4])
                    candis.append([mn3, mn4]) # 3-4 
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn3, mn6]) # 3-6 
                    if self._is_in_range(plow, phigh, mn6): candis.append([nextmn, mn6]) # 5-6
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    mnch2 = self._choose_chord_pitch_near_down(mnch1, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↓-ch↓
            elif nextd == 6:
                mn7 = self._nearest_scale_pitch_up(ch, nextmn)
                mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                mn4 = self._nearest_scale_pitch_down(ch, mn5)
                mn3 = self._nearest_scale_pitch_down(ch, mn4)
                if lastd == 5:
                    # 5 → 5-5/5-6/6-5/5-ch↑ → 6 
                    candis.append([mn5, mn5]) # 5-5
                    candis.append([mn5, nextmn]) # 5-6
                    candis.append([nextmn, mn5]) # 6-5
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn5, mnch1]) # 5-ch↑
                elif lastd == 4:
                    # 4 → 4-5/5-5/5-6/6-5/5-ch↑ → 6
                    candis.append([mn4, mn5]) # 4-5
                    candis.append([mn5, mn5]) # 5-5 
                    candis.append([mn5, nextmn]) # 5-6
                    candis.append([nextmn, mn5]) # 6-5 
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn5, mnch1]) # 5-ch↑
                else:
                    # 6/7/1/2/3 → 4-5/3-5/ch↓-ch↓ → 6
                    candis.append([mn4, mn5]) # 4-5 
                    candis.append([mn3, mn5]) # 3-5 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    mnch2 = self._choose_chord_pitch_near_down(mnch1, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↓-ch↓
            elif nextd == 7:
                mn1 = self._nearest_scale_pitch_up(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, nextmn)
                mn5 = self._nearest_scale_pitch_down(ch, mn6)
                mn4 = self._nearest_scale_pitch_down(ch, mn5)
                if lastd == 6:
                    # 6 → 5-6/6-6/7-6/7-1/1-1 → 7 
                    if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn6]) # 5-6
                    candis.append([mn6, mn6]) # 6-6
                    candis.append([nextmn, mn6]) # 7-6
                    if self._is_in_range(plow, phigh, mn1): candis.append([nextmn, mn1]) # 7-1
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn1]) # 1-1
                elif lastd == 5:
                    # 5 → 5-6/6-6/6-7/6-ch↑ → 7
                    candis.append([mn5, mn6]) # 5-6
                    candis.append([mn6, mn6]) # 6-6 
                    candis.append([mn6, nextmn]) # 6-7
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn6, mnch1]) # 6-ch↑
                else:
                    # 7/1/2/3/4 → 5-6/6-6 → 7
                    candis.append([mn5, mn6]) # 5-6
                    candis.append([mn6, mn6]) # 6-6 
            else:
                raise NotImplementedError(f"Error: unknown next_d={nextd}, nextmn={nextmn}")
        elif nextmn < lastmn:
            #  lastmn
            #         ↘ 
            #            nextmn
            if nextd == 1:
                mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, mn7)
                mn3 = self._nearest_scale_pitch_up(ch, mn2)
                mn4 = self._nearest_scale_pitch_up(ch, mn3)
                if lastd == 2:
                    # 2 → 3-2/2-2/2-1/2-7/1-2/6-7/2-ch↓/1-ch↓ → 1
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, lastmn]) # 3-2
                    candis.append([lastmn, lastmn]) # 2-2 
                    candis.append([lastmn, nextmn]) # 2-1
                    if self._is_in_range(plow, phigh, mn7): candis.append([lastmn, mn7]) # 2-7
                    candis.append([nextmn, lastmn]) # 1-2 
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn7]) # 6-7 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn2, mnch1]) # 2-ch↓
                        candis.append([nextmn, mnch1]) # 1-ch↓
                elif lastd == 3:
                    # 3 → 3-2/2-7/1-2/2-2/2-ch↓ → 1
                    candis.append([mn3, mn2]) # 3-2
                    if self._is_in_range(plow, phigh, mn7): candis.append([mn2, mn7]) # 2-7
                    candis.append([nextmn, mn2]) # 1-2 
                    candis.append([mn2, mn2]) # 2-2 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn2, mnch1]) # 2-ch↓
                elif lastd == 4:
                    # 4 → 3-2/1-2/2-2/2-3 → 1   | boost3-2
                    candis.append([mn3, mn2]) # 3-2
                    candis.append([mn3, mn2])
                    candis.append([mn3, mn2])
                    candis.append([nextmn, mn2]) # 1-2
                    candis.append([mn2, mn2]) # 2-2 
                    candis.append([mn2, mn3]) # 2-3 
                else:
                    # 1/7/6/5 → 3-2/1-2/ch↑-ch↑ → 1   | boost3-2
                    candis.append([mn3, mn2]) # 3-2
                    candis.append([mn3, mn2])
                    candis.append([mn3, mn2])
                    candis.append([nextmn, mn2]) # 1-2
                    candis.append([mn2, mn2]) # 2-2 
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    mnch2 = self._choose_chord_pitch_near_up(mnch1, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↑-ch↑
            elif nextd == 2:
                mn3 = self._nearest_scale_pitch_up(ch, nextmn)
                mn1 = self._nearest_scale_pitch_down(ch, nextmn)
                mn4 = self._nearest_scale_pitch_up(ch, mn3)
                mn5 = self._nearest_scale_pitch_up(ch, mn4)
                if lastd == 3:
                    # 3 → 5-3/4-3/3-3/3-2/3-1/2-3/2-1/1-3/1-2/1-1/ch↑-3/2-ch↓ → 2
                    if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn3]) # 5-3
                    if self._is_in_range(plow, phigh, mn4): candis.append([mn4, mn3]) # 4-3
                    candis.append([lastmn, lastmn]) # 3-3
                    candis.append([lastmn, nextmn]) # 3-2
                    if self._is_in_range(plow, phigh, mn1): candis.append([lastmn, mn1]) # 3-1
                    candis.append([nextmn, mn3]) # 2-3
                    if self._is_in_range(plow, phigh, mn1): candis.append([nextmn, mn1]) # 2-1 
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn3]) # 1-3
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, nextmn]) # 1-2
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn1]) # 1-1 
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mnch1, mn3]) # ch↑-3
                    mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch2):
                        candis.append([nextmn, mnch2]) # 2-ch↓
                elif lastd == 4:
                    # 4 → 4-3/3-3/3-2/3-1/3-ch↓ → 2 
                    candis.append([lastmn, mn3]) # 4-3
                    candis.append([mn3, mn3]) # 3-3
                    candis.append([mn3, nextmn]) # 3-2
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn3, mn1]) # 3-1
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn3, mnch1]) # 3-ch↓
                elif lastd == 5:
                    # 5 → 5-3/4-3/3-3/3-2/3-1/2-3 → 2   |boost4-3
                    candis.append([lastmn, mn3]) # 5-3
                    candis.append([mn4, mn3])
                    candis.append([mn4, mn3])
                    candis.append([mn4, mn3]) # 4-3
                    candis.append([mn3, mn3]) # 3-3
                    candis.append([mn3, nextmn]) # 3-2
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn3, mn1]) # 3-1
                    candis.append([nextmn, mn3]) # 2-3
                else:
                    # 2/1/7/6 → 4-3/5-3/ch↑-3 → 2
                    candis.append([mn4, mn3]) # 4-3
                    candis.append([mn5, mn3]) # 5-3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mnch1, mn3]) # ch↑-3
            elif nextd == 3:
                mn4 = self._nearest_scale_pitch_up(ch, nextmn)
                mn2 = self._nearest_scale_pitch_down(ch, nextmn)
                mn1 = self._nearest_scale_pitch_down(ch, mn2)
                mn5 = self._nearest_scale_pitch_up(ch, mn4)
                mn6 = self._nearest_scale_pitch_up(ch, mn5)
                if lastd == 4:
                    # 4 → 5-5/5-4/5-2/4-4/3-4/3-2/2-2/1-2/3-ch↓ → 3
                    if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn5]) # 5-5
                    if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn4]) # 5-4
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn5, mn2]) # 5-2
                    candis.append([mn4, mn4]) # 4-4 
                    candis.append([nextmn, mn4]) # 3-4
                    if self._is_in_range(plow, phigh, mn2): candis.append([nextmn, mn2]) # 3-2
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn2]) # 2-2
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn2]) # 1-2
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 3-ch↓
                elif lastd == 5:
                    # 5 → 6-5/5-5/5-4/4-4/4-3/4-2/3-4/3-2/4-ch↓/3-ch↓ → 3 
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn5]) # 6-5
                    candis.append([mn5, mn4]) # 5-5
                    candis.append([mn5, mn4]) # 5-4
                    candis.append([mn4, mn4]) # 4-4
                    candis.append([mn4, nextmn]) # 4-3
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn4, mn2]) # 4-2
                    candis.append([nextmn, mn4]) # 3-4
                    if self._is_in_range(plow, phigh, mn2): candis.append([nextmn, mn2]) # 3-2
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn4, mnch1]) # 4-ch↓
                        candis.append([nextmn, mnch1]) # 3-ch↓
                elif lastd == 6:
                    # 6 → 6-5/5-5/5-4/5-3/5-2 → 3   |boost5-4
                    candis.append([mn6, mn5]) # 6-5
                    candis.append([mn5, mn5]) # 5-5
                    candis.append([mn5, mn4])
                    candis.append([mn5, mn4])
                    candis.append([mn5, mn4]) # 5-4
                    candis.append([mn5, nextmn]) # 5-3
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn5, mn2]) # 5-2
                else:
                    # 3/2/1/7 → 6-5/5-5/5-4/5-3/ch↑-5 → 3   |boost5-4
                    candis.append([mn6, mn5]) # 6-5
                    candis.append([mn5, mn5]) # 5-5
                    candis.append([mn5, mn4])
                    candis.append([mn5, mn4])
                    candis.append([mn5, mn4]) # 5-4
                    candis.append([mn5, nextmn]) # 5-3
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mnch1, mn5]) # ch↑-5
            elif nextd == 4:
                mn5 = self._nearest_scale_pitch_up(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, nextmn)
                mn6 = self._nearest_scale_pitch_up(ch, mn5)
                mn7 = self._nearest_scale_pitch_up(ch, mn6)
                if lastd == 5:
                    # 5 → 6-5/4-5/4-3/5-3/5-5/3-3/4-ch↓ → 4
                    if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn5]) # 6-5
                    candis.append([nextmn, mn5]) # 4-5
                    if self._is_in_range(plow, phigh, mn3): candis.append([nextmn, mn3]) # 4-3
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn5, mn3]) # 5-3
                    candis.append([mn5, mn5]) # 5-5
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn3]) # 3-3
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 4-ch↓
                elif lastd == 6:
                    # 6 → 6-5/5-5/5-4/5-3/5-ch↓ → 4 
                    candis.append([mn6, mn5]) # 6-5
                    candis.append([mn5, mn5]) # 5-5
                    candis.append([mn5, nextmn]) # 5-4
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn5, mn3]) # 5-3
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn5, mnch1]) # 5-ch↓
                elif lastd == 7:
                    # 7 → 6-5 → 4   |boost6-5
                    candis.append([mn6, mn5]) # 6-5
                else:
                    # 4/3/2/1 → 6-5 → 4   |boost6-5
                    candis.append([mn6, mn5]) # 6-5
            elif nextd == 5:
                mn6 = self._nearest_scale_pitch_up(ch, nextmn)
                mn4 = self._nearest_scale_pitch_down(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, mn4)
                mn7 = self._nearest_scale_pitch_up(ch, mn6)
                mn1 = self._nearest_scale_pitch_up(ch, mn7)
                if lastd == 6:
                    # 6 → 6-6/6-5/5-6/5-3/3-4/5-ch↓ → 5
                    candis.append([lastmn, lastmn]) # 6-6
                    candis.append([lastmn, nextmn]) # 6-5
                    candis.append([nextmn, lastmn]) # 5-6
                    if self._is_in_range(plow, phigh, mn3): candis.append([nextmn, mn3]) # 5-3
                    if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn4]) # 3-4
                    #candis.append([mn5, mn5]) # 5-5
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 5-ch↓
                elif lastd == 7:
                    # 7 → 7-6/6-6/6-5/5-6/6-ch↓ → 5 
                    candis.append([lastmn, mn6]) # 7-6
                    candis.append([mn6, mn6]) # 6-6
                    candis.append([mn6, nextmn]) # 6-5
                    candis.append([nextmn, mn6]) # 5-6 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn6, mnch1]) # 6-ch↓
                elif lastd == 1:
                    # 1 → 1-6/7-6/6-6/6-5/5-6/6-ch↓ → 5   |boost7-6
                    candis.append([lastmn, mn6]) # 1-6
                    candis.append([mn7, mn6])
                    candis.append([mn7, mn6])
                    candis.append([mn7, mn6]) # 7-6
                    candis.append([mn6, mn6]) # 6-6
                    candis.append([mn6, nextmn]) # 6-5
                    candis.append([nextmn, mn6]) # 5-6 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn6, mnch1]) # 6-ch↓
                else:
                    # 5/4/3/2 → 1-6/7-6/5-6/ch↑-ch↑ → 5   |boost1-6
                    candis.append([mn1, mn6])
                    candis.append([mn1, mn6])
                    candis.append([mn1, mn6]) # 1-6
                    candis.append([mn7, mn6]) # 7-6
                    candis.append([nextmn, mn6]) # 5-6
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    mnch2 = self._choose_chord_pitch_near_up(mnch1, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↑-ch↑
            elif nextd == 6:
                mn7 = self._nearest_scale_pitch_up(ch, nextmn)
                mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                mn4 = self._nearest_scale_pitch_down(ch, mn5)
                mn1 = self._nearest_scale_pitch_up(ch, mn7)
                mn2 = self._nearest_scale_pitch_up(ch, mn1)
                if lastd == 7:
                    # 7 → 1-7/7-7/7-6/6-7/6-5/6-ch↓ → 6
                    if self._is_in_range(plow, phigh, mn1): candis.append([mn1, lastmn]) # 1-7
                    candis.append([lastmn, lastmn]) # 7-7
                    candis.append([lastmn, nextmn]) # 7-6
                    candis.append([nextmn, lastmn]) # 6-7
                    if self._is_in_range(plow, phigh, mn5): candis.append([nextmn, mn5]) # 6-5
                    #if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn5]) # 5-5 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 6-ch↓
                elif lastd == 1:
                    # 1 → 2-1/1-1/1-7/1-6/7-7/7-6/6-6/6-5/6-ch↓ → 6
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, lastmn]) # 2-1
                    candis.append([lastmn, lastmn]) # 1-1
                    candis.append([lastmn, mn7]) # 1-7
                    candis.append([lastmn, nextmn]) # 1-6
                    candis.append([mn7, mn7]) # 7-7
                    candis.append([mn7, nextmn]) # 7-6
                    candis.append([nextmn, nextmn]) # 6-6
                    if self._is_in_range(plow, phigh, mn5): candis.append([nextmn, mn5]) # 6-5 
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([nextmn, mnch1]) # 6-ch↓
                elif lastd == 2:
                    # 2 → 2-1/1-1/1-7/1-6/1-ch↓ → 6   |boost1-7
                    candis.append([lastmn, mn1]) # 2-1
                    candis.append([mn1, mn1]) # 1-1
                    candis.append([mn1, mn7])
                    candis.append([mn1, mn7])
                    candis.append([mn1, mn7]) # 1-7
                    candis.append([mn1, nextmn]) # 1-6
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn1, mnch1]) # 1-ch↓
                else:
                    # 6/5/4/3 → 2-1/1-6/1-7/ch↑-ch↑ → 6   |boost1-7
                    candis.append([mn2, mn1]) # 2-1
                    candis.append([mn1, nextmn]) # 1-6
                    candis.append([mn1, mn7])
                    candis.append([mn1, mn7])
                    candis.append([mn1, mn7]) # 1-7
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    mnch2 = self._choose_chord_pitch_near_up(mnch1, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↑-ch↑
            elif nextd == 7:
                mn1 = self._nearest_scale_pitch_up(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, nextmn)
                mn5 = self._nearest_scale_pitch_down(ch, mn6)
                mn2 = self._nearest_scale_pitch_up(ch, mn1)
                mn3 = self._nearest_scale_pitch_up(ch, mn2)
                if lastd == 1:
                    # 1 → 2-1/1-1/1-7/7-6/1-ch↓ → 7
                    if self._is_in_range(plow, phigh, mn2): candis.append([mn2, lastmn]) # 2-1
                    candis.append([lastmn, lastmn]) # 1-1
                    candis.append([lastmn, nextmn]) # 1-7
                    if self._is_in_range(plow, phigh, mn6): candis.append([nextmn, mn6]) # 7-6
                    if self._is_in_range(plow, phigh, mn6): candis.append([nextmn, mn6]) # 7-6
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([lastmn, mnch1]) # 1-ch↓
                elif lastd == 2:
                    # 2 → 2-1/1-1/1-7/1-ch↓ → 7
                    candis.append([lastmn, mn1]) # 2-1
                    candis.append([mn1, mn1]) # 1-1
                    candis.append([mn1, nextmn]) # 1-7
                    mnch1 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                    if self._is_in_range(plow, phigh, mnch1):
                        candis.append([mn1, mnch1]) # 1-ch↓
                elif lastd == 3:
                    # 3 → 2-1/1-1 → 7   |boost2-1
                    candis.append([mn2, mn1])
                    candis.append([mn2, mn1])
                    candis.append([mn2, mn1]) # 2-1 
                    candis.append([mn1, mn1]) # 1-1 
                else:
                    # 7/6/5/4 → 2-1/1-1/ch↑-ch↑ → 7   |boost2-1
                    candis.append([mn2, mn1])
                    candis.append([mn2, mn1])
                    candis.append([mn2, mn1]) # 2-1 
                    candis.append([mn1, mn1]) # 1-1 
                    mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                    mnch2 = self._choose_chord_pitch_near_up(mnch1, ch) #(↑chord)
                    if self._is_in_range(plow, phigh, mnch1) and self._is_in_range(plow, phigh, mnch2):
                        candis.append([mnch2, mnch1]) # ch↑-ch↑
            else:
                raise NotImplementedError(f"Error: unknown next_d={nextd}, nextmn={nextmn}")
        else:
            # lastmn == nextmn
            if nextd == lastd == 1:
                # 1 → 3-2/2-2/2-1/6-7/7-7/7-1/ch↑-1/ch↓-1/1-ch↑/1-ch↓ → 1 
                mn2 = self._nearest_scale_pitch_up(ch, nextmn)
                mn7 = self._nearest_scale_pitch_down(ch, nextmn)
                mn1 = nextmn
                mn6 = self._nearest_scale_pitch_down(ch, mn7)
                mn3 = self._nearest_scale_pitch_up(ch, mn2) 
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn2]) # 3-2
                if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn2]) # 2-2
                if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn1]) # 2-1
                if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn7]) # 6-7
                if self._is_in_range(plow, phigh, mn7): candis.append([mn7, mn7]) # 7-7
                if self._is_in_range(plow, phigh, mn7): candis.append([mn7, mn1]) # 7-1 
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-1
                    candis.append([mnch1, nextmn]) # 1-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-1
                    candis.append([mnch2, nextmn]) # 1-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            elif nextd == lastd == 2:
                # 2 → 3-3/3-2/3-1/2-3/2-1/1-1/1-2/1-3/ch↑-2/ch↓-2/2-ch↑/2-ch↓ → 2 
                mn3 = self._nearest_scale_pitch_up(ch, nextmn)
                mn1 = self._nearest_scale_pitch_down(ch, nextmn)
                mn2 = nextmn
                mn7 = self._nearest_scale_pitch_down(ch, mn1)
                mn4 = self._nearest_scale_pitch_up(ch, mn3) 
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn3]) # 3-3
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn2]) # 3-2
                if self._is_in_range(plow, phigh, mn3) and self._is_in_range(plow, phigh, mn1): candis.append([mn3, mn1]) # 3-1
                if self._is_in_range(plow, phigh, mn3): candis.append([mn2, mn3]) # 2-3
                if self._is_in_range(plow, phigh, mn1): candis.append([mn2, mn1]) # 2-1
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn1]) # 1-1
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn2]) # 1-2 
                if self._is_in_range(plow, phigh, mn1) and self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn3]) # 1-3 
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-2
                    candis.append([mnch1, nextmn]) # 2-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-2
                    candis.append([mnch2, nextmn]) # 2-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            elif nextd == lastd == 3:
                # 3 → 5-5/5-4/4-4/4-3/4-2/3-4/3-2/2-4/2-3/2-2/1-2/ch↑-3/ch↓-3/3-ch↑/3-ch↓ → 3
                mn4 = self._nearest_scale_pitch_up(ch, nextmn)
                mn2 = self._nearest_scale_pitch_down(ch, nextmn)
                mn3 = nextmn
                mn1 = self._nearest_scale_pitch_down(ch, mn2)
                mn5 = self._nearest_scale_pitch_up(ch, mn4) 
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn5]) # 5-5
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn4]) # 5-4
                if self._is_in_range(plow, phigh, mn4): candis.append([mn4, mn4]) # 4-4
                if self._is_in_range(plow, phigh, mn4): candis.append([mn4, mn3]) # 4-3
                if self._is_in_range(plow, phigh, mn4) and self._is_in_range(plow, phigh, mn2): candis.append([mn4, mn2]) # 4-2
                if self._is_in_range(plow, phigh, mn4): candis.append([mn3, mn4]) # 3-4
                if self._is_in_range(plow, phigh, mn2): candis.append([mn3, mn2]) # 3-2
                if self._is_in_range(plow, phigh, mn2) and self._is_in_range(plow, phigh, mn4): candis.append([mn2, mn4]) # 2-4
                if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn3]) # 2-3
                if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn2]) # 2-2
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn2]) # 1-2 
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-3
                    candis.append([mnch1, nextmn]) # 3-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-3
                    candis.append([mnch2, nextmn]) # 3-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            elif nextd == lastd == 4:
                # 4 → 6-5/5-5/5-4/5-3/4-5/4-3/3-5/3-4/3-3/2-3/ch↑-4/ch↓-4/4-ch↑/4-ch↓ → 4
                mn5 = self._nearest_scale_pitch_up(ch, nextmn)
                mn3 = self._nearest_scale_pitch_down(ch, nextmn)
                mn4 = nextmn
                mn2 = self._nearest_scale_pitch_down(ch, mn3)
                mn6 = self._nearest_scale_pitch_up(ch, mn5) 
                if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn5]) # 6-5
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn5]) # 5-5
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn4]) # 5-4
                if self._is_in_range(plow, phigh, mn5) and self._is_in_range(plow, phigh, mn3): candis.append([mn5, mn3]) # 5-3
                if self._is_in_range(plow, phigh, mn5): candis.append([mn4, mn5]) # 4-5
                if self._is_in_range(plow, phigh, mn3): candis.append([mn4, mn3]) # 4-3
                if self._is_in_range(plow, phigh, mn3) and self._is_in_range(plow, phigh, mn5): candis.append([mn3, mn5]) # 3-5
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn4]) # 3-4
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn3]) # 3-3
                if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn3]) # 2-3
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-4
                    candis.append([mnch1, nextmn]) # 4-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-4
                    candis.append([mnch2, nextmn]) # 4-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            elif nextd == lastd == 5:
                # 5 → 6-6/6-5/6-4/5-6/5-3/4-6/3-4/3-3/ch↑-5/ch↓-5/5-ch↑/5-ch↓ → 5
                mn6 = self._nearest_scale_pitch_up(ch, nextmn)
                mn4 = self._nearest_scale_pitch_down(ch, nextmn)
                mn5 = nextmn
                mn3 = self._nearest_scale_pitch_down(ch, mn4)
                mn7 = self._nearest_scale_pitch_up(ch, mn6)  
                if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn6]) # 6-6
                if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn5]) # 6-5
                #if self._is_in_range(plow, phigh, mn6) and self._is_in_range(plow, phigh, mn4): candis.append([mn6, mn4]) # 6-4
                if self._is_in_range(plow, phigh, mn6): candis.append([mn5, mn6]) # 5-6
                if self._is_in_range(plow, phigh, mn3): candis.append([mn5, mn3]) # 5-3
                #if self._is_in_range(plow, phigh, mn6) and self._is_in_range(plow, phigh, mn4): candis.append([mn4, mn6]) # 4-6
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn4]) # 3-4
                if self._is_in_range(plow, phigh, mn3): candis.append([mn3, mn3]) # 3-3
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-5
                    candis.append([mnch1, nextmn]) # 5-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-5
                    candis.append([mnch2, nextmn]) # 5-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            elif nextd == lastd == 6:
                # 6 → 1-1/1-7/1-6/6-5/5-6/5-5/ch↑-6/ch↓-6/6-ch↑/6-ch↓ → 6
                mn7 = self._nearest_scale_pitch_up(ch, nextmn)
                mn5 = self._nearest_scale_pitch_down(ch, nextmn)
                mn6 = nextmn
                mn4 = self._nearest_scale_pitch_down(ch, mn5)
                mn1 = self._nearest_scale_pitch_up(ch, mn7)  
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn1]) # 1-1
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn7]) # 1-7
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn6]) # 1-6
                if self._is_in_range(plow, phigh, mn5): candis.append([mn6, mn5]) # 6-5
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn6]) # 5-6
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn5]) # 5-5
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-6
                    candis.append([mnch1, nextmn]) # 6-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-6
                    candis.append([mnch2, nextmn]) # 6-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            elif nextd == lastd == 7:
                # 7 → 2-1/1-1/1-7/1-6/7-1/7-6/6-1/6-6/5-6/ch↑-6/ch↓-6/6-ch↑/6-ch↓ → 7
                mn1 = self._nearest_scale_pitch_up(ch, nextmn)
                mn6 = self._nearest_scale_pitch_down(ch, nextmn)
                mn7 = nextmn
                mn5 = self._nearest_scale_pitch_down(ch, mn6)
                mn2 = self._nearest_scale_pitch_up(ch, mn1)  
                if self._is_in_range(plow, phigh, mn2): candis.append([mn2, mn1]) # 2-1
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn1]) # 1-1
                if self._is_in_range(plow, phigh, mn1): candis.append([mn1, mn7]) # 1-7
                if self._is_in_range(plow, phigh, mn1) and self._is_in_range(plow, phigh, mn6): candis.append([mn1, mn6]) # 1-6
                if self._is_in_range(plow, phigh, mn1): candis.append([mn7, mn1]) # 7-1
                if self._is_in_range(plow, phigh, mn6): candis.append([mn7, mn6]) # 7-6
                if self._is_in_range(plow, phigh, mn6) and self._is_in_range(plow, phigh, mn1): candis.append([mn6, mn1]) # 6-1
                if self._is_in_range(plow, phigh, mn6): candis.append([mn6, mn6]) # 6-6
                if self._is_in_range(plow, phigh, mn5): candis.append([mn5, mn6]) # 5-6
                mnch1 = self._choose_chord_pitch_near_up(nextmn, ch) #(↑chord)
                mnch2 = self._choose_chord_pitch_near_down(nextmn, ch) #(↓chord)
                if self._is_in_range(plow, phigh, mnch1):
                    candis.append([nextmn, mnch1]) # ch↑-6
                    candis.append([mnch1, nextmn]) # 6-ch↑
                    candis.append([mnch1, mnch1]) # ch↑-ch↑
                if self._is_in_range(plow, phigh, mnch2):
                    candis.append([nextmn, mnch2]) # ch↓-6
                    candis.append([mnch2, nextmn]) # 6-ch↓
                    candis.append([mnch2, mnch2]) # ch↓-ch↓
            else:
                raise NotImplementedError(f"Error: unknown next_d={nextd}, lastd={lastd}")
        # --------- post ---------
        # filter by interval
        filter_candis = []
        for x in candis:
            up = nextmn+max_interval
            down = nextmn-max_interval
            if down <= x[0] <= up and down <= x[1] <= up:
                filter_candis.append(x)
        if len(filter_candis) > 0:
            candis = filter_candis
        # use ref if possible
        if ref_pitches != None and len(ref_pitches) == 2:
            if ref_pitches in candis:
                # in candidate
                sel_pitches = ref_pitches
            else:
                # find nearest
                self.rand.shuffle(candis)
                min_diff = 9999
                nearest_pitches = candis[0]
                for x in candis:
                    diff = abs(x[1] - ref_pitches[1])
                    diff += abs(x[0] - ref_pitches[0])
                    if diff < min_diff:
                        min_diff = diff
                        nearest_pitches = x
                sel_pitches = nearest_pitches
        else:
            # sample
            sel_pitches = self.rand.choice(candis)
        # return
        return sel_pitches


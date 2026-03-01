#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pop_rhythm.py

Rhythm skeleton generation.

Responsibilities:
  - Generate bar-level onset patterns (4/4 grid).
  - Control note density, syncopation, rests, and allowed durations.
  - Provide rhythmic constraints used by melody pitch assignment.

Typical functions:
  - sample_rhythm(cfg, structure, rng) -> RhythmPattern

Author:      <ludy>
Modified:    2026/01/16
Licence:     <MIT>
"""

from expert_melody.pop_settings import Tempo, Key, SectionType, Mode
from expert_melody.pop_settings import RHYTHM_CHARS_RANGE, IS_DEBUG

from dataclasses import dataclass
from typing import List
import random
import math

# P(Phrase2 repeats Phrase1) = 50%
# P(Phrase3 repeats Phrase1) = 50%
# P(Phrase4 repeats Phrase2) = 50%
P_REPEAT_12 = 0.5
P_REPEAT_13 = 0.5
P_REPEAT_24 = 0.5

##-----------------------------
# Rhythm for melody
##-----------------------------
@dataclass
class RhythmPattern:
    pid: int
    durations: List[float]
    is_rests: List[bool]
    pattern_type: str  # "upbeat" / "full" / "unfull" / "ending"

    def nnote(self):
        # number of notes (no rest)
        n = 0
        for b in self.is_rests:
            if not b:
                n += 1
        return n

    def left_dur(self):
        d = 0
        if self.is_rests[-1] == True:
            d = self.durations[-1]
        return d

    def init_rest_dur(self):
        d = 0
        for k, r in enumerate(self.is_rests):
            if r:
                d += self.durations[k]
            else:
                break
        return d

    def __len__(self):
        return len(self.durations)

    def __repr__(self):
        s = f"Dur={self.durations} | is_rests="
        s += str([1 if x else 0 for x in self.is_rests])
        s += f" | {self.pattern_type}"
        return s

    def __eq__(self, pat):
        return self.durations==pat.durations and \
               self.is_rests==pat.is_rests and \
               self.pattern_type==pat.pattern_type

class RhythmGenerator:
    """
    For a given section (num_bars, tempo_class, quant_unit),
    - Sample 4 phrase syllable counts
    - Split each phrase into upbeat/ful/unfull/ending portions (bars & chars)
    - Choose bar-level rhythm patterns for each bar.
    """
    def __init__(self, structure, tempo_class, quant_unit, chord_prog_seq, seed, _log=print):
        self.structure = structure
        self.tempo_class = tempo_class
        self.quant_unit = quant_unit
        self.chord_prog_seq = chord_prog_seq
        self.rand = random.Random(seed)
        self._log = _log

        # generate all patterns
        self.upbeat_patterns = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], }
        self.full_patterns = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], }
        self.ending_patterns = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], }
        self.unfull_patterns = { 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], }

        self._enumerate_rhythm_patterns()

    def _enumerate_rhythm_patterns(self, bar_length=4.0, max_notes=8):
        """
         Enumerate all patterns: upbeat / full / unfull / ending
         Quantize: 8th note
        """
        # Allowed note length
        allowed = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

        patterns = []

        def gen_inner(remaining, cur, res):
            if remaining < -1e-6:
                return
            if abs(remaining) < 1e-6:
                if 1 <= len(cur) <= max_notes:
                    res.append(cur.copy())
                return
            if len(cur) >= max_notes:
                return
            for d in allowed:
                cur.append(d)
                gen_inner(remaining - d, cur, res)
                cur.pop()

        # Calculate all FULL
        inner = []  # List[List[float]]
        gen_inner(bar_length, [], inner)

        # save FULL
        pid = 1
        for seq in inner:
            durations = seq.copy()
            nnote = len(seq)
            is_rests = [False] * nnote

            if abs(sum(durations) - bar_length) > 1e-6:
                continue

            # create pattern
            pat = RhythmPattern(pid, durations, is_rests, "full")

            # save
            self.full_patterns[nnote].append(pat)

            # update
            pid += 1

        def combine_rest(durs, b_rests):
            a = durs.copy()
            b = b_rests.copy()
            n = len(a)
            i = 0
            while i < n-1:
                if b[i] == b[i+1] == True:
                    a[i] += a[i+1]
                    a = a[:i+1] + a[i+2:]
                    b = b[:i+1] + b[i+2:]
                    n = len(a)
                    i = 0
                else:
                    i += 1
            return a, b

        # derive UPBEAT from FULL
        pid = 1
        for seq in inner:
            durations = seq.copy()
            nnote = len(seq)
            is_rests = [False] * nnote

            if abs(sum(durations) - bar_length) > 1e-6:
                continue

            n = nnote
            for k in range(0, nnote-1):
                # make notes rest from head
                is_rests[k] = True
                n -= 1

                durs, b_rests = combine_rest(durations, is_rests)

                # ignore certain pattern
                is_ignore = False 
                for idx in range(len(durs)):
                    if not b_rests[idx] and durs[idx] > 1.0:
                        is_ignore = True 
                        break
                if is_ignore:
                    continue

                # create pattern
                pat = RhythmPattern(pid, durs, b_rests, "upbeat")

                # save
                if pat not in self.upbeat_patterns[n]:
                    self.upbeat_patterns[n].append(pat)
                    # update
                    pid += 1
 
        # derive ENDING from FULL
        pid = 1
        for seq in inner:
            durations = seq.copy()
            nnote = len(seq)
            is_rests = [False] * nnote

            if abs(sum(durations) - bar_length) > 1e-6:
                continue

            n = nnote
            for k in range(nnote-1, 0, -1):
                # make notes rest from end
                is_rests[k] = True
                n -= 1

                durs, b_rests = combine_rest(durations, is_rests)
 
                # Force final note dur → 1.0
                if b_rests[-1] == True:
                    if durs[-2] > 1.0:
                        dis = durs[-2] - 1.0
                        durs[-2] -= dis 
                        durs[-1] += dis
                    elif durs[-2] < 1.0:
                        dis = 1.0 - durs[-2]
                        durs[-2] += dis 
                        durs[-1] -= dis
                        if durs[-1] < 1e-6:
                            durs = durs[:-1]
                            b_rests = b_rests[:-1]
                else:
                    if durs[-1] > 1.0:
                        dis = durs[-1] - 1.0
                        durs[-1] -= dis
                        durs.append(dis)
                        b_rests.append(True)
                    else:
                        pass  

                # For ending, longest note MUST be final note [2026-01-03]
                if b_rests[-1]:
                    tmp_rests = b_rests[:-1]
                    tmp_durs = durs[:-1]
                else:
                    tmp_rests = b_rests[:]
                    tmp_durs = durs[:]
                assert True not in tmp_rests
                max_dur = max(tmp_durs)
                if tmp_durs[-1] < max_dur:
                    continue

                # create pattern
                pat = RhythmPattern(pid, durs, b_rests, "ending")

                # save
                if pat not in self.ending_patterns[n]:
                    self.ending_patterns[n].append(pat)
                    # update
                    pid += 1
 
        # derive UNFULL from FULL
        pid = 1
        for seq in inner:
            durations = seq.copy()
            nnote = len(seq)
            is_rests = [False] * nnote

            if abs(sum(durations) - bar_length) > 1e-6:
                continue

            # make notes rest from head if dur = 0.5/1.0
            if durations[0] in (0.5, 1.0):
                is_rests[0] = True
 
                # create pattern
                pat = RhythmPattern(pid, durations, is_rests, "unfull")

                # save
                if pat not in self.unfull_patterns[nnote-1]:
                    self.unfull_patterns[nnote-1].append(pat)
                    # update
                    pid += 1
        if IS_DEBUG:
            self._log("    ------------------------------------------------------")
            self._log(f"    > Rhythm: all full_patterns = {[len(self.full_patterns[i]) for i in range(1,9)]}")
            self._log(f"    > Rhythm: all upbeat_patterns = {[len(self.upbeat_patterns[i]) for i in range(1,8)]}")
            self._log(f"    > Rhythm: all ending_patterns = {[len(self.ending_patterns[i]) for i in range(1,8)]}")
            self._log(f"    > Rhythm: all unfull_patterns = {[len(self.unfull_patterns[i]) for i in range(1,8)]}")

        return patterns

    def _phrase_bar_in_section(self, num_bar):
        """
        Allocate bar for 4 phrases
        Simple policy：
          - 4 bars: [1,1,1,1]
          - 8 bars: [2,2,2,2]
          - 16 bars: [4,4,4,4]
        """
        if num_bar == 4:
            lens = [1] * 4
        elif num_bar == 8:
            lens = [2] * 4
        elif num_bar == 16:
            lens = [4] * 4 
        else:
            raise NotImplementedError(f"Error: unknown num_bar = {num_bar}")
        return lens

    def sample_phrase_chars(self, phrase_nbars):
        """ Sample syllable count for a phrase according to tempo_class & n_bar. """
        min_c, max_c = RHYTHM_CHARS_RANGE[self.tempo_class][phrase_nbars]

        # assume probability follows a normal distribution
        mu = (min_c + max_c) / 2
        sigma = (max_c - min_c) / 6 + 2

        ws = []
        for x in range(min_c, max_c+1):
            # exp(-(x-μ)²/(2σ²))
            w = math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            ws.append(w)
    
        # choose by weights
        n_chars = self.rand.choices(list(range(min_c, max_c + 1)), weights=ws, k=1)[0]
        
        return n_chars

    def _random_split_int(self, total, k):
        """ Split total_chars into k parts. (each part >= 1) """
        if k <= 0:
            return []
        if total < k:
            return [1] * (total - 1) + [max(1, total - (k-1))]
        cuts = sorted(self.rand.sample(range(1, total), k-1))
        parts = []
        prev = 0
        for c in cuts:
            parts.append(c - prev)
            prev = c
        parts.append(total - prev)

        # 3 part: max=[7,8,7]
        if k == 3:
            while parts[0] > 7:
                parts[0] -= 1
                parts[1] += 1
            while parts[1] > 8:
                parts[1] -= 1
                if parts[0] < 7:
                    parts[0] += 1
                else:
                    parts[2] += 1
            while parts[2] > 7:
                parts[2] -= 1
                if parts[1] < 8:
                    parts[1] += 1
                else:
                    parts[0] += 1 
            # len(part[1]) must > 1
            while parts[1] < 2:
                parts[1] += 1
                if parts[0] > 1:
                    parts[0] -= 1
                else:
                    parts[2] -= 1
        # 2 part: max=[7,7]
        if k == 2:
            while parts[0] > 7:
                parts[0] -= 1
                parts[1] += 1
            while parts[1] > 7:
                parts[1] -= 1
                parts[0] += 1 
            # len(part[0]) must > 1
            while parts[0] < 2:
                parts[0] += 1
                parts[1] -= 1
        # 4 part: max=[7,8,8,7]
        if k == 4:
            while parts[0] > 7:
                parts[0] -= 1
                parts[1] += 1
            while parts[1] > 8:
                parts[1] -= 1
                if parts[0] < 7:
                    parts[0] += 1
                else:
                    parts[2] += 1
            while parts[2] > 8:
                parts[2] -= 1
                if parts[1] < 8:
                    parts[1] += 1
                elif parts[0] < 7:
                    parts[0] += 1
                else:
                    parts[3] += 1 
            while parts[3] > 7:
                parts[3] -= 1
                if parts[2] < 8:
                    parts[2] += 1
                elif parts[1] < 8:
                    parts[1] += 1
                else:
                    parts[0] += 1 
            # len(part[0]) must > 1
            while parts[0] < 2:
                parts[0] += 1
                parts[1] -= 1
            # len(part[1]) must > 1
            while parts[1] < 2:
                parts[1] += 1
                parts[2] -= 1
            # len(part[2]) must > 1
            while parts[2] < 2:
                parts[2] += 1
                if parts[3] > 1:
                    parts[3] -= 1
                elif parts[0] > 2:
                    parts[0] -= 1
                elif parts[1] > 2:
                    parts[1] -= 1
                else:
                    raise NotImplementedError("Error: shouldn't be here.")
        # 5 part: max=[7,8,8,8,7]
        if k == 5:
            while parts[0] > 7:
                parts[0] -= 1
                parts[1] += 1
            while parts[1] > 8:
                parts[1] -= 1
                if parts[0] < 7:
                    parts[0] += 1
                else:
                    parts[2] += 1
            while parts[2] > 8:
                parts[2] -= 1
                if parts[1] < 8:
                    parts[1] += 1
                elif parts[0] < 7:
                    parts[0] += 1
                else:
                    parts[3] += 1 
            while parts[3] > 8:
                parts[3] -= 1
                if parts[2] < 8:
                    parts[2] += 1
                elif parts[1] < 8:
                    parts[1] += 1
                elif parts[0] < 7:
                    parts[0] += 1
                else:
                    parts[4] += 1 
            while parts[4] > 7:
                parts[4] -= 1
                if parts[3] < 8:
                    parts[3] += 1
                elif parts[2] < 8:
                    parts[2] += 1
                elif parts[1] < 8:
                    parts[1] += 1
                else:
                    parts[0] += 1 
            # len(part[1]) must > 1
            while parts[1] < 2:
                parts[1] += 1
                parts[2] -= 1
            # len(part[2]) must > 1
            while parts[2] < 2:
                parts[2] += 1
                parts[3] -= 1
            # len(part[3]) must > 1
            while parts[3] < 2:
                parts[3] += 1
                if parts[4] > 1:
                    parts[4] -= 1
                elif parts[1] > 2:
                    parts[1] -= 1
                elif parts[2] > 2:
                    parts[2] -= 1
                elif parts[0] > 1:
                    parts[0] -= 1
                else:
                    raise NotImplementedError("Error: shouldn't be here.")
        # return
        return parts

    def _assign_rhythm_for_phrase(self, nbar, nchar, allow_upbeat_dur=4.0, ref_bar_pats=None, allow_ending_dur=4.0):
        """ 
            nbar: num bar
            nchar: num char
            allow_upbeat_dur: duration for upbeat
            ref_bar_pats: reference 
            allow_ending_dur: duration for ending
        """ 
        # 1. Combinations of rhythmic components for a phrase
        if nbar == 2:    
            # Allow：up-full-end / full-end / unfull-end 
            if allow_upbeat_dur >= 2.0:
                # up-full-end = 50%
                # full-end = 30%
                # unfull-end = 20%
                candi_pat_types = [ ["upbeat", "full", "ending"], 
                                    ["full", "ending"], 
                                    ["unfull", "ending"]]
                candi_weights = [65, 25, 10] 
            elif allow_upbeat_dur > 0.5:
                # up-full-end = 30%
                # full-end = 50%
                # unfull-end = 20%
                candi_pat_types = [ ["upbeat", "full", "ending"], 
                                    ["full", "ending"], 
                                    ["unfull", "ending"]]
                candi_weights = [50, 35, 15] 
            else:
                # no upbeat
                # full-end = 50%
                # unfull-end = 50%
                candi_pat_types = [ ["full", "ending"], 
                                    ["unfull", "ending"]]
                candi_weights = [80, 20]
        elif nbar == 4:
            # up-full-full-full-end
            # Allow：up-full-full-full-end / full-full-full-end / unfull-full-full-end 
            if allow_upbeat_dur >= 2.0:
                # up-full-end = 50%
                # full-end = 30%
                # unfull-end = 20%
                candi_pat_types = [ ["upbeat", "full", "full", "full", "ending"],
                                    ["full", "full", "full", "ending"], 
                                    ["unfull", "full", "full", "ending"]]
                candi_weights = [65, 25, 10]
            elif allow_upbeat_dur > 0.5:
                # up-full-end = 30%
                # full-end = 50%
                # unfull-end = 20%
                candi_pat_types = [ ["upbeat", "full", "full", "full", "ending"],
                                    ["full", "full", "full", "ending"], 
                                    ["unfull", "full", "full", "ending"]]
                candi_weights = [50, 35, 15]
            else:
                # no upbeat
                # full-end = 50%
                # unfull-end = 50%
                candi_pat_types = [ ["full", "full", "full", "ending"], 
                                    ["unfull", "full", "full", "ending"]]
                candi_weights = [80, 20] 
        else:
            raise NotImplementedError(f"Error: unknown nbar={nbar} while tempo_class={self.tempo}")

        # Use reference if ref exists
        ref_pat_types = [x.pattern_type for x in ref_bar_pats] if ref_bar_pats != None else None

        if ref_pat_types in candi_pat_types:
            #print("select ref pat_types:", ref_pat_types)
            pat_types = ref_pat_types
        else:
            # Sample one if ref not exists
            #print("random pat_types from:", candi_pat_types)
            pat_types = self.rand.choices(candi_pat_types, weights=candi_weights)[0] 

        if IS_DEBUG:
            self._log(f"        > -   pattern_types = {pat_types}")

        # 2. Allocate syllable count for each component (upbeat/full/unfull/ending)
        allow_upbeat_nchar = int(allow_upbeat_dur/0.5)
        allow_ending_nchar = int(allow_ending_dur/0.5) - 0.5 # final note dur=1 for ending
        use_ref = False
        ref_nchars = [x.nnote() for x in ref_bar_pats] if ref_bar_pats != None else None
        #print(" ref_nchars =", ref_nchars)
        if ref_nchars is not None:
            ref_nchar = sum(ref_nchars)
            if ref_nchar > nchar:
                # syllable count in ref > current syllable count
                #print("   ref_nchar > nchar :", ref_nchar, nchar)
                diff_nchar = ref_nchar - nchar
                if ref_pat_types[0] == "upbeat" and ref_nchars[0] >= diff_nchar:
                    # delete syllable from upbeat
                    ref_nchars[0] -= diff_nchar
                    if ref_nchars[0] == 0:
                        # delete upbeat for its nchar=0
                        ref_pat_types = ref_pat_types[1:]
                        ref_nchars = ref_nchars[1:]
                        if pat_types[0] == "upbeat":
                            pat_types = pat_types[1:]
                    # use ref
                    part_nchars = ref_nchars[:]
                    use_ref = True
                elif ref_nchars[0] > diff_nchar:
                    # delete syllable from full/unfull
                    ref_nchars[0] -= diff_nchar
                    # use ref
                    part_nchars = ref_nchars[:]
                    use_ref = True
                #print("   part_nchars :", part_nchars)
            elif ref_nchar < nchar:
                # number of notes in ref < current number of count
                #print("   ref_nchar < nchar :", ref_nchar, nchar, "ref_pat_types =", ref_pat_types)
                diff_nchar = nchar - ref_nchar
                if ref_pat_types[0] == "upbeat":
                    # add syllable from upbeat
                    for d in range(diff_nchar):
                        if ref_nchars[0] < allow_upbeat_nchar:
                            ref_nchars[0] += 1
                        elif ref_nchars[1] < 8:
                            ref_nchars[1] += 1
                        else:
                            if nbar == 2:
                                ref_nchars[2] += 1
                            else:
                                if ref_nchars[2] < 8:
                                    ref_nchars[2] += 1
                                else:
                                    ref_nchars[3] += 1
                    # use ref
                    part_nchars = ref_nchars[:]
                    use_ref = True
                elif ref_pat_types[0] == "unfull":
                    # add syllable from unfull
                    for d in range(diff_nchar):
                        if ref_nchars[0] < 7:
                            ref_nchars[0] += 1
                        else:
                            if nbar == 2:
                                ref_nchars[1] += 1
                            else:
                                if ref_nchars[1] < 8:
                                    ref_nchars[1] += 1
                                else:
                                    ref_nchars[2] += 1
                    # use ref
                    part_nchars = ref_nchars[:]
                    use_ref = True
                else:
                    # add syllable 
                    if ref_nchars[0]+diff_nchar < 8:
                        # add syllable from full
                        ref_nchars[0] += diff_nchar
                        # use ref
                        part_nchars = ref_nchars[:]
                        use_ref = True
                    elif diff_nchar <= allow_upbeat_nchar:
                        # add syllable from upbeat
                        pat_types = ["upbeat",] + pat_types
                        ref_nchars = [diff_nchar,] + ref_nchars
                        # use ref
                        part_nchars = ref_nchars[:]
                        use_ref = True
                    else:
                        # add syllable from upbeat + full 
                        if allow_upbeat_nchar > 0:
                            pat_types = ["upbeat",] + pat_types
                            ref_nchars = [allow_upbeat_nchar,] + ref_nchars
                            full_idx = 1
                        else:
                            full_idx = 0
                        d = diff_nchar - allow_upbeat_nchar
                        while d > 0: 
                            if ref_nchars[full_idx] < 8:
                                ref_nchars[full_idx] += 1
                            else:
                                if nbar == 2:
                                    ref_nchars[full_idx+1] += 1
                                else:
                                    if ref_nchars[full_idx+1] < 8:
                                        ref_nchars[full_idx+1] += 1
                                    else:
                                        ref_nchars[full_idx+2] += 1
                            # update
                            d -= 1
                        # use ref
                        part_nchars = ref_nchars[:]
                        use_ref = True
            else:
                # same syllable count
                part_nchars = ref_nchars[:]
                use_ref = True

        # if not use reference, just random
        if not use_ref:
            npat = len(pat_types)
            part_nchars = self._random_split_int(nchar, npat)
            if IS_DEBUG:
                self._log(f"        > -   nchars by random={part_nchars}") 
        else:
            if IS_DEBUG:
                self._log(f"        > -   nchars by ref={part_nchars}") 

        # 3. Adjust syllable count under constraints
        if nbar == 2:    
            if pat_types[0] == "upbeat":
                while part_nchars[0] > allow_upbeat_nchar: 
                    part_nchars[0] -= 1
                    if part_nchars[1] < 8:
                        part_nchars[1] += 1
                    else:
                        part_nchars[2] += 1
            if pat_types[-1] == "ending":
                while part_nchars[-1] > allow_ending_nchar: 
                    part_nchars[-1] -= 1
                    if pat_types[-2] == "unfull" and part_nchars[-2] < 7:
                        part_nchars[-2] += 1
                    elif pat_types[-2] == "unfull" and part_nchars[-2] >= 7:
                        pat_types[-2] = "full"
                        part_nchars[-2] += 1
                    elif pat_types[-2] == "full" and part_nchars[-2] < 8:
                        part_nchars[-2] += 1
                    else:
                        if len(part_nchars) > 2:
                            part_nchars[-3] += 1
                        else:
                            part_nchars = [1, ] + part_nchars
                            pat_types = ["upbeat", ] + pat_types
            # upbeat + ending <= 7
            if pat_types[0] == "upbeat":
                while part_nchars[0] + part_nchars[-1] > 7:
                    if part_nchars[-1] > 1:
                        part_nchars[-1] -= 1
                        if pat_types[-2] == "unfull" and part_nchars[-2] < 7:
                            part_nchars[-2] += 1
                        elif pat_types[-2] == "unfull" and part_nchars[-2] >= 7:
                            pat_types[-2] = "full"
                            part_nchars[-2] += 1
                        elif pat_types[-2] == "full" and part_nchars[-2] < 8:
                            part_nchars[-2] += 1
                        else:
                            raise NotImplementedError(f"Error: part_nchars = {part_nchars}.")
                    else:
                        part_nchars[0] -= 1
                        if part_nchars[1] < 8:
                            part_nchars[1] += 1
                        else:
                            raise NotImplementedError(f"Error: part_nchars = {part_nchars}.")
        elif nbar == 4:
            # upbeat constraints
            if pat_types[0] == "upbeat":
                while part_nchars[0] > allow_upbeat_nchar: 
                    part_nchars[0] -= 1
                    if part_nchars[1] < 8:
                        part_nchars[1] += 1
                    elif part_nchars[2] < 8:
                        part_nchars[2] += 1
                    elif part_nchars[3] < 8:
                        part_nchars[3] += 1
                    else:
                        part_nchars[4] += 1
            # ending constraints
            if pat_types[-1] == "ending":
                while part_nchars[-1] > allow_ending_nchar: 
                    part_nchars[-1] -= 1
                    if part_nchars[-2] < 8:
                        part_nchars[-2] += 1
                    elif part_nchars[-3] < 8:
                        part_nchars[-3] += 1
                    elif pat_types[-4] == "unfull" and part_nchars[-4] < 7:
                        part_nchars[-4] += 1
                    elif pat_types[-4] == "unfull" and part_nchars[-4] >= 7:
                        pat_types[-4] = "full"
                        part_nchars[-4] += 1
                    elif pat_types[-4] == "full" and part_nchars[-4] < 8:
                        part_nchars[-4] += 1
                    else:
                        if len(part_nchars) > 4:
                            part_nchars[-5] += 1
                        else:
                            part_nchars = [1, ] + part_nchars
                            pat_types = ["upbeat", ] + pat_types
            # upbeat + ending <= 7
            if pat_types[0] == "upbeat":
                while part_nchars[0] + part_nchars[-1] > 7:
                    if part_nchars[-1] > 1:
                        part_nchars[-1] -= 1
                        if part_nchars[-2] < 8:
                            part_nchars[-2] += 1
                        elif part_nchars[-3] < 8:
                            part_nchars[-3] += 1
                        elif pat_types[-4] == "unfull" and part_nchars[-4] < 7:
                            part_nchars[-4] += 1
                        elif pat_types[-4] == "unfull" and part_nchars[-4] >= 7:
                            pat_types[-4] = "full"
                            part_nchars[-4] += 1
                        elif pat_types[-4] == "full" and part_nchars[-4] < 8:
                            part_nchars[-4] += 1
                        else:
                            raise NotImplementedError(f"Error: part_nchars = {part_nchars}.")
                    else:
                        part_nchars[0] -= 1
                        if part_nchars[1] < 8:
                            part_nchars[1] += 1
                        elif part_nchars[2] < 8:
                            part_nchars[2] += 1
                        elif part_nchars[3] < 8:
                            part_nchars[1] += 1
                        elif part_nchars[1] < 8:
                            part_nchars[1] += 1
                        else:
                            raise NotImplementedError(f"Error: part_nchars = {part_nchars}.")
        else:
            raise NotImplementedError(f"Error: unknown nbar={nbar} while tempo_class={self.tempo}")

        if IS_DEBUG:
            self._log(f"        > -   new_nchars={part_nchars}") 
            self._log(f"        > -   new_pat_types={pat_types}") 
        #print(f"        > -   new_nchars={part_nchars}") 
        #print(f"        > -   new_pat_types={pat_types}") 
 
        # Random patterns for each bar
        rhythm_pats = []
        refidx = 0
        pickup_dur = 0
        for k, (ptype, nchar) in enumerate(zip(pat_types, part_nchars)):
            # candidate pattern
            if ptype == "upbeat":
                min_ending_dur = part_nchars[-1]*0.5+0.5
                pat_bank = self._pattern_bank(ptype, nchar, min(allow_upbeat_dur, 4.0-min_ending_dur))
            elif ptype == "ending":
                pat_bank = self._pattern_bank(ptype, nchar, allow_ending_dur=min(allow_ending_dur, 4.0-pickup_dur))
            else:
                pat_bank = self._pattern_bank(ptype, nchar)

            # ref?
            use_ref = False
            if ref_bar_pats != None:
                if ptype != "upbeat" and ref_bar_pats[refidx].pattern_type == "upbeat":
                    refidx += 1
                if ptype == ref_bar_pats[refidx].pattern_type:
                    if nchar == ref_bar_pats[refidx].nnote():
                        # same nchar, use ref pattern
                        if ref_bar_pats[refidx] in pat_bank:
                            a_pat = ref_bar_pats[refidx]
                            use_ref = True
                    # update 
                    refidx += 1

            # No ref → random pattern
            if not use_ref:
                a_pat = self.rand.choice(pat_bank)

            # save pattern
            rhythm_pats.append(a_pat) 
            if ptype == "upbeat":
                pickup_dur = 4.0 - a_pat.init_rest_dur()

        return rhythm_pats

    def _pattern_bank(self, role, nchar, allow_upbeat_dur=4.0, allow_ending_dur=4.0):
        if role == "upbeat":
            bank = self.upbeat_patterns[nchar]
            # upbeat MUST follow allow_upbeat_dur
            filtered_bank = []
            for pat in bank:
                if pat.is_rests[0] and pat.durations[0]+allow_upbeat_dur >= 4.0:
                    filtered_bank.append(pat)
            if len(filtered_bank) > 0:
                bank = filtered_bank
            else:
                raise NotImplementedError(f"Error: No pattern follow allow_upbeat_dur={allow_upbeat_dur} while nchar={nchar}")
        elif role == "full":
            bank = self.full_patterns[nchar]
        elif role == "ending":
            bank = self.ending_patterns[nchar]
            # ending MUST follow allow_ending_dur
            filtered_bank = []
            for pat in bank:
                if 4.0 - pat.left_dur() <= allow_ending_dur:
                    filtered_bank.append(pat)
            if len(filtered_bank) > 0:
                bank = filtered_bank
            else:
                raise NotImplementedError(f"Error: No pattern follow allow_ending_dur={allow_upbeat_dur} & allow_ending_dur={allow_ending_dur} while nchar={nchar}")
        elif role == "unfull":
            bank = self.unfull_patterns[nchar]
        else:
            raise NotImplementedError(f"Error: unknown role={role}")
        return bank


    def gen(self):
        """
        Returns:
          list of dict {
                    "phrase_nchars": [b1,b2,b3,b4],
                    "bar_patterns": [RhythmPattern,...]  # len = num_bars
                       }
        """
        # section info
        section_rhythm_plans = {}   # for temp use
        rhythm_seq = []             # for return

        # ---------------- Loop each section ----------------
        # Only deal with vocal part(verse/pre-chorus/chorus)
        allow_upbeat_dur = 3.0  # duration for pickup(default 3.0)
        
        for i, (sec, chord_prog) in enumerate(zip(self.structure, self.chord_prog_seq)):
            # section info
            stype = sec.section_type
            num_bar = sec.num_bar
            #if self.quant_unit == 0.25:
            #    num_bar *= 2
            assert self.quant_unit == 0.5, f"Error: unknown quant_unit == {self.quant_unit}."
            #print(self.quant_unit, num_bar)
            assert num_bar in (8,16), "Error: num_bar in (8,16), but got {num_bar}."

            # check section type
            if stype in (SectionType.VERSE, SectionType.PRECHORUS, \
                         SectionType.CHORUS): 
                if stype not in section_rhythm_plans:
                    # if this kind of section is not processed yet...
                    if IS_DEBUG:
                        self._log(f"    > Rhythm: {stype.name} section")
                    r1 = self.rand.random()
                    r2 = self.rand.random()
                    r3 = self.rand.random()
                    r4 = self.rand.random()

                    # (1) decide nbars of 4 phrases
                    phrase_nbars = self._phrase_bar_in_section(num_bar)

                    if IS_DEBUG:
                        self._log(f"        > - (1) bars of 4 phrases: {phrase_nbars}")

                    # (2) decide syllable count of 4 phrases
                    phrase_nchars = []
                    for nbar in phrase_nbars:
                        nchars = self.sample_phrase_chars(nbar)
                        phrase_nchars.append(nchars)

                    if IS_DEBUG:
                        self._log(f"        > - (2) syllable count of 4 phrases: {phrase_nchars}")

                    # (3) decide pattern of 4 phrases
                    # patterns of each bar
                    bar_patterns = [[] for k in range(num_bar)]

                    # if chord1/2 and chord2/3 is repeated
                    is_chord_repeat12 = False 
                    is_chord_repeat13 = False 
                    is_chord_repeat24 = False 
                    if num_bar == 8:
                        is_chord_repeat12 = chord_prog[0]==chord_prog[2] 
                        is_chord_repeat13 = chord_prog[0]==chord_prog[4] 
                        is_chord_repeat24 = chord_prog[2]==chord_prog[6] 
                    elif num_bar == 16:
                        is_chord_repeat12 = chord_prog[0]==chord_prog[4] and chord_prog[1]==chord_prog[5]
                        is_chord_repeat13 = chord_prog[0]==chord_prog[8] and chord_prog[1]==chord_prog[9]
                        is_chord_repeat24 = chord_prog[4]==chord_prog[12] and chord_prog[5]==chord_prog[13]

                    # loop each phrase
                    cur_bar = 0
                    allow_ending_dur4 = 4.0  # allow duration for 4th phrase

                    for phrase_idx in range(4):
                        nbar = phrase_nbars[phrase_idx]
                        nchar = phrase_nchars[phrase_idx]

                        if IS_DEBUG:
                            self._log(f"        > - (3-{phrase_idx+1}) nbar={nbar}, nchar={nchar}, allow_upbeat_dur={allow_upbeat_dur}, allow_ending_dur4={allow_ending_dur4}")

                        # Note:
                        # Phrase2 repeat phrase1: near syllable count && (bool_repeat or P_REPEAT_12)
                        # Phrase3 repeat phrase1: near syllable count && (bool_repeat or P_REPEAT_13)
                        # Phrase4 repeat phrase2: near syllable count && (bool_repeat or P_REPEAT_24)
                        ref_pats = ref_bar_pats = None
                        if phrase_idx == 1 and (is_chord_repeat12 and r2 < 0.9 \
                                                or r2 < P_REPEAT_12):
                            nchar0 = phrase_nchars[0]
                            if nchar0-1 <= nchar <= nchar0+1:
                                ref_bar_pats = bar_patterns[:2] if nbar==2 else bar_patterns[:4]
                        elif phrase_idx == 2 and (is_chord_repeat13 and r3 < 0.9 \
                                                  or r3 < P_REPEAT_13):
                            nchar0 = phrase_nchars[0]
                            if nchar0-1 <= nchar <= nchar0+1:
                                ref_bar_pats = bar_patterns[:2] if nbar==2 else bar_patterns[:4]
                        elif phrase_idx == 3 and (is_chord_repeat24 and r3 < 0.9 \
                                                  or r4 < P_REPEAT_24):
                            nchar1 = phrase_nchars[1]
                            if nchar1-1 <= nchar <= nchar1+1:
                                ref_bar_pats = bar_patterns[2:4] if nbar==2 else bar_patterns[4:8]
                        if ref_bar_pats:
                            ref_pats = []
                            for x in ref_bar_pats:
                                ref_pats += x

                        # Assign rhythm patterns for current phrase
                        if phrase_idx < 3:
                            rhythm_pats = self._assign_rhythm_for_phrase(nbar, nchar, allow_upbeat_dur, ref_pats)
                        else:
                            rhythm_pats = self._assign_rhythm_for_phrase(nbar, nchar, allow_upbeat_dur, ref_pats, allow_ending_dur4)
                        
                        # save pattern
                        idx = 0
                        for pat in rhythm_pats:
                            pat_type = pat.pattern_type 
                            if pat_type == "upbeat":
                                bar_patterns[cur_bar].append(pat)
                            else:
                                bar_patterns[cur_bar+idx].append(pat)
                                idx += 1

                        # save initial pickup for ending duration of 4th phrase
                        if phrase_idx == 0:
                            if "upbeat" == bar_patterns[cur_bar][0].pattern_type:
                                allow_ending_dur4 = bar_patterns[cur_bar][0].init_rest_dur()
                        
                        # update
                        cur_bar += nbar
                        allow_upbeat_dur = rhythm_pats[-1].left_dur()

                    # save result
                    section_rhythm_plans[stype] = {
                        "phrase_nchars": phrase_nchars,
                        "bar_patterns": bar_patterns,
                        }
                    # update 
                    allow_upbeat_dur -= 0.5  # longer interval at section end
                    # print("  bar_patterns =", len(bar_patterns))
                    # for x in bar_patterns:
                    #     print(x)
                else:
                    # update
                    left_dur = section_rhythm_plans[stype]["bar_patterns"][-1][-1].left_dur()
                    allow_upbeat_dur = left_dur-0.5  # longer interval at section end

            else:
                # Intro/Outro has no rhythm, enough space for upbeat of next section
                section_rhythm_plans[stype] = {}
                # update
                allow_upbeat_dur = 3

            # save seq
            rhythm_seq.append(section_rhythm_plans[stype])

        # return
        return rhythm_seq

 
    
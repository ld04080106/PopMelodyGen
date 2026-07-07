#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tokenizer.py

Author:      <ludy>
Modified:    2026/06/19
Licence:     <MIT>
"""
import pretty_midi
import numpy as np

class REMITokenizer:
    """REMI (RElated MIDI) Tokenizer"""
    
    def __init__(self):
        """
        Init REMI Tokenizer
        """
        self.token_to_id = {}
        self.id_to_token = []

        # init
        self._create_default_vocab()
        
    def __len__(self):
        return len(self.id_to_token)

    def pad_id(self): 
        return self.token_to_id["<PAD>"]

    def bos_id(self): 
        return self.token_to_id["<BOS>"]

    def eos_id(self): 
        return self.token_to_id["<EOS>"]

    def unk_id(self): 
        return self.token_to_id["<UNK>"]

    def _create_default_vocab(self):
        """ default vocabulary """ 
        self.pitch_min = 33 #35
        self.pitch_max = 68 #66

        # 8th note
        dur_bins = [1,2,3,4,5,6,7,8]
        max_pos = 8
        
        keys = list(range(12))
        
        modes = [0, ]  # Major only in this version
        
        tempos = ["SLOW","MID","FAST"]

        gender = ["MALE", "FEMALE"]
        
        sections = ["INTRO","VERSE1", "PRECHORUS", "CHORUS1",
                            "VERSE2", "CHORUS2", "OUTRO"]
        
        structs = ["A16B16", "A16C8B16", "A8A8B16", "A16A16B16", 
                    "A8A8C8B16", "A16A16C8B16", 
                    "A16B8B8", "A16B16B16",
                    "A16C8B8B8", "A16C8B16B16", 
                    "A8A8B8B8", "A16A16B8B8", "A8A8B16B16", "A16A16B16B16", 
                    "A8A8C8B8B8", "A16A16C8B8B8", "A8A8C8B16B16", "A16A16C8B16B16"]

        chord_names = ['I', 'vi', 'IV', 'V', 'iii', 'ii', 'vi7', 'iii7', 'Vsus4', 'bVII', 'VI', 'III7', 'ii7', 'V7', 'II7', 'III', 'bIII', 'iv', 'v', 'VI7', 
                       'iv6', 'bIIIdim', 'II', 'bVdim', 'bVIIM7', 'IVM7', 'IVdim', 'iv7', 'vii7b5', 'bv7b5', 'ii7b5', 'bVI', 'IIIsus4', 'v7', 'I7',
                       'IM7', 'bvii', 'VII', 'IV7', 'Isus4', 'i', 'vii7', 'VIsus4', 'VIIdim', 'bII', 'bV', 'bVIdim', 'bIIdim', 'VII7', 'bv', 'IIsus4', 'IVsus2', 
                       'bvii7', 'VM7', 'bIIM7', 'bVIM7', 'biii7', 'IIIdim', 'Iaug', 'VIIsus4', 'i7', 'bVII7', 'bvi7b5', ]
        chord_tokens = [f"CH_{x}" for x in chord_names]
  
        # tokens
        toks = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "BAR"] + \
               [f"POS_{i}" for i in range(max_pos)] + \
               [f"DUR_{d}" for d in dur_bins] + \
               [f"PITCH_{p}" for p in range(self.pitch_min, self.pitch_max + 1)] + \
               [f"KEY_{k}" for k in keys] + \
               [f"MODE_{m}" for m in modes] + \
               [f"TEMPO_{t}" for t in tempos] + \
               [f"GENDER_{g}" for g in gender] + \
               [f"SEC_{s}" for s in sections] + \
               [f"STRUCT_{st}" for st in structs] + \
               chord_tokens

        self.token_to_id = {t:i for i,t in enumerate(toks)}
        self.id_to_token = toks
 

    def encode(self, path_midi, meta_dict, midi_pitch_offset=-12):
        """
        path_midi -> midi events -> tokens -> id
        Args:
            path_midi: MIDI path
            meta_dict: meta info of song
            midi_pitch_offset: 
        Returns:
            token list
        """
        tokens = ["<BOS>",]

        # Read midi
        pm = pretty_midi.PrettyMIDI(path_midi)

        # melody track (1st track) 
        inst = None
        for ins in pm.instruments:
            if not ins.is_drum and len(ins.notes) > 0:
                inst = ins
                break
        assert inst != None, f"Error: melody not found in midi file = {path_midi}"

        # prefix
        key = "KEY_{}".format(meta_dict["key"])
        mode = "MODE_{}".format(meta_dict["mode"])
        tempo_class = "TEMPO_{}".format(meta_dict["tempo_class"].upper())
        gender = "GENDER_{}".format(meta_dict["gender"].upper())

        structure = ""
        total_bars = 0
        for struct in meta_dict["structure"]:
            stype = struct["type"]
            nbar = int(struct["num_bars"])
            total_bars += nbar
            if stype.startswith("VERSE"):
                t = "A"
            elif stype.startswith("PRECHORUS"):
                t = "C"
            elif stype.startswith("CHORUS"):
                t = "B"
            else:
                continue  
            structure += f"{t}{nbar}"
        structure = meta_dict.get("structure_token", f"STRUCT_{structure}")

        tokens.append(key)
        tokens.append(mode)
        tokens.append(tempo_class)
        tokens.append(gender)
        tokens.append(structure)

        # downbeats by PrettyMIDI
        downbeats = list(pm.get_downbeats())
        assert len(downbeats) > 2, f"Error: too short downbeats! len(downbeats) = {len(downbeats)}"
        if len(downbeats) <= total_bars:
            bar_lens = np.diff(np.asarray(downbeats, dtype=float))
            bar_len = float(np.median(bar_lens)) if len(bar_lens) else 60.0 / float(meta_dict["bpm"]) * 4.0
            while len(downbeats) <= total_bars:
                downbeats.append(float(downbeats[-1]) + bar_len)

        # Total bars. The STRUCT_* control token is built only from A/B/C
        # sections, but tokenization must cover INTRO/OUTRO too.
        num_bars = total_bars

        # notes sort
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))

        # section of each bar: use for SEC token 
        # chords of each bar(1bar=2chords): use for CHORD token 
        sec_per_bar = [None for x in range(num_bars)]
        chords_per_bar = [None for x in range(num_bars)]
        tmpb = 0
        pre_sec = None
        for struct in meta_dict["structure"]:
            stype = struct["type"]
            nbar = struct["num_bars"]
            chordprog = struct["chords"]

            if stype == "INTRO": 
                sec_name = "INTRO"
            elif stype == "OUTRO":
                sec_name = "OUTRO"
            elif stype.startswith("VERSE"):
                sec_name = "VERSE2" if pre_sec and pre_sec.startswith("VERSE") else "VERSE1"
            elif stype.startswith("PRECHORUS"):
                sec_name = "PRECHORUS"
            elif stype.startswith("CHORUS"):
                sec_name = "CHORUS2" if pre_sec and pre_sec.startswith("CHORUS") else "CHORUS1"
            else:
                raise ValueError(f"Error: unknown stype = {stype}.")

            for k in range(nbar):
                if tmpb+k < num_bars:
                    # section 
                    sec_per_bar[tmpb+k] = sec_name

                    # chords (1bar=2chords)
                    ch = chordprog[k]
                    if '(' in ch or '[' in ch:
                        ch = ch.strip('(').strip(')').strip('[').strip(']')
                        chs = ch.split(',')
                        chords_per_bar[tmpb+k] = ["CH_" + x.strip() for x in chs]
                    elif type(ch) in (tuple, list):
                        chords_per_bar[tmpb+k] = ["CH_"+x for x in ch]
                    else:
                        chords_per_bar[tmpb+k] = [f"CH_{ch}", f"CH_{ch}"]
            # update
            tmpb += nbar
            pre_sec = stype

        # loop each bar (ignore Intro1~7 & Outro1~8, Intro's Bar8 is for pickup of Intro)
        ni = 0  # note index
        for bi in range(7, num_bars):
            bar_start = float(downbeats[bi])
            if bi + 1 < len(downbeats):
                bar_end = float(downbeats[bi + 1])
            elif bi > 0:
                bar_end = bar_start + (float(downbeats[bi]) - float(downbeats[bi - 1]))
            else:
                raise ValueError(f"Error: cannot infer final bar length for midi file = {path_midi}")
            bar_len = bar_end - bar_start

            # SEC token
            cur_sec = sec_per_bar[bi]
            if cur_sec is None or chords_per_bar[bi] is None:
                raise ValueError(
                    f"Error: missing section/chord metadata at bar {bi + 1} "
                    f"for midi file = {path_midi}"
                )
            #tokens.append(self.token_to_id.get(f"SEC_{cur_sec}", self.unk_id))
            tokens.append(f"SEC_{cur_sec}")

            # Bar token
            tokens.append("BAR")

            # chord tokens (1bar = 2chords)
            tokens.extend(chords_per_bar[bi])

            # notes in this bar
            bar_notes = []
            while ni < len(notes) and notes[ni].start < bar_start - 1e-5:
                ni += 1
            while ni < len(notes) and bar_start-1e-5 <= notes[ni].start < bar_end-1e-5:
                bar_notes.append(notes[ni])
                # update
                ni += 1

            # tokens
            for nt in bar_notes:
                pitch = nt.pitch + midi_pitch_offset
                pos = self._quantize_to_grid(nt.start, bar_start, bar_end)
                dur = self._quantize_to_grid(nt.end - nt.start + bar_start, bar_start, bar_end)

                tokens.append(f"PITCH_{pitch}")
                tokens.append(f"POS_{pos}")
                tokens.append(f"DUR_{dur}")

                if pitch < self.pitch_min or pitch > self.pitch_max:
                    print(f"> Warning : pitch({pitch}) out of pitch range({self.pitch_min}~{self.pitch_max})")
                    return False

        # finally
        tokens += ["<EOS>", ]
        
        # token -> id
        ids = [self.token_to_id[x] for x in tokens]
        #print("ids =", ids, len(ids))

        return tokens, ids

    def decode(self, token_ids):
        """ token ID -> MIDIevent """
        events = []
        i = 0
        
        while i < len(token_ids):
            token = self.id_to_token.get(token_ids[i], '[UNK]')
            
            if token.startswith('note_'):
                # get pitch
                pitch = int(token.split('_')[1])
                
                # get vel, dur
                if i + 2 < len(token_ids):
                    velocity_token = self.id_to_token.get(token_ids[i+1], 'velocity_80')
                    duration_token = self.id_to_token.get(token_ids[i+2], 'duration_4')
                    
                    velocity = int(velocity_token.split('_')[1])
                    duration = int(duration_token.split('_')[1])
                    
                    events.append({
                        'type': 'note',
                        'pitch': pitch,
                        'velocity': velocity,
                        'duration': duration
                    })
                    i += 3
                    
            elif token.startswith('chord_'):
                # get chord
                chord_type = token.split('_')[1]
                events.append({
                    'type': 'chord',
                    'chord_type': chord_type
                })
                i += 1
            else:
                # skip
                i += 1
        
        return events

    def _quantize_to_grid(self, time_in_sec, bar_start, bar_end):
        """
        quantize second to [0, GRID_PER_BAR-1]
        assume 4/4 and tempo unchange
        """
        GRID_PER_BAR = 8
        bar_len = max(1e-5, bar_end - bar_start)
        rel = (time_in_sec - bar_start) / bar_len
        pos = int(np.clip(np.round(rel * GRID_PER_BAR)+1e-5, 0, GRID_PER_BAR-1))
        return pos

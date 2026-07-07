#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/midi.py

Author:      <ludy>
Modified:    2026/01/22
Licence:     <MIT>
"""

import pretty_midi

def melody_to_pretty_midi(melody_seq, song_meta, program=0, midi_pitch_offset=0):
    """  Melody → pretty_midi.PrettyMIDI
         Args:
            - melody_seq (list): synthesized melody
            - song_meta (dict): synthesized meta
            - program (int): GM program (0=Piano; 73=Flute; 85=LeadVocal)
            - midi_pitch_offset (int): pitch offset, +12 to adjust melody pitch range from singing-voice to flute
         Returns:
            pretty_midi.
    """
    bpm = song_meta["bpm"]
    quant_unit = song_meta["quant_unit"]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=program)

    second_per_beat = 60.0 / bpm

    beats_per_bar = 4.0     # FIX

    for i, melody in enumerate(melody_seq):
        if melody != None:
            sec_st = song_meta['structure'][i]['bar_index'] * beats_per_bar
            for nt in melody.notes:
                # read note info
                bar_st = nt.bar_idx * beats_per_bar
                st_second = (sec_st + bar_st + nt.start_beat) * second_per_beat
                ed_second = st_second + nt.dur_beat * second_per_beat

                # quant is fixed to 0.5 (8th note), 0.25(16th) is for future use
                # if quant_unit == 0.25:
                #     st_second /= 2
                #     ed_second /= 2

                # save midi note
                inst.notes.append(pretty_midi.Note(
                    velocity=nt.velocity,
                    pitch=nt.pitch + midi_pitch_offset,
                    start=st_second,
                    end=ed_second,
                )) 
    pm.instruments.append(inst)
    return pm

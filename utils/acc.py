#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/acc.py

Author:      <ludy>
Modified:    2026/04/17
Licence:     <MIT>
"""

import json
import pretty_midi
from expert_acc.styles import STYLES_BY_TEMPO
from expert_acc import InstrGroupType

## =========================
# Generate acc
## =========================
# Simple mix params. (default=1.0)
# Drum | Bass | Chord1-instrument | Chord2-instrument
VEL_RATIOES = [0.7, 0.7, 0.5, 0.5  ]

def gen_acc(path_in_melody, path_in_meta, path_out_song, acc_id=0,
            melody_pitch_off=0, melody_program_no=85):
    """
        Generate 1 accompaniment for melody
    """
    # read Meta
    in_meta = json.load(open(path_in_meta, "r", encoding="utf-8"))

    tempo_class = in_meta["tempo_class"]   # eg. 'SLOW'
    bpm = float(in_meta["bpm"])
    key = int(in_meta["key"])
    mode = int(in_meta["mode"])
    quant_unit = in_meta.get("quant_unit", 0.5)
    structure = in_meta["structure"]

    # read melody MIDI
    pm = pretty_midi.PrettyMIDI(path_in_melody)

    # melody track
    instrument = _get_or_create_melody_track(pm, melody_program_no)
    instrument.program = melody_program_no
    instrument.name = "Melody"

    # change pitch
    if melody_pitch_off != 0:
        for note in instrument.notes:
            new_pitch = note.pitch + melody_pitch_off
            new_pitch = max(0, min(new_pitch, 127))
            note.pitch = new_pitch
     
    # pick style of accompaniment
    style = STYLES_BY_TEMPO[tempo_class][acc_id]    # todo

    # gen acc notes
    track_dicts = style.gen(bpm, key, mode, structure)

    # save to midi 
    for i, trk_dict in enumerate(track_dicts):
        instr_no = trk_dict["instr_no"]
        instr_group = trk_dict["instr_group"]
        sec_notes = trk_dict["sec_notes"]
        is_drum = instr_group==InstrGroupType.Drum
        vel_ratio = VEL_RATIOES[i]
        
        _add_acc_track(pm, f"Track{i+1}", instr_no, bpm, sec_notes, quant_unit, is_drum, vel_ratio)

    # write full midi
    pm.write(path_out_song)


def _get_or_create_melody_track(pm, melody_program_no):
    """Return the first non-drum melody track, creating an empty one if needed."""
    for instrument in pm.instruments:
        if not instrument.is_drum:
            return instrument
    instrument = pretty_midi.Instrument(
        program=melody_program_no,
        is_drum=False,
        name="Melody",
    )
    pm.instruments.insert(0, instrument)
    return instrument


def _add_acc_track(pm, name, instr_no, bpm, sec_notes, quant_unit, is_drum=False, vel_ratio=1.0):
    """
        add 1 track
    """
    inst = pretty_midi.Instrument(program=instr_no, is_drum=is_drum, name=name)
    # 64th=3,32th=6,16th=12,8th=24,4th=48,2nd=96,whole=192
    note192_per_beat = 48
    beats_per_bar = 4.0     # FIX
    note192_per_bar = note192_per_beat * beats_per_bar

    note192_per_sec =  float(bpm) / 60. *note192_per_beat

    # loop each bar
    for i, bar_notes in enumerate(sec_notes):
        bar_sec = (i * note192_per_bar) / note192_per_sec

        for nt in bar_notes:
            # read note info
            st = bar_sec + nt.start / note192_per_sec
            ed = st + nt.dur / note192_per_sec

            # quant
            if quant_unit == 0.25:
                st /= 2
                ed /= 2

            # save note
            inst.notes.append(pretty_midi.Note(
                velocity=int(nt.vel * vel_ratio),
                pitch=nt.pitch,
                start=st,
                end=ed,
            ))
    # save track
    pm.instruments.append(inst)

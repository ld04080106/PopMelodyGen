

from enum import IntEnum, Enum, unique
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@unique
class InstrGroupType(IntEnum):
    # 乐器种类
    PianoA = 0      # 大钢琴
    PianoE = 1      # 电钢琴
    Drum = 2        # 鼓(打击乐)
    Bass = 3        # 贝司
    GuitarA = 4     # 木吉他
    GuitarClean = 5     # 电吉他Clean
    GuitarJazz = 6     # 电吉他Jazz
    GuitarDist = 7     # 电吉他Dist
    Percussion = 8  # 敲击乐器
    Organ = 9       # 风琴
    String = 10      # 弦乐
    Ensemble = 11    # 合奏
    Brass = 12      # 铜管
    Reed = 13       # 簧乐器
    Wind = 14       # 木管
    SynLead = 15    # 合成Lead
    SynPad = 16     # 合成Pad
    FX = 17         # FX
    Chinese = 18    # 中国
    Pizzicato = 19    # 拨奏


@dataclass
class InstrumentProfile:
    name_en:str
    name_cn:str
    rng:Tuple[int,int]          # 音域
    min_note:int                # 允许的最短时值（1分=192|2分=96|4分=48|8分=24|16分=12|32分=6|64分=3）
    max_note:int                # 单次持续的最长时值（1分=192|2分=96|4分=48|8分=24|16分=12|32分=6|64分=3）
    sustain_bias:int            # + 越长越好；- 偏短音；0=长短均可
    prefer_above:Optional[bool] # True多走主旋律上方；False下方；None不约束
    breath_every_beats:Optional[int]  # 每 N 拍建议留微休止(管乐/吹奏)
    reattack_gap:float = 0.0    # 分割长音时的微休止长度时值（1分=192|2分=96|4分=48|8分=24|16分=12|32分=6|64分=3）

# 一些常用乐器预设（可自行增删改）
INSTRUMENT_INFO = {
    # 0–7 Pianos
    "1": InstrumentProfile("Acoustic Grand Piano", "大钢琴", (48,78), 6, 192, 0, None, None, 0),
    "2": InstrumentProfile("Bright Acoustic Piano", "亮音大钢琴", (55,85), 6, 192, 0, None, None, 0),
    "3": InstrumentProfile("Electric Grand Piano", "电钢琴", (48,78), 6, 192, 0, True,  12,   0),
    "4": InstrumentProfile("Honky-Tonk Piano", "酒吧钢琴", (48,78), 6, 192, 0, True,  12,   0),
    "5": InstrumentProfile("Rhodes Piano", "练习音钢琴", (55,85), 6, 192, 0, None, None, 0.05),
    "6": InstrumentProfile("Chorused Piano", "合唱加钢琴", (48,78), 6, 192, 0, None, None, 0),
    "7": InstrumentProfile("Harpsichord", "拨弦古钢琴", (48,78), 6, 192, 0, False, None, 0),
    "8": InstrumentProfile("Clavinet", "击弦古钢琴", (55,76), 6, 192, 0, None, None, 0),
    # 8–15 Chromatic Percussion
    "9": InstrumentProfile("Celesta", "钢片琴", (65,87), 6, 192, -1, None, None, 0),
    "10": InstrumentProfile("Glockenspiel", "钟琴", (66,86), 6, 192, -1, None, None, 0),
    "11": InstrumentProfile("Music Box", "八音盒", (60,84), 6, 192, -1, None, None, 0),
    "12": InstrumentProfile("Vibraphone", "电颤琴", (53,90), 6, 192, -1, None, None, 0),
    "13": InstrumentProfile("Marimba", "马林巴", (50,84), 6, 192, -1, None, None, 0),
    "14": InstrumentProfile("Xylophone", "木琴", (68,84), 6, 192, -1, None, None, 0),
    "15": InstrumentProfile("Tubular Bells", "管钟", (66,86), 6, 192, -1, None, None, 0),
    "16": InstrumentProfile("Dulcimer", "洋琴", (57,83), 6, 192, -1, None, None, 0),
    # 16–23 Organs
    "17": InstrumentProfile("Hammond Organ", "击杆风琴", (48,90), 6, 192, 0, None, None, 0),
    "18": InstrumentProfile("Percussive Organ", "打击型风琴", (64,85), 6, 192, 0, None, None, 0),
    "19": InstrumentProfile("Rock Organ", "摇滚风琴", (65,90), 6, 192, 0, None, None, 0),
    "20": InstrumentProfile("Church Organ", "管风琴", (46,86), 6, 192, 0, None, None, 0),
    "21": InstrumentProfile("Reed Organ", "簧风琴", (56,80), 6, 192, 0, None, None, 0),
    "22": InstrumentProfile("Accordion", "手风琴", (48,84), 6, 192, 0, None, None, 0),
    "23": InstrumentProfile("Harmonica", "口琴", (52,86), 6, 192, 0, None, None, 0),
    "24": InstrumentProfile("Tango Accordian", "探戈手风琴", (50,82), 6, 192, 0, None, None, 0),
    # 24–31 Guitars
    "25": InstrumentProfile("Acoustic Guitar (nylon)", "尼龙弦吉他", (45,80), 6, 192, -1, None, None, 3),
    "26": InstrumentProfile("Acoustic Guitar(steel)", "钢弦吉他", (45,80), 6, 192, -1, None, None, 3),
    "27": InstrumentProfile("Electric Guitar (jazz)", "爵士乐电吉他", (45,80), 6, 192, -1, None, None, 3),
    "28": InstrumentProfile("Electric Guitar (clean)", "清音电吉他", (45,80), 6, 192, -1, None, None, 3),
    "29": InstrumentProfile("Electric Guitar (muted)", "弱音电吉他", (45,80), 6, 192, -1, None, None, 3),
    "30": InstrumentProfile("Overdriven Guitar", "驱动音效吉他", (45,80), 6, 192, -1, None, None, 3),
    "31": InstrumentProfile("Distortion Guitar", "失真音效吉他", (45,80), 6, 192, -1, None, None, 3),
    "32": InstrumentProfile("Guitar Harmonics", "吉他泛音", (48,78), 6, 192, -1, None, None, 3),
    # 32–39 Basses
    "33": InstrumentProfile("Acoustic Bass", "原声贝司", (28,48), 6, 192, -1, False, None, 0),
    "34": InstrumentProfile("Electric Bass(finger)", "指拨电贝司", (28,48), 6, 192, -1, False, None, 0),
    "35": InstrumentProfile("Electric Bass(pick)", "拨片拨电贝司", (28,48), 6, 192, -1, False, None, 0),
    "36": InstrumentProfile("Fretless Bass", "无品贝司", (28,48), 6, 192, -1, False, None, 0),
    "37": InstrumentProfile("Slap Bass 1", "击弦贝司1", (28,48), 6, 192, -1, False, None, 0),
    "38": InstrumentProfile("Slap Bass 2", "击弦贝司2", (28,48), 6, 192, -1, False, None, 0),
    "39": InstrumentProfile("Synth Bass 1", "合成贝司1", (28,48), 6, 192, -1, False, None, 0),
    "40": InstrumentProfile("Synth Bass 2", "合成贝司2", (28,48), 6, 192, -1, False, None, 0),
    # 40–47 Strings & Timpani
    "41": InstrumentProfile("Violin", "小提琴", (60,86), 6, 192, 0, True, None, 0),
    "42": InstrumentProfile("Viola", "中提琴", (56,84), 6, 192, 0, None, None, 0),
    "43": InstrumentProfile("Cello", "大提琴", (50,80), 6, 192, 0, None, None, 0),
    "44": InstrumentProfile("Contrabass", "低音提琴", (38,64), 6, 192, 0, None, None, 0),
    "45": InstrumentProfile("Tremolo Strings", "弦乐震音", (44,80), 6, 192, 1, None, None, 0),
    "46": InstrumentProfile("Pizzicato Strings", "弦乐拨奏", (48,78), 6, 192, -1, None, None, 0),
    "47": InstrumentProfile("Orchestral Harp", "竖琴", (48,87), 6, 192, 0, None, None, 0),
    "48": InstrumentProfile("Timpani", "定音鼓", (45,64), 6, 192, 0, None, None, 0), 
    # 48–55 Ensemble/Choir
    "49": InstrumentProfile("String Ensemble 1", "弦乐合奏1", (40,90), 6, 192, 1, None, None, 0), 
    "50": InstrumentProfile("String Ensemble 2", "弦乐合奏2", (40,90), 6, 192, 1, None, None, 0), 
    "51": InstrumentProfile("SynthStrings 1", "合成弦乐1", (42,90), 6, 192, 1, None, None, 0), 
    "52": InstrumentProfile("SynthStrings 2", "合成弦乐2", (44,90), 6, 192, 1, None, None, 0), 
    "53": InstrumentProfile("Choir Aahs", "合唱“啊”音", (55,78), 6, 192, 0, None, None, 0), 
    "54": InstrumentProfile("Voice Oohs", "人声“哦”音", (56,77), 6, 192, 0, None, None, 0), 
    "55": InstrumentProfile("Synth Voice", "合成人声", (60,78), 6, 192, 0, None, None, 0), 
    "56": InstrumentProfile("Orchestra Hit", "乐队打击乐", (60,90), 6, 192, -1, None, None, 0), 
    # 56–63 Brass
    "57": InstrumentProfile("Trumpet", "小号", (57,80), 6, 192, 0, None, 12, 0), 
    "58": InstrumentProfile("Trombone", "长号", (44,71), 6, 192, 0, None, 12, 0), 
    "59": InstrumentProfile("Tuba", "大号", (36,66), 6, 192, 0, None, 12, 0), 
    "60": InstrumentProfile("Muted Trumpet", "弱音小号", (57,80), 6, 192, 0, None, 12, 0), 
    "61": InstrumentProfile("French Horn", "圆号", (42,76), 6, 192, 0, None, 12, 0), 
    "62": InstrumentProfile("Brass Section", "铜管组", (55,80), 6, 192, 0, None, 12, 0), 
    "63": InstrumentProfile("Synth Brass 1", "合成铜管1", (55,80), 6, 192, 0, None, 12, 0), 
    "64": InstrumentProfile("Synth Brass 2", "合成铜管2", (55,80), 6, 192, 0, None, 12, 0), 
    # 64–71 Reed
    "65": InstrumentProfile("Soprano Sax", "高音萨克斯", (60,84), 6, 192, 0, None, 12, 0), 
    "66": InstrumentProfile("Alto Sax", "中音萨克斯", (53,84), 6, 192, 0, None, 12, 0), 
    "67": InstrumentProfile("Tenor Sax", "次中音萨克斯", (50,80), 6, 192, 0, None, 12, 0), 
    "68": InstrumentProfile("Baritone Sax", "上低音萨克斯", (41,78), 6, 192, 0, None, 12, 0), 
    "69": InstrumentProfile("Oboe", "双簧管", (54,80), 6, 192, 0, None, 12, 0), 
    "70": InstrumentProfile("English Horn", "英国管", (54,80), 6, 192, 0, None, 12, 0), 
    "71": InstrumentProfile("Bassoon", "大管", (36,70), 6, 192, 0, None, 12, 0), 
    "72": InstrumentProfile("Clarinet", "单簧管", (54,83), 6, 192, 0, None, 12, 0), 
    # 72–79 Pipe
    "73": InstrumentProfile("Piccolo", "短笛", (61,90), 6, 192, 0, True, 12, 0),
    "74": InstrumentProfile("Flute", "长笛", (58,90), 6, 192, 0, True, 12, 0),
    "75": InstrumentProfile("Recorder", "竖笛", (67,86), 6, 192, 0, True, 12, 0),
    "76": InstrumentProfile("Pan Flute", "排笛", (67,92), 6, 192, 0, True, 12, 0),
    "77": InstrumentProfile("Bottle Blow", "吹瓶口", (58,84), 6, 192, 0, True, 12, 0),
    "78": InstrumentProfile("Skakuhachi", "尺八", (58,79), 6, 192, 0, True, 12, 0),
    "79": InstrumentProfile("Whistle", "哨", (66,94), 6, 192, 0, True, 12, 0),
    "80": InstrumentProfile("Ocarina", "洋埙", (65,92), 6, 192, 0, True, 12, 0),
    # 80–87 Synth Leads
    "81": InstrumentProfile("Lead 1 (Square)", "合成主音1（方波）", (58,90), 6, 192, 0, True, None, 0),
    "82": InstrumentProfile("Lead 2 (sawtooth)", "合成主音2（锯齿波）", (58,90), 6, 192, 0, True, None, 0),
    "83": InstrumentProfile("Lead 3 (calliope lead)", "合成主音3（汽笛风琴）", (58,90), 6, 192, 0, True, None, 0),
    "84": InstrumentProfile("Lead 4 (chiff lead)", "合成主音4 （吹管）", (58,90), 6, 192, 0, True, None, 0),
    "85": InstrumentProfile("Lead 5 (charang)", "合成主音5（吉他）", (58,90), 6, 192, 0, True, None, 0),
    "86": InstrumentProfile("Lead 6 (voice)", "合成主音6（人声）", (58,90), 6, 192, 0, True, None, 0),
    "87": InstrumentProfile("Lead 7 (fifths)", "合成主音7（五度）", (58,90), 6, 192, 0, True, None, 0),
    "88": InstrumentProfile("Lead 8 (bass+lead)", "合成主音8（低音加主音）", (58,90), 6, 192, 0, True, None, 0),
    # 88–95 Pads
    "89": InstrumentProfile("Pad 1 (new age)", "合成柔音1（新时代）", (54,91), 12, 192, 0, None, None, 0),
    "90": InstrumentProfile("Pad 2 (warm)", "合成柔音（暖音）", (54,91), 12, 192, 1, None, None, 0),
    "91": InstrumentProfile("Pad 3 (polysynth)", "合成柔音3（复合成）", (54,91), 12, 192, 1, None, None, 0),
    "92": InstrumentProfile("Pad 4 (choir)", "合成柔音4（合唱）", (54,91), 12, 192, 1, None, None, 0),
    "93": InstrumentProfile("Pad 5 (bowed)", "合成柔音5（弓弦）", (54,91), 12, 192, 1, None, None, 0),
    "94": InstrumentProfile("Pad 6 (metallic)", "合成柔音6（金属）", (54,91), 12, 192, 1, None, None, 0),
    "95": InstrumentProfile("Pad 7 (halo)", "合成柔音7（光环）", (54,91), 12, 192, 1, None, None, 0),
    "96": InstrumentProfile("Pad 8 (sweep)", "合成柔音8（扫弦）", (54,91), 12, 192, 1, None, None, 0),
    # 96–103 Synth FX
    "97": InstrumentProfile("FX 1 (rain)", "合成特效1（雨）", (58,91), 6, 192, 0, None, None, 0),
    "98": InstrumentProfile("FX 2 (soundtrack)", "合成特效2（音轨）", (58,91), 6, 192, 0, None, None, 0),
    "99": InstrumentProfile("FX 3 (crystal)", "合成特效3（水晶）", (58,91), 6, 192, 0, None, None, 0),
    "100": InstrumentProfile("FX 4 (atmosphere)", "合成特效4（大气）", (58,90), 6, 192, 0, None, None, 0),
    "101": InstrumentProfile("FX 5 (brightness)", "合成特效5（亮音）", (58,90), 6, 192, 0, None, None, 0),
    "102": InstrumentProfile("FX 6 (goblins)", "合成特效6（小妖）", (58,90), 6, 192, 0, None, None, 0),
    "103": InstrumentProfile("FX 7 (echoes)", "合成特效7（回声）", (58,90), 6, 192, 0, None, None, 0),
    "104": InstrumentProfile("FX 8 (sci-fi)", "合成特效8（科幻）", (58,90), 6, 192, 0, None, None, 0),
    # 104–111 Ethnic
    "105": InstrumentProfile("Sitar", "锡塔尔", (58,84), 6, 192, 0, None, None, 0),
    "106": InstrumentProfile("Banjo", "班卓", (58,80), 6, 192, 0, None, None, 0),
    "107": InstrumentProfile("Shamisen", "三味线", (55,80), 6, 192, 0, None, None, 0),
    "108": InstrumentProfile("Koto", "筝", (48,82), 6, 192, 0, None, None, 0),
    "109": InstrumentProfile("Kalimba", "卡林巴", (65,84), 6, 192, 0, None, None, 0),
    "110": InstrumentProfile("Bagpipe", "风笛", (60,80), 6, 192, 0, None, None, 0),
    "111": InstrumentProfile("Fiddle", "古提琴", (58,85), 6, 192, 0, None, None, 0),
    "112": InstrumentProfile("Shanai", "唢呐", (58,84), 6, 192, 0, None, None, 0),
    # 112–119 Percussive
    "113": InstrumentProfile("Tinkle Bell", "铃铛", (64,90), 6, 192, -1, None, None, 0),
    "114": InstrumentProfile("Agogo", "拉丁打铃", (59,77), 6, 192, -1, None, None, 0),
    "115": InstrumentProfile("Steel Drums", "钢鼓", (50,80), 6, 192, -1, None, None, 0),
    "116": InstrumentProfile("Woodblock", "木块", (0,0), 6, 192, -1, None, None, 0),
    "117": InstrumentProfile("Taiko Drum", "太鼓", (0,0), 6, 192, -1, None, None, 0),
    "118": InstrumentProfile("Melodic Tom", "嗵鼓", (0,0), 6, 192, -1, None, None, 0),
    "119": InstrumentProfile("Synth Drum", "合成鼓", (0,0), 6, 192, -1, None, None, 0),
    "120": InstrumentProfile("Reverse Cymbal", "镲波形反转", (0,0), 6, 192, -1, None, None, 0),
    # 120–127 Sound Effects
    "121": InstrumentProfile("Guitar Fret Noise", "磨弦声", (0,0), 6, 192, -1, None, None, 0),
    "122": InstrumentProfile("Breath Noise", "呼吸声", (0,0), 6, 192, -1, None, None, 0),
    "123": InstrumentProfile("Seashore", "海浪声", (0,0), 6, 192, -1, None, None, 0),
    "124": InstrumentProfile("Bird Tweet", "鸟鸣声", (0,0), 6, 192, -1, None, None, 0),
    "125": InstrumentProfile("Telephone Ring", "电话铃声", (0,0), 6, 192, -1, None, None, 0),
    "126": InstrumentProfile("Helicopter", "直升机声", (0,0), 6, 192, -1, None, None, 0),
    "127": InstrumentProfile("Applause", "鼓掌声", (0,0), 6, 192, -1, None, None, 0),
    "128": InstrumentProfile("Gunshot", "枪声", (0,0), 6, 192, -1, None, None, 0),
    # 120–127 Chinese
    "129": InstrumentProfile("Erhu", "二胡", (0,0), 6, 192, 0, None, None, 0),
    "130": InstrumentProfile("Guzheng", "古筝", (0,0), 6, 192, 0, None, None, 0),
    "131": InstrumentProfile("Pipa", "琵琶", (0,0), 6, 192, 0, None, None, 0),
    "132": InstrumentProfile("Bangdi", "梆笛", (0,0), 6, 192, 0, None, None, 0),
    "133": InstrumentProfile("Hulusi", "葫芦丝", (0,0), 6, 192, 0, None, None, 0),
    "134": InstrumentProfile("Sheng", "笙", (0,0), 6, 192, 0, None, None, 0),
    "135": InstrumentProfile("Suona", "唢呐", (0,0), 6, 192, 0, None, None, 0),
    "136": InstrumentProfile("Yangqin", "扬琴", (0,0), 6, 192, 0, None, None, 0),
    #…
}
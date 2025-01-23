import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
import math

# --------------------------------------------------------------------
# 一、各项映射函数
# --------------------------------------------------------------------

# ---------------- (A) valence 相关 ---------------- #
def valence_to_mode(valence):
    """
    简单示例：valence > 0.5 认为是大调(major)，否则小调(minor)。
    你可改成更平滑或分段的方式。
    """
    return "major" if valence > 0.5 else "minor"

def valence_pitch_offset(valence):
    """
    例：让 valence 高时音区更高(更明亮)。
    energy=0 -> 不偏移, energy=1 -> +12 半音
    这里是示例，你可根据需要做更大或更小的偏移。
    """
    # 让 valence=0.0 -> 0, valence=1.0 -> +12
    return int(round(12 * valence))

# ---------------- (B) tension 相关 ---------------- #
def slash_factor_by_tension(tension):
    """
    tension 越高 -> 结束符 "/" 概率越低（让旋律更不容易结束）。
    tension=0 -> factor=1.0 (默认不变)
    tension=1 -> factor=0.2 (大幅降低结束概率)
    """
    return 1.0 - 0.8 * tension

def chord_suffix_probs_by_tension(tension):
    """
    返回 7/9/11/13 各自出现的概率(或权重)，根据 tension 做平滑上升。
    示例: tension=0.3 后开始出现7, tension=0.5 后出现9, etc.
    """
    p7  = 1.0 / (1.0 + math.exp(-10*(tension - 0.3)))
    p9  = 1.0 / (1.0 + math.exp(-10*(tension - 0.5)))
    p11 = 1.0 / (1.0 + math.exp(-10*(tension - 0.7)))
    p13 = 1.0 / (1.0 + math.exp(-10*(tension - 0.9)))
    return {"7": p7, "9": p9, "11": p11, "13": p13}

def underscore_offset_by_tension(tension):
    """
    示例：tension 高时稍微减少 '_'，使旋律更紧张（也可不做）。
    tension=0 -> +0.0, tension=1 -> -0.1
    """
    return -0.1 * tension

# ---------------- (C) energy 相关 ---------------- #
def map_energy_to_temperature(energy):
    """
    (5) 采样随机性:
    energy=0 -> 0.8 (较稳定),
    energy=1 -> 1.2 (更随机)
    """
    return 0.8 + 0.4 * energy

def note_velocity(energy):
    """
    (1) 音量/力度:
    energy=0 -> 40 (轻),
    energy=1 -> 110 (强)
    """
    return int(40 + 70 * energy)

def underscore_offset_for_energy(energy):
    """
    (2) 节奏密度: 减少延续符 '_'
    energy=1 -> -0.15,
    使音符更密集
    """
    return -0.15 * energy

def staccato_scale(energy):
    """
    (3) 时值缩放(Staccato):
    energy=0 -> 1.0,
    energy=1 -> 0.6
    """
    return 1.0 - 0.4 * energy

def map_energy_to_step_duration(energy):
    """
    (4) 全局速度: energy=0 -> 0.6s, energy=1 -> 0.3s
    """
    return 0.6 - 0.3 * energy

PRESET_SEEDS = {
    "1": ("五声音阶 (Pentatonic)",    "60 62 64 67 69 _ 69 67 64 62 60 _"),
    "2": ("流行常见动机 (Pop Riff)",  "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"),
    "3": ("带有休止符 (Rest style)",  "60 r 62 _ 62 r 64 _ 64 _ 67 r 67 _"),
    "4": ("小调偏暗 (Minor Dark)",    "57 _ 59 _ 60 59 57 _ r 55 _ 57 59"),
    "5": ("稠密起手式 (Dense start)", "60 60 62 62 64 65 67 69 71 72"),
    "6": ("循环动机 (Loop Motif)",    "60 _ 62 60 r 60 _ 62 60 r"),
    "7": ("和弦分解 (Chord Arp)",     "60 64 67 72 _ 67 64 60 _ 62 65 69 74 _")
}

# --------------------------------------------------------------------
# 二、MelodyGenerator 整合
# --------------------------------------------------------------------

class MelodyGenerator:
    """
    同时结合 valence、tension、energy 进行多维度音乐生成的示例。
    其中:
      - valence: 决定大/小调 & 可选的音区偏移
      - tension: 决定结束符概率、和弦后缀(7/9/11/13)出现率等
      - energy: 决定 velocity、节奏密度、staccato、step_duration、temperature
    """

    def __init__(self, model_path="model.h5", mapping_path="mapping.json", sequence_length=64):
        self.model = keras.models.load_model(model_path)
        with open(mapping_path, "r") as fp:
            self._mappings = json.load(fp)

        self.sequence_length = sequence_length
        self._start_symbols = ["/"] * self.sequence_length

    def _clip_and_norm(self, probs):
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= np.sum(probs)
        return probs

    def _sample_with_temperature(self, probs, temperature):
        probs = np.clip(probs, 1e-8, 1.0)
        logit = np.log(probs) / temperature
        exp_p = np.exp(logit)
        probs = exp_p / np.sum(exp_p)
        return np.random.choice(len(probs), p=probs)

    # =============== 功能和弦示例 (T-S-D) + tension 后缀 =============== #
    def _get_functional_chord(self, step, mode, tension):
        """
        根据 (T, S, D) 循环选择和弦，并根据 tension 来添加 7/9/11/13。
        mode: "major"/"minor" (由 valence 控制)
        tension: 决定后缀概率
        """
        if mode == "major":
            functional_map = {
                "T": ["C", "Am"], 
                "S": ["F", "Dm"],
                "D": ["G", "E7"]
            }
        else:
            # 简化的小调示例
            functional_map = {
                "T": ["Am", "C"],
                "S": ["Dm", "F"],
                "D": ["E", "G"]
            }

        func_prog = ["T", "S", "D", "T"]  # 每 16 步换一个功能
        current_func = func_prog[(step // 16) % len(func_prog)]
        base_chord = np.random.choice(functional_map[current_func])

        # tension -> 后缀概率
        suffix_dict = chord_suffix_probs_by_tension(tension)  # {"7": p7, "9":..., "11":..., "13":...}
        # 构建可选后缀及其概率
        candidates = []
        c_probs = []
        for sfx, val in suffix_dict.items():
            if val > 0.05:  # 概率门槛
                candidates.append(sfx)
                c_probs.append(val)
        if candidates:
            c_probs = np.array(c_probs) / np.sum(c_probs)
            chosen_sfx = np.random.choice(candidates, p=c_probs)
            base_chord += chosen_sfx  # 在原和弦后加 7/9/11/13

        return base_chord

    def get_notes_in_chord(self, chord):
        """
        返回给定和弦包含的 MIDI 音高(字符串)，过滤 mapping 中不存在的音。
        """
        chord_sym = m21.harmony.ChordSymbol(chord)
        allowed = [str(n.midi) for n in chord_sym.pitches]
        allowed = [n for n in allowed if n in self._mappings]
        return allowed

    def apply_chord_constraints(self, probs, allowed_notes, chord_weight=0.4):
        """
        提升和弦内音的概率，非和弦内音则降低，休止符 r、延续符 _ 保留。
        """
        adjusted = np.zeros_like(probs)
        for sym, idx in self._mappings.items():
            if sym in allowed_notes:
                adjusted[idx] += probs[idx] * chord_weight
            elif sym == "r" or sym == "_":
                adjusted[idx] += probs[idx]
            else:
                adjusted[idx] += probs[idx] * (1 - chord_weight)
        if np.sum(adjusted) == 0:
            adjusted = np.ones_like(probs) / len(probs)
        else:
            adjusted /= np.sum(adjusted)
        return adjusted

    # =============== 旋律生成主函数 =============== #
    def generate_melody(self,
                        seed,
                        valence=0.5,
                        tension=0.5,
                        energy=0.5,
                        num_steps=200,
                        min_steps=64):
        """
        综合 valence(调性/音区), tension(和弦复杂度/结束概率), energy(节奏/音量/随机性) 来生成旋律
        """
        # (1) 由 valence 判断大/小调
        mode = valence_to_mode(valence)
        # (2) valence -> 也可控制额外的 pitch 偏移
        valence_offset = valence_pitch_offset(valence)

        # (3) 由 energy -> temperature
        temperature = map_energy_to_temperature(energy)

        seed_list = self._start_symbols + seed.split()
        seed_idx = [self._mappings[s] for s in seed_list]
        melody = []

        for step in range(num_steps):
            x = seed_idx[-self.sequence_length:]
            onehot = keras.utils.to_categorical(x, num_classes=len(self._mappings))
            onehot = onehot[np.newaxis, ...]

            probs = self.model.predict(onehot)[0]

            # (A) tension -> 调整结束符 '/' 概率
            if "/" in self._mappings:
                slash_idx = self._mappings["/"]
                factor = slash_factor_by_tension(tension)  # tension 越大 -> factor 越小
                if step < min_steps:
                    # 前 min_steps 不允许结束
                    probs[slash_idx] = 0.0
                else:
                    probs[slash_idx] *= factor
                probs = self._clip_and_norm(probs)

            # (B) tension -> 稍微减少 '_' (可选，不想要可注释)
            off_ten = underscore_offset_by_tension(tension)  # 例如 -0.1 * tension
            if "_" in self._mappings and off_ten != 0:
                u_idx = self._mappings["_"]
                probs[u_idx] += off_ten
                probs = self._clip_and_norm(probs)

            # (C) energy -> 减少 '_' (进一步调整)
            off_ene = underscore_offset_for_energy(energy)  # -0.15*energy
            if "_" in self._mappings and off_ene != 0:
                u_idx = self._mappings["_"]
                probs[u_idx] += off_ene
                probs = self._clip_and_norm(probs)

            # ============= 功能和声(选和弦) + 应用和弦约束 =============
            current_chord = self._get_functional_chord(step, mode, tension)
            chord_notes = self.get_notes_in_chord(current_chord)
            # tension 有时可允许一些非和弦音, 这里仅示例纯和弦
            probs = self.apply_chord_constraints(probs, chord_notes, chord_weight=0.7)

            # ============= 采样 =============
            out_idx = self._sample_with_temperature(probs, temperature)
            seed_idx.append(out_idx)
            out_symbol = [k for k,v in self._mappings.items() if v==out_idx][0]

            if out_symbol == "/":
                # 如果抽到结束符
                break

            melody.append(out_symbol)

            # ============= energy -> 随机插入短音  =============
            if np.random.rand() < 0.3 * energy:
                short_idx = self._sample_with_temperature(probs, temperature)
                seed_idx.append(short_idx)
                short_sym = [k for k,v in self._mappings.items() if v==short_idx][0]
                if short_sym != "/":
                    melody.append(short_sym)

        # ============= valence -> pitch offset (可选) =============
        # 将 melody 中的数字符号做整体偏移
        final_melody = []
        for sym in melody:
            if sym.isdigit():
                new_pitch = int(sym) + valence_offset
                new_pitch = max(0, min(127, new_pitch))
                final_melody.append(str(new_pitch))
            else:
                final_melody.append(sym)

        return final_melody

    # =============== 保存函数 =============== #
    def save_melody(self, melody, energy=0.5, file_name="output.mid"):
        """
        在保存时，结合 energy 映射到:
          - step_duration (整体速度)
          - staccato_scale (音符时值)
          - note_velocity (力度)
        """
        step_dur = map_energy_to_step_duration(energy)
        staccato_factor = staccato_scale(energy)
        vel_val = note_velocity(energy)

        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1

        i = 0
        while i < len(melody):
            symbol = melody[i]
            if symbol != "_" or i+1 == len(melody):
                if start_symbol is not None:
                    dur = step_dur * step_counter
                    dur *= staccato_factor

                    if start_symbol == "r":
                        event = m21.note.Rest(quarterLength=dur)
                    else:
                        event = m21.note.Note(int(start_symbol), quarterLength=dur)
                        vol = m21.volume.Volume()
                        vol.velocity = vel_val
                        event.volume = vol
                    stream.append(event)
                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter += 1
            i += 1

        stream.write("midi", file_name)
        print(f"Melody saved to {file_name} with energy={energy:.2f}.")

# --------------------------------------------------------------------
# 三、脚本入口: 读取 valence, tension, energy, 生成并保存
# --------------------------------------------------------------------
if __name__ == "__main__":
    mg = MelodyGenerator(model_path="model.h5",
                         mapping_path="mapping.json",
                         sequence_length=64)

    # 1) 显示可选预设种子
    print("Please select a preset seed：")
    for num_key, (style_label, seed_str) in PRESET_SEEDS.items():
        print(f"{num_key}. {style_label} => {seed_str[:30]}...")  
        # 只显示前30字符做参考
    
    # 2) 读取用户输入的编号
    seed_choice = input("Please select a preset seed (1~7): ").strip()
    if seed_choice not in PRESET_SEEDS:
        print("无效输入，将使用默认编号 '1'")
        seed_choice = "1"
    
    # 3) 获取选定的种子
    chosen_label, chosen_seed = PRESET_SEEDS[seed_choice]
    print(f"The seed number you selected: {seed_choice}, style: {chosen_label}")
    print(f"content: {chosen_seed}")

    # 4) 让用户输入 valence/tension/energy
    valence = float(input("Enter valence (0.0 ~ 1.0): "))
    tension = float(input("Enter tension (0.0 ~ 1.0): "))
    energy  = float(input("Enter energy  (0.0 ~ 1.0): "))

    # 5) 调用生成
    melody = mg.generate_melody(
        seed=chosen_seed,
        valence=valence,
        tension=tension,
        energy=energy,
        num_steps=200,
        min_steps=64
    )

    # 6) 保存输出
    output_file = f"mel_val_{valence:.2f}_ten_{tension:.2f}_ene_{energy:.2f}_seed{seed_choice}.mid"
    mg.save_melody(melody, energy=energy, file_name=output_file)
    print(f"Generation complete! Saved to {output_file}")
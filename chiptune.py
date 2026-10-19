import argparse
import copy
import wave
import numpy as np

AUDIO_DTYPE = np.float32
WAV_WRITE_CHUNK_SAMPLES = 1_048_576

# for test

# note
# python chiptune.py --mode note --channel vocal --freq 440 --start 0 --end 1 --output vocal.wav

# chord
# python chiptune.py --mode chord --channel guitar --freqs 261.63,329.63,392.00 --start 0 --end 2 --output guitar_chord.wav

# rapid arp
# python chiptune.py --mode chord --channel guitar --freqs 261.63,329.63,392.00 --start 0 --end 2 --arp-interval 0.04 --output guitar_arp.wav

# drum
# python chiptune.py --channel drums --drum snare --start 0 --end 0.1 --loudness 0.8 --output snare.wav


# Configuration

DEFAULT_CONFIG = {
    "sample_rate": 44100,   # 采样率
    "master_volume": 1.0,   # 总音量
    "bit_depth": 8,         # 位深压缩

    # 包络的最大时长，秒
    # 0或None不限制。
    "max_attack_duration_sec": 0.10,
    "max_decay_duration_sec": 1.00,
    "max_release_duration_sec": 1.00,

    # 输出声道
    "output": {
        "channel_mode": "mono",   # mono / stereo
        "stereo_width": 1.0,      # 0=双单声道，1=标准宽度，2=增强宽度
        "pan_law_db": -3.0,       # 中央声像的每声道衰减
    },

    # RMS响度控制
    "rms_control": {
        "enabled": True,
        "block_ms": 50.0,
        "gate_db": -55.0,
        "track_peak_db": -3.0,
        "track_min_gain_db": -12.0,
        "track_max_gain_db": 12.0,
        "final_target_rms_db": -14.0,
        "final_peak_db": -1.0,
        "final_min_gain_db": -12.0,
        "final_max_gain_db": 12.0,
    },

    # 五个声部通道
    "channels": {
        # 人声
        "vocal": {
            "pan": 0.0,
            "waveform": "square",
            "volume": 0.65,
            "target_rms_db": -18.0,
            "duty": 0.25,
            "attack": 0,
            "decay": 0.3,
            "sustain": 0.70,
            "release": 0.3,
            # 交叉淡化
            "event_crossfade_sec": 0.012,
            "event_crossfade_max_gap_sec": 0.060,
        },

        # 吉他
        "guitar": {
            "pan": -0.45,
            "waveform": "square",
            "volume": 0.55,
            "target_rms_db": -20.0,
            "duty": 0.125,
            "attack": 0.002,
            "decay": 0.3,
            "sustain": 0.35,
            "release": 0.7,
            "arpeggio_interval": 0.0,
        },

        # 钢琴
        "piano": {
            "pan": 0.35,
            "waveform": "square",
            "volume": 0.55,
            "target_rms_db": -20.0,
            "duty": 0.25,
            "attack": 0.001,
            "decay": 0.1,
            "sustain": 0.30,
            "release": 0.7,
            "arpeggio_interval": 0.0,
        },

        # 贝斯
        "bass": {
            "pan": 0.0,
            "waveform": "triangle",
            "volume": 0.75,
            "target_rms_db": -19.0,
            "attack": 0.002,
            "decay": 0.5,
            "sustain": 0.5,
            "release": 0.5,
            "steps": 16,
        },

        # 其他乐器
        "other": {
            "pan": 0.55,
            "waveform": "square",
            "volume": 0.45,
            "target_rms_db": -22.0,
            "duty": 0.125,
            "attack": 0.002,
            "decay": 0.04,
            "sustain": 0.40,
            "release": 0.05,
            # 交叉淡化
            "event_crossfade_sec": 0.012,
            "event_crossfade_max_gap_sec": 0.060,
        },
    },

    # 鼓组音色
    "drums": {
        # 底鼓
        "kick": {
            "pan": 0.0,
            "volume": 0.95,
            "target_rms_db": -17.0,

            "start_freq": 140.0,
            "end_freq": 45.0,
            "pitch_curve": 0.35,

            "triangle_mix": 0.90,
            "square_mix": 0.20,
            "noise_mix": 0.04,

            "duty": 0.50,

            "noise_filter": "lowpass",
            "noise_low": None,
            "noise_high": 350.0,


            "attack": 0.00,
            "decay": 0.70,
            "sustain": 0.00,
            "release": 0.30,
        },

        # 军鼓
        "snare": {
            "pan": 0.0,
            "volume": 0.5,
            "target_rms_db": -18.0,

            "start_freq": 170.0,
            "end_freq": 105.0,
            "pitch_curve": 0.55,

            "triangle_mix": 0.1,
            "square_mix": 0.1,
            "noise_mix": 0.9,

            "duty": 0.50,

            "noise_filter": "bandpass",
            "noise_low": 200.0,
            "noise_high": 15000.0,

            "attack": 0.00,
            "decay": 0.10,
            "sustain": 0.10,
            "release": 0.30,
        },

        # 闭镲
        "hh": {
            "pan": 0.45,
            "volume": 0.45,
            "target_rms_db": -24.0,

            "start_freq": 8000.0,
            "end_freq": 8000.0,
            "pitch_curve": 1.00,

            "triangle_mix": 0.00,
            "square_mix": 0.05,
            "noise_mix": 1.00,

            "duty": 0.25,

            "noise_filter": "highpass",
            "noise_low": 5500.0,
            "noise_high": None,

            "attack": 0.05,
            "decay": 0.85,
            "sustain": 0.1,
            "release": 0.1,
        },

        # 镲片
        "cymbals": {
            "pan": -0.35,
            "volume": 0.38,
            "target_rms_db": -25.0,

            "start_freq": 2000.0,
            "end_freq": 7000.0,
            "pitch_curve": 1.00,

            "triangle_mix": 0.00,
            "square_mix": 0.04,
            "noise_mix": 1.00,

            "duty": 0.25,

            "noise_filter": "highpass",
            "noise_low": 2000.0,
            "noise_high": None,

            "attack": 0.00,
            "decay": 0.35,
            "sustain": 0.12,
            "release": 0.65,
        },

        # 嗵鼓
        "toms": {
            "pan": 0.20,
            "volume": 0.75,
            "target_rms_db": -19.0,

            "start_freq": 190.0,
            "end_freq": 75.0,
            "pitch_curve": 0.55,

            "triangle_mix": 0.75,
            "square_mix": 0.35,
            "noise_mix": 0.12,

            "duty": 0.50,

            "noise_filter": "lowpass",
            "noise_low": None,
            "noise_high": 1200.0,

            "attack": 0.00,
            "decay": 0.20,
            "sustain": 0.50,
            "release": 0.80,
        },
    },
}

MELODY_CHANNEL_NAMES = tuple(DEFAULT_CONFIG["channels"].keys())
CHANNEL_NAMES = MELODY_CHANNEL_NAMES + ("drums",)
DRUM_NAMES = tuple(DEFAULT_CONFIG["drums"].keys())


# 处理配置

# 递归更新配置
def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value

    return base

# 生成合成配置
def make_config(overrides: dict | None = None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)

    if overrides is not None:
        deep_update(config, overrides)

    return config

# 获取旋律通道参数
def get_channel_params(channel: str, config: dict) -> dict:
    if channel not in config["channels"]:
        raise ValueError(
            "旋律声部 channel 只能是 vocal、guitar、piano、bass 或 other"
        )

    return config["channels"][channel]

# 获取鼓组参数
def get_drum_params(drum: str, config: dict) -> dict:
    if drum not in config["drums"]:
        raise ValueError(
            "drum 只能是 kick、snare、hh、cymbals 或 toms"
        )

    return config["drums"][drum]


# 获取输出参数
def get_output_params(config: dict) -> dict:
    params = config.get("output", {}) or {}
    channel_mode = str(params.get("channel_mode", "mono")).strip().lower()
    if channel_mode not in {"mono", "stereo"}:
        raise ValueError("channel_mode 只能是 mono 或 stereo")

    stereo_width = float(params.get("stereo_width", 1.0))
    if not 0.0 <= stereo_width <= 2.0:
        raise ValueError("stereo_width 必须在 0 到 2 之间")

    pan_law_db = float(params.get("pan_law_db", -3.0))
    if not -12.0 <= pan_law_db <= 0.0:
        raise ValueError("pan_law_db 必须在 -12 到 0 dB 之间")

    return {
        "channel_mode": channel_mode,
        "stereo_width": stereo_width,
        "pan_law_db": pan_law_db,
    }


# 检查声像参数
def validate_pan(pan: float) -> float:
    pan = float(pan)
    if not -1.0 <= pan <= 1.0:
        raise ValueError("pan 必须在 -1（左）到 1（右）之间")
    return pan


# 计算立体声声像增益
def stereo_pan_gains(pan: float, pan_law_db: float = -3.0) -> tuple[float, float]:
    """返回可调中心衰减的声像增益。极左/极右为 1/0，中央为 pan_law_db。"""
    pan = validate_pan(pan)
    center_gain = 10.0 ** (float(pan_law_db) / 20.0)

    if pan <= 0.0:
        left = center_gain + (-pan) * (1.0 - center_gain)
        right = center_gain * (1.0 + pan)
    else:
        left = center_gain * (1.0 - pan)
        right = center_gain + pan * (1.0 - center_gain)

    return float(left), float(right)


# 将单声道转换为立体声
def pan_mono_to_stereo(
    audio: np.ndarray,
    pan: float,
    pan_law_db: float = -3.0,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=AUDIO_DTYPE)
    if audio.ndim != 1:
        raise ValueError("pan_mono_to_stereo 只接受一维单声道音频")

    left_gain, right_gain = stereo_pan_gains(pan, pan_law_db)
    stereo = np.empty((len(audio), 2), dtype=AUDIO_DTYPE)
    stereo[:, 0] = audio * left_gain
    stereo[:, 1] = audio * right_gain
    return stereo


# 调整立体声宽度
def apply_stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    audio = np.asarray(audio, dtype=AUDIO_DTYPE)
    width = float(width)
    if not 0.0 <= width <= 2.0:
        raise ValueError("stereo_width 必须在 0 到 2 之间")
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("apply_stereo_width 只接受 shape=(samples, 2) 的双声道音频")

    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * (0.5 * width)
    result = np.empty_like(audio)
    result[:, 0] = mid + side
    result[:, 1] = mid - side
    return result


# 整理输出音频格式
def format_output_audio(audio: np.ndarray, config: dict, pan: float = 0.0) -> np.ndarray:
    params = get_output_params(config)
    audio = np.asarray(audio, dtype=AUDIO_DTYPE)
    if params["channel_mode"] == "mono":
        return audio

    stereo = pan_mono_to_stereo(
        audio,
        pan=pan,
        pan_law_db=params["pan_law_db"],
    )
    return apply_stereo_width(stereo, params["stereo_width"])


# 保存WAV音频
def save_wav(
    filename: str,
    audio: np.ndarray,
    sr: int = 44100,
    chunk_size: int = WAV_WRITE_CHUNK_SAMPLES,
):
    audio = np.asarray(audio)
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2 and audio.shape[1] in {1, 2}:
        channels = int(audio.shape[1])
    else:
        raise ValueError("audio 必须是一维单声道，或 shape=(samples, 1/2) 的音频")

    with wave.open(filename, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)

        for start in range(0, len(audio), chunk_size):
            chunk = np.asarray(audio[start:start + chunk_size], dtype=AUDIO_DTYPE)
            chunk = np.clip(chunk, -1.0, 1.0)
            audio_int16 = (chunk * 32767).astype(np.int16)
            wf.writeframes(audio_int16.reshape(-1).tobytes())


# 芯片音乐信号生成

# 检查包络参数
def validate_envelope_params(
    attack: float,
    decay: float,
    sustain: float,
    release: float,
):
    if attack < 0:
        raise ValueError("attack 不能小于 0")

    if decay < 0:
        raise ValueError("decay 不能小于 0")

    if release < 0:
        raise ValueError("release 不能小于 0")

    if sustain < 0:
        raise ValueError("sustain 不能小于 0")

    if sustain > 1:
        raise ValueError("sustain 不能大于 1")

# 限制包络阶段采样数
def _limit_envelope_stage_samples(
    samples: int,
    max_duration_sec: float | None,
    sr: int,
    param_name: str,
) -> int:
    samples = max(0, int(samples))
    if max_duration_sec is None:
        return samples

    max_duration_sec = float(max_duration_sec)
    if not np.isfinite(max_duration_sec) or max_duration_sec < 0:
        raise ValueError(f"{param_name} 必须是大于等于 0 的有限数值；0 表示不限制")
    if max_duration_sec == 0:
        return samples

    limit_samples = max(1, int(round(max_duration_sec * sr)))
    return min(samples, limit_samples)


# 生成ADSR包络
def adsr_envelope(
    duration: float,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sr: int,
    max_attack_duration_sec: float | None = None,
    max_decay_duration_sec: float | None = None,
    max_release_duration_sec: float | None = None,
) -> np.ndarray:
    validate_envelope_params(
        attack=attack,
        decay=decay,
        sustain=sustain,
        release=release,
    )

    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    n = int(duration * sr)

    if n <= 0:
        return np.zeros(0, dtype=AUDIO_DTYPE)

    a = _limit_envelope_stage_samples(
        int(attack * n),
        max_attack_duration_sec,
        sr,
        "max_attack_duration_sec",
    )
    d = _limit_envelope_stage_samples(
        int(decay * n),
        max_decay_duration_sec,
        sr,
        "max_decay_duration_sec",
    )
    r = _limit_envelope_stage_samples(
        int(release * n),
        max_release_duration_sec,
        sr,
        "max_release_duration_sec",
    )

    # 防止ADR超出1
    adr_total = a + d + r
    if adr_total > n:
        scale = n / max(adr_total, 1)
        scaled = [a * scale, d * scale, r * scale]
        counts = [int(value) for value in scaled]
        remaining = n - sum(counts)
        order = sorted(
            range(3),
            key=lambda index: scaled[index] - counts[index],
            reverse=True,
        )
        for index in order[:remaining]:
            counts[index] += 1
        a, d, r = counts

    s = n - a - d - r

    parts = []

    # Attack
    if a > 0:
        parts.append(
            np.linspace(
                0.0,
                1.0,
                a,
                endpoint=False,
                dtype=AUDIO_DTYPE,
            )
        )

    # Decay
    if d > 0:
        parts.append(
            np.linspace(
                1.0,
                sustain,
                d,
                endpoint=False,
                dtype=AUDIO_DTYPE,
            )
        )

    # Sustain
    if s > 0:
        parts.append(
            np.full(
                s,
                sustain,
                dtype=AUDIO_DTYPE,
            )
        )

    # Release
    if r > 0:
        parts.append(
            np.linspace(
                sustain,
                0.0,
                r,
                endpoint=True,
                dtype=AUDIO_DTYPE,
            )
        )

    env = np.concatenate(parts) if parts else np.zeros(n, dtype=AUDIO_DTYPE)

    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    else:
        env = env[:n]

    return env

# 归一化相位
def normalize_phase(phase: float) -> float:
    return float(phase) % 1.0


# 计算采样后的相位
def phase_after_samples(
    initial_phase: float,
    freq: float,
    n_samples: int,
    sr: int,
) -> float:
    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    return normalize_phase(
        float(initial_phase) + float(freq) * max(0, int(n_samples)) / float(sr)
    )


# 生成连续相位
def make_phase_by_samples(
    freq: float,
    n_samples: int,
    sr: int,
    initial_phase: float = 0.0,
) -> np.ndarray:
    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    n_samples = max(0, int(n_samples))
    sample_offsets = np.arange(n_samples, dtype=np.float64)
    phase = (
        normalize_phase(initial_phase)
        + float(freq) * sample_offsets / float(sr)
    ) % 1.0

    return phase.astype(AUDIO_DTYPE, copy=False)

# 生成频率曲线
def make_frequency_curve(
    start_freq: float,
    end_freq: float,
    n_samples: int,
    curve: float,
) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros(0, dtype=AUDIO_DTYPE)

    start_freq = max(float(start_freq), 1e-8)
    end_freq = max(float(end_freq), 1e-8)
    curve = max(float(curve), 1e-8)

    x = np.linspace(0.0, 1.0, n_samples, dtype=AUDIO_DTYPE)

    if abs(start_freq - end_freq) < 1e-8:
        return np.full(n_samples, start_freq, dtype=AUDIO_DTYPE)

    shaped_x = x ** curve
    freqs = start_freq * ((end_freq / start_freq) ** shaped_x)

    return freqs

# 根据频率曲线生成相位
def make_phase_by_frequency_curve(
    freqs: np.ndarray,
    sr: int,
    initial_phase: float = 0.0,
) -> np.ndarray:
    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.size == 0:
        return np.zeros(0, dtype=AUDIO_DTYPE)

    increments = np.empty_like(freqs, dtype=np.float64)
    increments[0] = 0.0
    if freqs.size > 1:
        increments[1:] = np.cumsum(freqs[:-1], dtype=np.float64) / float(sr)

    phase = (normalize_phase(initial_phase) + increments) % 1.0
    return phase.astype(AUDIO_DTYPE, copy=False)

# 执行位深压缩
def bit_crush(audio: np.ndarray, bit_depth: int) -> np.ndarray:
    levels = 2 ** bit_depth
    return np.round(audio * (levels / 2)) / (levels / 2)


# 原地执行位深压缩
def bit_crush_inplace(audio: np.ndarray, bit_depth: int) -> np.ndarray:
    levels = 2 ** bit_depth
    factor = levels / 2
    np.multiply(audio, factor, out=audio)
    np.rint(audio, out=audio)
    np.divide(audio, factor, out=audio)
    return audio


# 计算峰值振幅
def peak_amplitude(
    audio: np.ndarray,
    chunk_size: int = WAV_WRITE_CHUNK_SAMPLES,
) -> float:
    if len(audio) == 0:
        return 0.0

    peak = 0.0
    for start in range(0, len(audio), chunk_size):
        chunk = np.asarray(audio[start:start + chunk_size])
        if len(chunk) == 0:
            continue
        chunk_peak = float(np.max(np.abs(chunk)))
        if chunk_peak > peak:
            peak = chunk_peak

    return peak


# 原地缩放音频
def scale_audio_inplace(
    audio: np.ndarray,
    scale: float,
    chunk_size: int = WAV_WRITE_CHUNK_SAMPLES,
) -> np.ndarray:
    if len(audio) == 0 or scale == 1.0:
        return audio

    for start in range(0, len(audio), chunk_size):
        audio[start:start + chunk_size] *= scale

    return audio


# 将分贝转换为振幅
def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


# 将振幅转换为分贝
def amplitude_to_db(amplitude: float, floor_db: float = -120.0) -> float:
    amplitude = abs(float(amplitude))
    if amplitude <= 1e-12:
        return float(floor_db)
    return max(float(floor_db), 20.0 * float(np.log10(amplitude)))


# 计算RMS振幅
def rms_amplitude(audio: np.ndarray) -> float:
    audio = np.asarray(audio)
    if audio.size == 0:
        return 0.0

    total_energy = 0.0
    total_samples = 0
    for start in range(0, len(audio), WAV_WRITE_CHUNK_SAMPLES):
        chunk = np.asarray(audio[start:start + WAV_WRITE_CHUNK_SAMPLES], dtype=np.float64)
        if chunk.size == 0:
            continue
        total_energy += float(np.dot(chunk, chunk))
        total_samples += int(chunk.size)

    if total_samples <= 0:
        return 0.0
    return float(np.sqrt(total_energy / total_samples))


# 计算门控RMS振幅
def gated_rms_amplitude(
    audio: np.ndarray,
    sr: int,
    block_ms: float = 50.0,
    gate_db: float = -55.0,
) -> float:
    audio = np.asarray(audio)
    if audio.size == 0:
        return 0.0
    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    block_samples = max(1, int(round(float(sr) * float(block_ms) / 1000.0)))
    gate_amplitude = db_to_amplitude(gate_db)
    active_energy = 0.0
    active_samples = 0

    for start in range(0, len(audio), block_samples):
        block = np.asarray(audio[start:start + block_samples], dtype=np.float64)
        if block.size == 0:
            continue
        energy = float(np.dot(block, block))
        block_rms = float(np.sqrt(energy / block.size))
        if block_rms >= gate_amplitude:
            active_energy += energy
            active_samples += int(block.size)

    if active_samples <= 0:
        return rms_amplitude(audio)
    return float(np.sqrt(active_energy / active_samples))


# 计算RMS控制增益
def compute_rms_control_gain(
    audio: np.ndarray,
    sr: int,
    target_rms_db: float,
    settings: dict | None = None,
) -> tuple[float, dict]:
    settings = settings or {}
    current_rms = gated_rms_amplitude(
        audio=audio,
        sr=sr,
        block_ms=float(settings.get("block_ms", 50.0)),
        gate_db=float(settings.get("gate_db", -55.0)),
    )
    current_peak = peak_amplitude(audio)

    stats = {
        "current_rms": current_rms,
        "current_rms_db": amplitude_to_db(current_rms),
        "current_peak": current_peak,
        "current_peak_db": amplitude_to_db(current_peak),
        "target_rms_db": float(target_rms_db),
        "gain": 1.0,
        "gain_db": 0.0,
        "peak_limited": False,
    }

    if current_rms <= 1e-10 or current_peak <= 1e-10:
        return 1.0, stats

    requested_gain_db = float(target_rms_db) - stats["current_rms_db"]
    gain_db = max(
        float(settings.get("track_min_gain_db", -12.0)),
        min(float(settings.get("track_max_gain_db", 12.0)), requested_gain_db),
    )
    gain = db_to_amplitude(gain_db)

    peak_limit = db_to_amplitude(float(settings.get("track_peak_db", -3.0)))
    max_peak_gain = peak_limit / current_peak
    if gain > max_peak_gain:
        gain = max(0.0, max_peak_gain)
        gain_db = amplitude_to_db(gain)
        stats["peak_limited"] = True

    stats["gain"] = gain
    stats["gain_db"] = gain_db
    stats["result_rms_db"] = amplitude_to_db(current_rms * gain)
    stats["result_peak_db"] = amplitude_to_db(current_peak * gain)
    return gain, stats


# 完成单轨音频处理
def finalize_track_audio(
    audio: np.ndarray,
    config: dict,
    target_rms_db: float | None,
    *,
    return_stats: bool = False,
):
    audio = np.asarray(audio, dtype=AUDIO_DTYPE)

    master_volume = float(config.get("master_volume", 1.0))
    if master_volume != 1.0:
        scale_audio_inplace(audio, master_volume)

    settings = config.get("rms_control", {}) or {}
    stats = None
    if bool(settings.get("enabled", True)) and target_rms_db is not None:
        gain, stats = compute_rms_control_gain(
            audio=audio,
            sr=int(config["sample_rate"]),
            target_rms_db=float(target_rms_db),
            settings=settings,
        )
        scale_audio_inplace(audio, gain)
    else:
        audio = normalize_audio(audio)


    bit_depth = config.get("bit_depth")
    if bit_depth is not None:
        bit_crush_inplace(audio, int(bit_depth))


    if stats is not None:
        final_rms = gated_rms_amplitude(
            audio=audio,
            sr=int(config["sample_rate"]),
            block_ms=float(settings.get("block_ms", 50.0)),
            gate_db=float(settings.get("gate_db", -55.0)),
        )
        final_peak = peak_amplitude(audio)
        stats["bit_depth"] = bit_depth
        stats["result_rms_db"] = amplitude_to_db(final_rms)
        stats["result_peak_db"] = amplitude_to_db(final_peak)

    if return_stats:
        return audio, stats
    return audio

# 归一化音频
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio

    max_amp = peak_amplitude(audio)

    if max_amp > 1.0:
        audio = scale_audio_inplace(audio, 1.0 / max_amp)

    return audio

# 执行峰值归一化
def peak_normalize(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio

    max_amp = peak_amplitude(audio)

    if max_amp <= 1e-8:
        return audio

    return scale_audio_inplace(audio, 1.0 / max_amp)

# 完成最终音频处理
def finalize_audio(audio: np.ndarray, config: dict) -> np.ndarray:
    audio = np.asarray(audio, dtype=AUDIO_DTYPE)

    master_volume = float(config["master_volume"])
    if master_volume != 1.0:
        scale_audio_inplace(audio, master_volume)

    audio = normalize_audio(audio)

    if config["bit_depth"] is not None:
        bit_crush_inplace(audio, int(config["bit_depth"]))

    return audio


# 将音频放入时间轴
def place_audio_on_timeline(
    event_audio: np.ndarray,
    start_time: float,
    end_time: float,
    sr: int,
) -> np.ndarray:
    if start_time < 0:
        raise ValueError("start_time 不能小于 0")

    if end_time <= start_time:
        raise ValueError("end_time 必须大于 start_time")

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    total_samples = end_sample

    timeline = np.zeros(total_samples, dtype=AUDIO_DTYPE)

    event_length = end_sample - start_sample

    if len(event_audio) < event_length:
        event_audio = np.pad(event_audio, (0, event_length - len(event_audio)))
    else:
        event_audio = event_audio[:event_length]

    timeline[start_sample:end_sample] = event_audio

    return timeline

# 混合多条音轨
def mix_tracks(tracks: list[np.ndarray]) -> np.ndarray:
    if len(tracks) == 0:
        return np.zeros(0, dtype=AUDIO_DTYPE)

    max_len = max(len(track) for track in tracks)
    output = np.zeros(max_len, dtype=AUDIO_DTYPE)

    for track in tracks:
        output[:len(track)] += track

    output = normalize_audio(output)

    return output


# 旋律通道调试

# 生成零均值脉冲波
def generate_zero_mean_pulse_by_phase(
    phase: np.ndarray,
    duty: float,
) -> np.ndarray:
    duty = float(duty)
    if not 0.0 < duty < 1.0:
        raise ValueError("duty 必须大于 0 且小于 1")

    high_level = 2.0 * (1.0 - duty)
    low_level = -2.0 * duty
    audio = np.where(phase < duty, high_level, low_level)

    return np.asarray(audio, dtype=AUDIO_DTYPE)


# 生成振荡器波形
def generate_oscillator_by_samples(
    channel: str,
    freq: float,
    n_samples: int,
    config: dict,
    initial_phase: float = 0.0,
    return_final_phase: bool = False,
):
    sr = config["sample_rate"]
    params = get_channel_params(channel, config)

    waveform = params["waveform"]
    n_samples = max(0, int(n_samples))

    phase = make_phase_by_samples(
        freq=freq,
        n_samples=n_samples,
        sr=sr,
        initial_phase=initial_phase,
    )

    if waveform == "square":
        audio = generate_zero_mean_pulse_by_phase(
            phase=phase,
            duty=params["duty"],
        )

    elif waveform == "triangle":
        audio = 4 * np.abs(phase - 0.5) - 1.0

        steps = params.get("steps", 16)

        if steps is not None and steps > 1:
            audio = np.round(audio * (steps - 1)) / (steps - 1)

        audio = np.asarray(audio, dtype=AUDIO_DTYPE)

    else:
        raise ValueError("waveform 只能是 square 或 triangle")

    final_phase = phase_after_samples(
        initial_phase=initial_phase,
        freq=freq,
        n_samples=n_samples,
        sr=sr,
    )

    if return_final_phase:
        return audio, final_phase

    return audio

# 应用通道包络
def apply_channel_envelope(
    channel: str,
    audio: np.ndarray,
    duration: float,
    config: dict,
) -> np.ndarray:
    sr = config["sample_rate"]
    params = get_channel_params(channel, config)

    env = adsr_envelope(
        duration=duration,
        attack=params["attack"],
        decay=params["decay"],
        sustain=params["sustain"],
        release=params["release"],
        sr=sr,
        max_attack_duration_sec=config.get("max_attack_duration_sec"),
        max_decay_duration_sec=config.get("max_decay_duration_sec"),
        max_release_duration_sec=config.get("max_release_duration_sec"),
    )

    if len(env) < len(audio):
        env = np.pad(env, (0, len(audio) - len(env)))
    else:
        env = env[:len(audio)]

    audio = audio * env * params["volume"]

    return audio

# 生成原始单音音频
def generate_raw_note_audio(
    channel: str,
    freq: float,
    duration: float,
    config: dict,
    initial_phase: float = 0.0,
    return_final_phase: bool = False,
):
    sr = config["sample_rate"]
    n_samples = int(duration * sr)

    audio, final_phase = generate_oscillator_by_samples(
        channel=channel,
        freq=freq,
        n_samples=n_samples,
        config=config,
        initial_phase=initial_phase,
        return_final_phase=True,
    )

    audio = apply_channel_envelope(
        channel=channel,
        audio=audio,
        duration=duration,
        config=config,
    )

    if return_final_phase:
        return audio, final_phase

    return audio


# 生成普通和弦音频
def generate_normal_chord_audio(
    channel: str,
    freqs: list[float],
    duration: float,
    config: dict,
    initial_phases: list[float] | None = None,
    return_final_phases: bool = False,
):
    if len(freqs) == 0:
        raise ValueError("freqs 不能为空")

    sr = config["sample_rate"]
    total_samples = int(duration * sr)

    if initial_phases is None:
        initial_phases = [0.0] * len(freqs)
    elif len(initial_phases) != len(freqs):
        raise ValueError("initial_phases 的数量必须与 freqs 相同")

    chord_audio = np.zeros(total_samples, dtype=AUDIO_DTYPE)
    final_phases: list[float] = []

    for freq, initial_phase in zip(freqs, initial_phases):
        note_audio, final_phase = generate_raw_note_audio(
            channel=channel,
            freq=freq,
            duration=duration,
            config=config,
            initial_phase=initial_phase,
            return_final_phase=True,
        )

        chord_audio += note_audio
        final_phases.append(final_phase)

    chord_audio = chord_audio / len(freqs)

    if return_final_phases:
        return chord_audio, final_phases

    return chord_audio


# 生成快速琶音音频
def generate_arpeggio_audio(
    channel: str,
    freqs: list[float],
    duration: float,
    arpeggio_interval: float,
    config: dict,
    initial_phase: float = 0.0,
    return_final_phase: bool = False,
):
    if len(freqs) == 0:
        raise ValueError("freqs 不能为空")

    sr = config["sample_rate"]
    total_samples = int(duration * sr)

    segment_samples = int(arpeggio_interval * sr)

    if segment_samples <= 0:
        raise ValueError("arpeggio_interval太小，导致间隔少于 1 个采样点")

    arp_audio = np.zeros(total_samples, dtype=AUDIO_DTYPE)

    position = 0
    note_index = 0
    current_phase = normalize_phase(initial_phase)

    while position < total_samples:
        current_freq = freqs[note_index % len(freqs)]

        remaining_samples = total_samples - position
        current_segment_samples = min(segment_samples, remaining_samples)

        segment_audio, current_phase = generate_oscillator_by_samples(
            channel=channel,
            freq=current_freq,
            n_samples=current_segment_samples,
            config=config,
            initial_phase=current_phase,
            return_final_phase=True,
        )

        arp_audio[position:position + current_segment_samples] = segment_audio

        position += current_segment_samples
        note_index += 1

    # 施加ADSR包络
    arp_audio = apply_channel_envelope(
        channel=channel,
        audio=arp_audio,
        duration=duration,
        config=config,
    )

    if return_final_phase:
        return arp_audio, current_phase

    return arp_audio


# 旋律声部

# 渲染单音事件
def render_note(
    channel: str,
    freq: float,
    start_time: float,
    end_time: float,
    config: dict | None = None,
) -> np.ndarray:
    config = make_config(config)
    sr = config["sample_rate"]

    if start_time < 0:
        raise ValueError("start_time 不能小于 0")

    if end_time <= start_time:
        raise ValueError("end_time 必须大于 start_time")

    duration = end_time - start_time

    event_audio = generate_raw_note_audio(
        channel=channel,
        freq=freq,
        duration=duration,
        config=config,
    )

    timeline_audio = place_audio_on_timeline(
        event_audio=event_audio,
        start_time=start_time,
        end_time=end_time,
        sr=sr,
    )

    target_rms_db = float(get_channel_params(channel, config).get("target_rms_db", -20.0))
    timeline_audio = finalize_track_audio(
        timeline_audio,
        config,
        target_rms_db=target_rms_db,
    )

    return timeline_audio


# 渲染和弦事件
def render_chord(
    channel: str,
    freqs: list[float],
    start_time: float,
    end_time: float,
    config: dict | None = None,
) -> np.ndarray:
    if len(freqs) == 0:
        raise ValueError("freqs 不能为空")

    config = make_config(config)
    sr = config["sample_rate"]

    if start_time < 0:
        raise ValueError("start_time 不能小于 0")

    if end_time <= start_time:
        raise ValueError("end_time 必须大于 start_time")

    duration = end_time - start_time

    params = get_channel_params(channel, config)
    arpeggio_interval = params.get("arpeggio_interval", 0.0)

    if arpeggio_interval < 0:
        raise ValueError("arpeggio_interval 不能小于 0")

    if arpeggio_interval == 0:
        event_audio = generate_normal_chord_audio(
            channel=channel,
            freqs=freqs,
            duration=duration,
            config=config,
        )

    else:
        event_audio = generate_arpeggio_audio(
            channel=channel,
            freqs=freqs,
            duration=duration,
            arpeggio_interval=arpeggio_interval,
            config=config,
        )

    timeline_audio = place_audio_on_timeline(
        event_audio=event_audio,
        start_time=start_time,
        end_time=end_time,
        sr=sr,
    )

    target_rms_db = float(get_channel_params(channel, config).get("target_rms_db", -20.0))
    timeline_audio = finalize_track_audio(
        timeline_audio,
        config,
        target_rms_db=target_rms_db,
    )

    return timeline_audio



# 生成鼓组


# 根据相位生成方波
def generate_square_by_phase(
    phase: np.ndarray,
    duty: float,
) -> np.ndarray:
    return generate_zero_mean_pulse_by_phase(phase=phase, duty=duty)

# 根据相位生成三角波
def generate_triangle_by_phase(
    phase: np.ndarray,
) -> np.ndarray:
    audio = 4 * np.abs(phase - 0.5) - 1.0

    return audio

# 生成白噪声
def generate_noise_by_samples(
    n_samples: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    audio = rng.uniform(
        low=-1.0,
        high=1.0,
        size=n_samples,
    )

    return audio

# 限制噪声频段
def filter_noise_by_fft(
    audio: np.ndarray,
    sr: int,
    mode: str,
    low_freq: float | None,
    high_freq: float | None,
) -> np.ndarray:
    if mode == "none":
        return audio

    if len(audio) <= 1:
        return audio

    spectrum = np.fft.rfft(audio)

    freqs = np.fft.rfftfreq(
        len(audio),
        d=1.0 / sr,
    )

    mask = np.ones_like(freqs, dtype=bool)

    if mode == "lowpass":
        if high_freq is not None:
            mask = freqs <= high_freq

    elif mode == "highpass":
        if low_freq is not None:
            mask = freqs >= low_freq

    elif mode == "bandpass":
        if low_freq is not None:
            mask = mask & (freqs >= low_freq)

        if high_freq is not None:
            mask = mask & (freqs <= high_freq)

    else:
        raise ValueError("noise_filter 只能是 none、lowpass、highpass 或 bandpass")

    spectrum = spectrum * mask

    filtered = np.fft.irfft(
        spectrum,
        n=len(audio),
    )

    return filtered



# 鼓组音色生成调试

# 生成原始鼓点音频
def generate_raw_drum_audio(
    drum: str,
    duration: float,
    loudness: float,
    config: dict,
    seed: int | None = None,
) -> np.ndarray:
    sr = config["sample_rate"]
    params = get_drum_params(drum, config)

    if duration <= 0:
        raise ValueError("duration 必须大于 0")

    if loudness < 0:
        raise ValueError("loudness 不能小于 0")

    n_samples = int(duration * sr)

    freqs = make_frequency_curve(
        start_freq=params["start_freq"],
        end_freq=params["end_freq"],
        n_samples=n_samples,
        curve=params["pitch_curve"],
    )

    phase = make_phase_by_frequency_curve(
        freqs=freqs,
        sr=sr,
    )

    square = generate_square_by_phase(
        phase=phase,
        duty=params["duty"],
    )

    triangle = generate_triangle_by_phase(
        phase=phase,
    )

    noise = generate_noise_by_samples(
        n_samples=n_samples,
        seed=seed,
    )

    noise = filter_noise_by_fft(
        audio=noise,
        sr=sr,
        mode=params["noise_filter"],
        low_freq=params["noise_low"],
        high_freq=params["noise_high"],
    )

    noise = peak_normalize(noise)

    audio = (
        triangle * params["triangle_mix"]
        + square * params["square_mix"]
        + noise * params["noise_mix"]
    )

    env = adsr_envelope(
        duration=duration,
        attack=params["attack"],
        decay=params["decay"],
        sustain=params["sustain"],
        release=params["release"],
        sr=sr,
        max_attack_duration_sec=config.get("max_attack_duration_sec"),
        max_decay_duration_sec=config.get("max_decay_duration_sec"),
        max_release_duration_sec=config.get("max_release_duration_sec"),
    )

    if len(env) < len(audio):
        env = np.pad(env, (0, len(audio) - len(env)))
    else:
        env = env[:len(audio)]

    audio = audio * env
    audio = audio * params["volume"]
    audio = audio * loudness

    return audio


# 鼓组音色生成

# 渲染鼓点事件
def render_drum(
    drum: str,
    start_time: float,
    end_time: float,
    loudness: float = 1.0,
    config: dict | None = None,
    seed: int | None = None,
) -> np.ndarray:
    config = make_config(config)
    sr = config["sample_rate"]

    if start_time < 0:
        raise ValueError("start_time 不能小于 0")

    if end_time <= start_time:
        raise ValueError("end_time 必须大于 start_time")

    if loudness < 0:
        raise ValueError("loudness 不能小于 0")

    duration = end_time - start_time

    event_audio = generate_raw_drum_audio(
        drum=drum,
        duration=duration,
        loudness=loudness,
        config=config,
        seed=seed,
    )

    timeline_audio = place_audio_on_timeline(
        event_audio=event_audio,
        start_time=start_time,
        end_time=end_time,
        sr=sr,
    )

    target_rms_db = float(get_drum_params(drum, config).get("target_rms_db", -20.0))
    timeline_audio = finalize_track_audio(
        timeline_audio,
        config,
        target_rms_db=target_rms_db,
    )

    return timeline_audio



# 统一接口

# 渲染统一事件
def render_event(
    channel: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    freq: float | None = None,
    freqs: list[float] | None = None,
    config: dict | None = None,
    drum: str | None = None,
    loudness: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    if start_time is None:
        raise ValueError("必须提供 start_time")

    if end_time is None:
        raise ValueError("必须提供 end_time")

    if channel is None:
        raise ValueError("必须提供 channel")

    if channel == "drums":
        if drum is None:
            raise ValueError("channel=drums 时必须提供 drum")

        if freq is not None or freqs is not None:
            raise ValueError("channel=drums 时不能提供 freq 或 freqs")

        return render_drum(
            drum=drum,
            start_time=start_time,
            end_time=end_time,
            loudness=loudness,
            config=config,
            seed=seed,
        )

    if drum is not None:
        raise ValueError("只有 channel=drums 时才能提供 drum")

    if channel not in MELODY_CHANNEL_NAMES:
        raise ValueError("channel 只能是 vocal、guitar、piano、bass、other 或 drums")

    # 旋律声部调用
    if freq is None and freqs is None:
        raise ValueError("生成旋律声部时，必须提供 freq 或 freqs")

    if freq is not None and freqs is not None:
        raise ValueError("freq 和 freqs 不能同时提供")

    if freq is not None:
        return render_note(
            channel=channel,
            freq=freq,
            start_time=start_time,
            end_time=end_time,
            config=config,
        )

    return render_chord(
        channel=channel,
        freqs=freqs,
        start_time=start_time,
        end_time=end_time,
        config=config,
    )


# 命令行调试

# 解析频率列表
def parse_freqs(text: str) -> list[float]:
    result = []

    for part in text.split(","):
        part = part.strip()

        if part:
            result.append(float(part))

    return result


# 命令行调试入口

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "芯片音乐合成器"
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["note", "chord"],
        help="旋律声部生成模式：单音note；和弦/快速琶音chord",
    )

    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        choices=["vocal", "guitar", "piano", "bass", "other", "drums"],
        help="选择声部：vocal / guitar / piano / bass / other / drums",
    )

    parser.add_argument(
        "--drum",
        type=str,
        default=None,
        choices=["kick", "snare", "hh", "cymbals", "toms"],
        help="当 --channel drums 时指定鼓件：kick / snare / hh / cymbals / toms",
    )

    parser.add_argument(
        "--freq",
        type=float,
        default=None,
        help="单音频率，单位Hz",
    )

    parser.add_argument(
        "--freqs",
        type=str,
        default=None,
        help="和弦频率列表，用逗号分隔",
    )

    parser.add_argument(
        "--start",
        type=float,
        required=True,
        help="起始时间，单位秒",
    )

    parser.add_argument(
        "--end",
        type=float,
        required=True,
        help="结束时间，单位秒",
    )

    parser.add_argument(
        "--loudness",
        type=float,
        default=1.0,
        help="鼓点归一化响度",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="鼓组调试用：固定噪声随机种子",
    )

    parser.add_argument(
        "--arp-interval",
        type=float,
        default=None,
        help="仅用于吉他或钢琴：快速琶音间隔。0 为普通和弦；大于 0 为快速琶音。",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 wav 文件名",
    )

    args = parser.parse_args()

    if args.channel == "drums":
        if args.mode is not None:
            raise ValueError("channel=drums 时不再使用 --mode")

        if args.drum is None:
            raise ValueError("channel=drums 时必须提供 --drum")

        if args.freq is not None or args.freqs is not None:
            raise ValueError("channel=drums 时不能提供 --freq 或 --freqs")

        config = make_config()

        audio = render_event(
            channel=args.channel,
            drum=args.drum,
            start_time=args.start,
            end_time=args.end,
            loudness=args.loudness,
            config=config,
            seed=args.seed,
        )

        save_wav(
            filename=args.output,
            audio=audio,
            sr=config["sample_rate"],
        )

        print(f"已生成 {args.drum} 鼓点：{args.output}")

    else:
        if args.mode is None:
            raise ValueError("旋律声部必须提供 --mode note 或 --mode chord")

        if args.drum is not None:
            raise ValueError("只有 --channel drums 时才能提供 --drum")

        config_override = {}

        if args.arp_interval is not None:
            if args.channel not in {"guitar", "piano"}:
                raise ValueError("--arp-interval 只适用于 guitar 和 piano")
            config_override = {
                "channels": {
                    args.channel: {
                        "arpeggio_interval": args.arp_interval,
                    }
                }
            }

        config = make_config(config_override)

        if args.mode == "note":
            if args.freq is None:
                raise ValueError("mode=note 时必须提供 --freq")

            if args.freqs is not None:
                raise ValueError("mode=note 时不能提供 --freqs")

            audio = render_event(
                channel=args.channel,
                freq=args.freq,
                start_time=args.start,
                end_time=args.end,
                config=config,
            )

            save_wav(
                filename=args.output,
                audio=audio,
                sr=config["sample_rate"],
            )

            print(f"已生成 {args.channel} 单音：{args.output}")

        elif args.mode == "chord":
            if args.freqs is None:
                raise ValueError("mode=chord 时必须提供 --freqs")

            if args.freq is not None:
                raise ValueError("mode=chord 时不能提供 --freq")

            freqs = parse_freqs(args.freqs)

            audio = render_event(
                channel=args.channel,
                freqs=freqs,
                start_time=args.start,
                end_time=args.end,
                config=config,
            )

            save_wav(
                filename=args.output,
                audio=audio,
                sr=config["sample_rate"],
            )

            arpeggio_interval = config["channels"][args.channel].get("arpeggio_interval", 0.0)

            if arpeggio_interval == 0:
                print(f"已生成 {args.channel} 普通和弦：{args.output}")
            else:
                print(
                    f"已生成 {args.channel} 快速琶音：{args.output}，"
                    f"arpeggio_interval={arpeggio_interval}"
                )

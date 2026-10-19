import argparse
import csv
from pathlib import Path

import json
from progress_utils import ProgressTracker, emit_progress


# 计算音频片段RMS
def compute_segment_rms(audio) -> float:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


# 读取配置覆盖项
def load_config_overrides(config_json_path, section_name):
    if config_json_path is None:
        return {}

    config_json_path = Path(config_json_path)
    if not config_json_path.exists():
        return {}

    with config_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    section = data.get(section_name, {})
    if not isinstance(section, dict):
        return {}

    return section


# 应用配置覆盖项
def apply_config_overrides(overrides):
    for key, value in overrides.items():
        if key in globals():
            globals()[key] = value


import librosa
import numpy as np
from scipy.signal import medfilt

# usage
# python pitch_extraction_bass.py --input test\test_bass.wav --output output\output_bass.csv

# 参数

# 音频重采样率
SR = 22050

# 帧移长度
# 数值越小，时间精度越高，但计算量越大
HOP_LENGTH = 256

# pYIN 分析的帧长度
# 数值越大，频率估计更稳定，但时间响应会变慢
FRAME_LENGTH = 2048

# 贝斯音高检测的最低频率
# B0 约为 30.87 Hz
FMIN = librosa.note_to_hz("B0")

# 贝斯音高检测的最高频率
# G4 约为 392 Hz
FMAX = librosa.note_to_hz("G4")

# 静音裁剪阈值，单位为 dB
# 数值越小，裁剪越严格；数值越大，保留的弱音更多
TRIM_TOP_DB = 70

# 谐波增强参数
# 数值越大，越强调谐波成分，削弱打击声、噪声等非谐波成分
HARMONIC_MARGIN = 3.0

# MIDI 音高轮廓中值滤波窗口大小
# 必须为奇数。数值越大，音高越平滑，但可能吞掉短促变化
SMOOTH_KERNEL_SIZE = 5

# 可填补的短暂无声帧数量
# 当前后音高相同，且中间无声间隙不超过该帧数时，会填补为同一音高
MAX_GAP_FRAMES = 2

# 最短音符时长，单位为秒
# 小于该时长的片段会被忽略，用于过滤误检和碎片音符
MIN_NOTE_DURATION_SEC = 0.04

# 长音频分块分析长度，单位秒。用于避免 pYIN 长时间单步卡住 UI 进度。
PITCH_ANALYSIS_CHUNK_SEC = 30.0


# 输出线性进度
def emit_linear_progress(fraction: float, message: str):
    fraction = max(0.0, min(1.0, float(fraction)))
    emit_progress(
        stage="pitch_extraction_bass",
        current=int(round(fraction * 10000)),
        total=10000,
        message=message,
        unit="percent",
    )


# 映射区间进度
def progress_between(start_fraction: float, end_fraction: float, local_fraction: float) -> float:
    local_fraction = max(0.0, min(1.0, float(local_fraction)))
    return start_fraction + (end_fraction - start_fraction) * local_fraction


# 分块运行pYIN分析
def run_pyin_in_chunks(y_harm: np.ndarray, start_progress: float = 0.22, end_progress: float = 0.68):
    total_samples = len(y_harm)
    if total_samples == 0:
        return np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0)

    chunk_samples = max(int(PITCH_ANALYSIS_CHUNK_SEC * SR), FRAME_LENGTH * 4)
    # 让切块边界尽量落在 hop 上，减少拼接后时间轴误差。
    chunk_samples = max(HOP_LENGTH, (chunk_samples // HOP_LENGTH) * HOP_LENGTH)

    f0_parts = []
    voiced_flag_parts = []
    voiced_prob_parts = []

    for chunk_index, start_sample in enumerate(range(0, total_samples, chunk_samples), start=1):
        end_sample = min(total_samples, start_sample + chunk_samples)
        chunk = y_harm[start_sample:end_sample]

        f0_chunk, voiced_flag_chunk, voiced_prob_chunk = librosa.pyin(
            chunk,
            fmin=FMIN,
            fmax=FMAX,
            sr=SR,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
        )

        f0_parts.append(np.nan_to_num(f0_chunk, nan=0.0))
        voiced_flag_parts.append(np.asarray(voiced_flag_chunk, dtype=bool))
        voiced_prob_parts.append(np.nan_to_num(voiced_prob_chunk, nan=0.0))

        local = end_sample / total_samples
        emit_linear_progress(
            progress_between(start_progress, end_progress, local),
            f"执行 pYIN 音高检测（分块 {chunk_index}）",
        )

    return (
        np.concatenate(f0_parts) if f0_parts else np.zeros(0),
        np.concatenate(voiced_flag_parts) if voiced_flag_parts else np.zeros(0, dtype=bool),
        np.concatenate(voiced_prob_parts) if voiced_prob_parts else np.zeros(0),
    )

# 平滑MIDI音高轨迹
def smooth_midi_contour(f0_hz, voiced_flag, kernel_size=SMOOTH_KERNEL_SIZE):
    midi = np.full(len(f0_hz), np.nan, dtype=float)
    valid = voiced_flag & (f0_hz > 0)
    midi[valid] = librosa.hz_to_midi(f0_hz[valid])

    starts = np.flatnonzero(np.diff(np.r_[False, valid, False]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, valid, False]) == -1)

    for start, end in zip(starts, ends):
        segment = midi[start:end]
        if len(segment) >= kernel_size:
            midi[start:end] = medfilt(segment, kernel_size=kernel_size)

    return midi


# 填补短暂无声间隙
def fill_short_unvoiced_gaps(midi_int, max_gap_frames=MAX_GAP_FRAMES):
    result = midi_int.copy()
    n = len(result)
    i = 0

    while i < n:
        if not np.isnan(result[i]):
            i += 1
            continue

        gap_start = i
        while i < n and np.isnan(result[i]):
            i += 1
        gap_end = i

        gap_len = gap_end - gap_start
        left_idx = gap_start - 1
        right_idx = gap_end

        if (
            gap_len <= max_gap_frames
            and left_idx >= 0
            and right_idx < n
            and not np.isnan(result[left_idx])
            and not np.isnan(result[right_idx])
            and int(result[left_idx]) == int(result[right_idx])
        ):
            result[gap_start:gap_end] = result[left_idx]

    return result


# 将MIDI音高转换为音名
def midi_to_name(midi_pitch):
    return librosa.midi_to_note(int(midi_pitch), unicode=False)


# 提取音符事件
def extract_note_events(midi_int, f0_hz, voiced_prob, y_audio, trim_start_sec):
    events = []
    frame_times = librosa.frames_to_time(
        np.arange(len(midi_int)),
        sr=SR,
        hop_length=HOP_LENGTH
    ) + trim_start_sec

    i = 0
    n = len(midi_int)

    while i < n:
        if np.isnan(midi_int[i]):
            i += 1
            continue

        start = i
        pitch = int(midi_int[i])

        while i < n and not np.isnan(midi_int[i]) and int(midi_int[i]) == pitch:
            i += 1

        end = i

        start_sec = frame_times[start]
        end_sec = frame_times[end - 1] + HOP_LENGTH / SR
        duration_sec = end_sec - start_sec

        if duration_sec < MIN_NOTE_DURATION_SEC:
            continue

        segment_f0 = f0_hz[start:end]
        segment_f0 = segment_f0[segment_f0 > 0]
        if len(segment_f0) == 0:
            continue

        median_f0 = float(np.median(segment_f0))

        start_sample = max(0, int(round((start_sec - trim_start_sec) * SR)))
        end_sample = min(len(y_audio), int(round((end_sec - trim_start_sec) * SR)))
        event_rms = compute_segment_rms(y_audio[start_sample:end_sample])

        events.append([
            start_sec,
            end_sec,
            "note",
            midi_to_name(pitch),
            str(pitch),
            median_f0,
            event_rms,
        ])

    return events


# 运行贝斯音高提取
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-json", type=Path, default=None, help="UI 传入的音高提取参数 JSON")
    args = parser.parse_args()

    pitch_overrides = load_config_overrides(args.config_json, "bass")
    apply_config_overrides(pitch_overrides)

    emit_linear_progress(0.02, "加载贝斯音频")
    y, _ = librosa.load(args.input, sr=SR, mono=True)
    emit_linear_progress(0.05, f"贝斯音频读取完成：{len(y) / max(SR, 1):.2f} 秒")

    emit_linear_progress(0.08, "裁剪静音")
    y, trim_idx = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    trim_start_sec = trim_idx[0] / SR
    emit_linear_progress(0.12, "静音裁剪完成")

    emit_linear_progress(0.16, "提取谐波成分")
    y_harm = librosa.effects.harmonic(y, margin=HARMONIC_MARGIN)
    emit_linear_progress(0.20, "谐波成分提取完成")

    emit_linear_progress(0.22, "执行 pYIN 音高检测")
    f0, voiced_flag, voiced_prob = run_pyin_in_chunks(y_harm)

    emit_linear_progress(0.70, "整理 pYIN 频率结果")
    f0 = np.nan_to_num(f0, nan=0.0)
    emit_linear_progress(0.72, "整理 voiced 标记")
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    emit_linear_progress(0.74, "整理 voiced 置信度")
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)

    emit_linear_progress(0.76, "平滑音高轮廓")
    midi_contour = smooth_midi_contour(
        f0,
        voiced_flag,
        kernel_size=SMOOTH_KERNEL_SIZE
    )

    emit_linear_progress(0.80, "量化 MIDI 音高")
    midi_int = np.full(len(midi_contour), np.nan, dtype=float)
    valid = ~np.isnan(midi_contour)
    midi_int[valid] = np.round(midi_contour[valid])

    emit_linear_progress(0.82, "填补短暂无声间隙")
    midi_int = fill_short_unvoiced_gaps(
        midi_int,
        max_gap_frames=MAX_GAP_FRAMES
    )

    emit_linear_progress(0.84, "提取贝斯音符事件")
    events = extract_note_events(
        midi_int,
        f0,
        voiced_prob,
        y,
        trim_start_sec
    )
    emit_linear_progress(0.88, f"贝斯音符事件整理完成：{len(events)} 个事件")

    emit_linear_progress(0.94, "整理事件 RMS")

    emit_linear_progress(0.97, "写入 CSV 文件")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "start_sec",
            "end_sec",
            "event_type",
            "note_names",
            "midi_pitches",
            "frequencies_hz",
            "rms",
        ])

        for event in events:
            start_sec, end_sec, event_type, note_names, midi_pitches, frequencies_hz, event_rms = event
            writer.writerow([
                f"{start_sec:.2f}",
                f"{end_sec:.2f}",
                event_type,
                note_names,
                midi_pitches,
                f"{frequencies_hz:.2f}",
                f"{max(0.0, float(event_rms)):.8f}",
            ])

    emit_linear_progress(1.0, "贝斯音高识别完成")
    print(f"完成：{output_path}")


if __name__ == "__main__":
    main()
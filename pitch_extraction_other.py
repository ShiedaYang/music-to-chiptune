import argparse
import csv
import sys
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
import torch
import torchcrepe
from scipy.signal import medfilt

# usage
# python pitch_extraction_other.py --input test\test_other.wav --output output\output_other.csv



# 参数

# 音频采样率
SR = 16000

# 帧移长度，单位为采样点
# 数值越小，时间分辨率越高，但计算量越大
HOP_LENGTH = 80

# torchcrepe检测的最低频率，单位 Hz
# 低于该频率的音高不会被检测
FMIN = 80.0

# torchcrepe检测的最高频率，单位 Hz
# 高于该频率的音高不会被检测
FMAX = 2000.0

# 静音裁剪阈值，单位 dB
# 数值越小，裁剪越严格；数值越大，保留的弱音更多
TRIM_TOP_DB = 70

# 谐波增强参数
# 数值越大，越强调谐波成分，越抑制打击声和噪声
HARMONIC_MARGIN = 3.0

# torchcrepe模型类型
# full精度较高但速度较慢；tiny速度较快但精度较低
TORCHCREPE_MODEL = "full"

# torchcrepe推理批大小
# 数值越大，速度可能越快，但显存或内存占用也越高
TORCHCREPE_BATCH_SIZE = 2048

# periodicity中值滤波窗口大小
# 用于平滑周期性置信度，通常使用奇数
PERIODICITY_MEDIAN_KERNEL = 3

# 静音阈值，单位 dB
# 低于该能量的区域会被认为不可靠
SILENCE_THRESHOLD_DB = -60.0

# pitch 置信度阈值
# 数值越高，保留的音高越可靠，但可能漏掉弱音
PITCH_CONFIDENCE_THRESHOLD = 0.28

# pitch均值滤波窗口大小
# 用于平滑音高轨迹，通常使用奇数
PITCH_MEAN_FILTER_KERNEL = 3

# MIDI音高轨迹中值滤波窗口大小
# 数值越大，音高越平滑，但可能削弱短音或快速变化
MIDI_SMOOTH_KERNEL_SIZE = 7

# 可填补的短暂无声间隙帧数
# 如果前后音高相同，短间隙会被补成同一个音
MAX_UNVOICED_GAP_FRAMES = 3

# 最短音符时长，单位秒
# 短于该时长的片段会被忽略
MIN_NOTE_SEC = 0.04

# onset 检测参数：峰值搜索前向最大窗口
ONSET_PRE_MAX = 4

# onset 检测参数：峰值搜索后向最大窗口
ONSET_POST_MAX = 4

# onset 检测参数：局部平均前向窗口
ONSET_PRE_AVG = 8

# onset 检测参数：局部平均后向窗口
ONSET_POST_AVG = 8

# onset检测灵敏度
# 数值越小，越容易检测到起音；数值越大，检测更保守
ONSET_DELTA = 0.04

# onset检测等待帧数
# 数值越大，越不容易把相近起音分开
ONSET_WAIT = 6

# 遇到onset时，当前音符至少已经持续的帧数
# 满足该条件才允许在 onset 位置切分
MIN_FRAMES_BEFORE_ONSET_SPLIT = 8

# 音高变化切分阈值，单位为半音
# 当前后音高差达到该值时，可能切分为新音符
PITCH_CHANGE_SEMITONES = 1

# 判断音高变化是否稳定时向后观察的帧数
PITCH_CHANGE_LOOKAHEAD_FRAMES = 4

# 判断音高变化是否稳定所需的最少有效帧数
PITCH_CHANGE_CONFIRM_FRAMES = 3

# 音符片段的最低平均periodicity
# 数值越高，越倾向于保留音高稳定、周期性强的片段
MIN_MEAN_PERIODICITY = 0.35

# 长音频分块分析长度，单位秒。用于避免 torchcrepe 长时间单步卡住 UI 进度。
PITCH_ANALYSIS_CHUNK_SEC = 30.0


# 输出线性进度
def emit_linear_progress(fraction: float, message: str):
    fraction = max(0.0, min(1.0, float(fraction)))
    emit_progress(
        stage="pitch_extraction_other",
        current=int(round(fraction * 10000)),
        total=10000,
        message=message,
        unit="percent",
    )


# 映射区间进度
def progress_between(start_fraction: float, end_fraction: float, local_fraction: float) -> float:
    local_fraction = max(0.0, min(1.0, float(local_fraction)))
    return start_fraction + (end_fraction - start_fraction) * local_fraction


# 分块运行TorchCrepe分析
def run_torchcrepe_in_chunks(y_harm: np.ndarray, device: str, start_progress: float = 0.30, end_progress: float = 0.70):
    total_samples = len(y_harm)
    if total_samples == 0:
        return np.zeros(0), np.zeros(0)

    chunk_samples = max(int(PITCH_ANALYSIS_CHUNK_SEC * SR), HOP_LENGTH * 64)
    chunk_samples = max(HOP_LENGTH, (chunk_samples // HOP_LENGTH) * HOP_LENGTH)

    pitch_parts = []
    periodicity_parts = []

    for chunk_index, start_sample in enumerate(range(0, total_samples, chunk_samples), start=1):
        end_sample = min(total_samples, start_sample + chunk_samples)
        chunk_audio_np = y_harm[start_sample:end_sample]
        audio = torch.tensor(chunk_audio_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode():
            pitch, periodicity = torchcrepe.predict(
                audio,
                SR,
                HOP_LENGTH,
                FMIN,
                FMAX,
                model=TORCHCREPE_MODEL,
                batch_size=TORCHCREPE_BATCH_SIZE,
                device=device,
                return_periodicity=True,
            )

            periodicity = torchcrepe.filter.median(periodicity, PERIODICITY_MEDIAN_KERNEL)
            periodicity = torchcrepe.threshold.Silence(SILENCE_THRESHOLD_DB)(
                periodicity,
                audio,
                SR,
                HOP_LENGTH,
            )
            pitch = torchcrepe.threshold.At(PITCH_CONFIDENCE_THRESHOLD)(pitch, periodicity)
            pitch = torchcrepe.filter.mean(pitch, PITCH_MEAN_FILTER_KERNEL)

        pitch_parts.append(pitch.squeeze(0).detach().cpu().numpy())
        periodicity_parts.append(periodicity.squeeze(0).detach().cpu().numpy())
        del audio, pitch, periodicity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local = end_sample / total_samples
        emit_linear_progress(
            progress_between(start_progress, end_progress, local),
            f"运行 torchcrepe 音高检测（分块 {chunk_index}）",
        )

    return (
        np.concatenate(pitch_parts) if pitch_parts else np.zeros(0),
        np.concatenate(periodicity_parts) if periodicity_parts else np.zeros(0),
    )

# 平滑MIDI音高轨迹
def smooth_midi_contour(f0_hz):
    midi = np.full(len(f0_hz), np.nan, dtype=float)
    valid = f0_hz > 0
    midi[valid] = librosa.hz_to_midi(f0_hz[valid])

    starts = np.flatnonzero(np.diff(np.r_[False, valid, False]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, valid, False]) == -1)

    for start, end in zip(starts, ends):
        segment = midi[start:end]
        if len(segment) >= MIDI_SMOOTH_KERNEL_SIZE:
            midi[start:end] = medfilt(
                segment,
                kernel_size=MIDI_SMOOTH_KERNEL_SIZE,
            )

    return midi


# 填补短暂无声间隙
def fill_short_unvoiced_gaps(midi_int):
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
            gap_len <= MAX_UNVOICED_GAP_FRAMES
            and left_idx >= 0
            and right_idx < n
            and not np.isnan(result[left_idx])
            and not np.isnan(result[right_idx])
            and int(result[left_idx]) == int(result[right_idx])
        ):
            result[gap_start:gap_end] = result[left_idx]

    return result


# 检测起音位置
def detect_onsets(y_harm):
    onset_env = librosa.onset.onset_strength(
        y=y_harm,
        sr=SR,
        hop_length=HOP_LENGTH,
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=SR,
        hop_length=HOP_LENGTH,
        units="frames",
        pre_max=ONSET_PRE_MAX,
        post_max=ONSET_POST_MAX,
        pre_avg=ONSET_PRE_AVG,
        post_avg=ONSET_POST_AVG,
        delta=ONSET_DELTA,
        wait=ONSET_WAIT,
    )

    return set(onset_frames.tolist())


# 提取音符事件
def extract_note_events(
    midi_int,
    f0_hz,
    periodicity,
    y_audio,
    trim_start_sec,
    onset_frames,
):
    events = []

    frame_times = librosa.frames_to_time(
        np.arange(len(midi_int)),
        sr=SR,
        hop_length=HOP_LENGTH,
    ) + trim_start_sec

    i = 0
    n = len(midi_int)

    while i < n:
        if np.isnan(midi_int[i]):
            i += 1
            continue

        start = i
        base_pitch = int(midi_int[i])
        j = i + 1

        while j < n:
            if np.isnan(midi_int[j]):
                break

            if (
                j in onset_frames
                and (j - start) >= MIN_FRAMES_BEFORE_ONSET_SPLIT
            ):
                break

            if abs(midi_int[j] - base_pitch) >= PITCH_CHANGE_SEMITONES:
                look_end = min(n, j + PITCH_CHANGE_LOOKAHEAD_FRAMES)
                future = midi_int[j:look_end]
                future = future[~np.isnan(future)]

                if (
                    len(future) >= PITCH_CHANGE_CONFIRM_FRAMES
                    and abs(np.median(future) - base_pitch)
                    >= PITCH_CHANGE_SEMITONES
                ):
                    break

            j += 1

        end = j

        start_sec = frame_times[start]
        end_sec = frame_times[end - 1] + HOP_LENGTH / SR
        duration_sec = end_sec - start_sec

        if duration_sec < MIN_NOTE_SEC:
            i = j
            continue

        segment_f0 = f0_hz[start:end]
        segment_f0 = segment_f0[segment_f0 > 0]

        if len(segment_f0) == 0:
            i = j
            continue

        segment_periodicity = periodicity[start:end]
        mean_periodicity = float(np.mean(segment_periodicity))

        if mean_periodicity < MIN_MEAN_PERIODICITY:
            i = j
            continue

        median_f0 = float(np.median(segment_f0))
        midi_pitch = int(round(librosa.hz_to_midi(median_f0)))
        note_name = librosa.midi_to_note(midi_pitch, unicode=False)

        start_sample = max(
            0,
            int(round((start_sec - trim_start_sec) * SR)),
        )
        end_sample = min(
            len(y_audio),
            int(round((end_sec - trim_start_sec) * SR)),
        )

        event_rms = compute_segment_rms(
            y_audio[start_sample:end_sample],
        )

        events.append([
            start_sec,
            end_sec,
            "note",
            note_name,
            str(midi_pitch),
            median_f0,
            event_rms,
        ])

        i = j

    return events


# 运行其他乐器音高提取
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-json", type=Path, default=None, help="UI 传入的音高提取参数 JSON")
    args = parser.parse_args()

    pitch_overrides = load_config_overrides(args.config_json, "other")
    apply_config_overrides(pitch_overrides)

    emit_linear_progress(0.02, "读取其他乐器音频")
    y, _ = librosa.load(args.input, sr=SR, mono=True)
    emit_linear_progress(0.05, f"其他乐器音频读取完成：{len(y) / max(SR, 1):.2f} 秒")

    emit_linear_progress(0.08, "裁剪静音")
    y, trim_idx = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    trim_start_sec = trim_idx[0] / SR
    emit_linear_progress(0.12, "静音裁剪完成")

    emit_linear_progress(0.16, "提取谐波成分")
    y_harm = librosa.effects.harmonic(y, margin=HARMONIC_MARGIN)
    emit_linear_progress(0.20, "谐波成分提取完成")

    emit_linear_progress(0.24, "准备 torchcrepe 音高检测")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emit_linear_progress(0.28, f"音高检测设备：{device}")

    emit_linear_progress(0.30, "运行 torchcrepe 音高检测")
    pitch, periodicity = run_torchcrepe_in_chunks(y_harm, device=device)

    emit_linear_progress(0.72, "平滑 MIDI 音高轨迹")
    midi_contour = smooth_midi_contour(pitch)

    emit_linear_progress(0.76, "量化 MIDI 音高")
    midi_int = np.full(len(midi_contour), np.nan, dtype=float)
    valid = ~np.isnan(midi_contour)
    midi_int[valid] = np.round(midi_contour[valid])

    emit_linear_progress(0.80, "填补短暂无声间隙")
    midi_int = fill_short_unvoiced_gaps(midi_int)

    emit_linear_progress(0.82, "检测起音")
    onset_frames = detect_onsets(y_harm)
    emit_linear_progress(0.86, f"起音检测完成：{len(onset_frames)} 个候选")

    emit_linear_progress(0.90, "切分其他乐器音符事件")
    events = extract_note_events(
        midi_int,
        pitch,
        periodicity,
        y,
        trim_start_sec,
        onset_frames,
    )
    emit_linear_progress(0.93, f"其他乐器音符事件整理完成：{len(events)} 个事件")

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
            (
                start_sec,
                end_sec,
                event_type,
                note_names,
                midi_pitches,
                frequencies_hz,
                event_rms,
            ) = event

            writer.writerow([
                f"{start_sec:.2f}",
                f"{end_sec:.2f}",
                event_type,
                note_names,
                midi_pitches,
                f"{frequencies_hz:.2f}",
                f"{max(0.0, float(event_rms)):.8f}",
            ])

    emit_linear_progress(1.0, "其他乐器音高识别完成")
    print(f"完成：{output_path}")


if __name__ == "__main__":
    main()
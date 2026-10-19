import argparse
import csv
from pathlib import Path

import json
from progress_utils import ProgressTracker, emit_progress


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


# 参数

# 帧移长度
# 数值越小，时间精度越高，但计算量越大
HOP_LENGTH = 512

# onset 检测灵敏度
# 数值越小，检测越敏感，可能检测出更多弱鼓点
# 数值越大，检测越保守，能减少误检
ONSET_DELTA = 0.22

# 两个鼓点之间的最小间隔，单位为秒
# 用于避免一个鼓点被重复检测
MIN_ONSET_INTERVAL_SEC = 0.06

# 对onset envelope做平滑的帧数
# 数值越大，越能抑制细碎伪影，但可能让极短鼓点变钝
ONSET_ENV_SMOOTH_FRAMES = 3

# onset 检测前的 RMS 绝对门限。
# 当前帧 RMS 低于该值时，对应的 onset envelope 会被压成 0
# 用于过滤分离模型产生的极弱伪影
RMS_GATE_ABS = 0.002

# onset检测前的RMS相对门限
# 会根据整轨RMS的95分位数动态计算门限
# 数值越大，检测越保守
RMS_GATE_RELATIVE_TO_P95 = 0.15

# 单个鼓点事件的最短持续时间，单位为秒
MIN_EVENT_DURATION_SEC = 0.03

# 单个鼓点事件的最长持续时间，单位为秒
MAX_EVENT_DURATION_SEC = 1.20

# 判断鼓点结束的相对能量阈值
# 当前帧 RMS 低于该鼓点峰值 RMS 的这个比例时，认为可能进入衰减结束
END_RELATIVE_THRESHOLD = 0.12

# 判断鼓点结束的绝对能量下限
END_ABSOLUTE_FLOOR = 1e-5

# 连续多少帧低于结束阈值后，才确认事件结束
END_SILENCE_FRAMES = 3

# 单个事件的最小峰值
# 低于这个值的 onset 会被认为是模型伪影或背景噪声
MIN_EVENT_PEAK = 0.015

# 单个事件的最小 RMS
# 低于这个值的事件会被过滤掉
MIN_EVENT_RMS = 0.002

# 单个事件RMS相对于整轨RMS 95分位数的最低比例
# 用来过滤比整轨有效能量弱很多的伪事件
MIN_EVENT_RMS_RATIO_TO_TRACK_P95 = 0.18

# onset前用于计算背景能量的时间长度，单位为秒
PRE_EVENT_SEC = 0.05

# 事件RMS至少要比onset前背景RMS高出多少倍
# 用于判断是否真的有瞬态冲击
MIN_TRANSIENT_RMS_RATIO = 1.60

# 当前景背景RMS太小时，不启用瞬态比判断，避免除以极小值导致不稳定
PRE_RMS_MIN_FOR_TRANSIENT_CHECK = 0.001

# 输出时间的小数位数
TIME_DECIMALS = 2

# 输出归一化响度的小数位数
RMS_DECIMALS = 8




# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入单轨鼓件 wav 文件")
    parser.add_argument("--output", required=True, help="输出 csv 文件路径")
    parser.add_argument("--config-json", type=Path, default=None, help="UI 传入的鼓点提取参数 JSON")
    return parser.parse_args()


# 输出处理进度
def print_progress(current, total, prefix="处理进度", width=32):
    emit_progress(
        stage="pitch_extraction_drums",
        current=current,
        total=total,
        message="正在提取鼓点",
        unit="event",
    )

# 加载鼓组音频
def load_audio(input_path):
    y, sr = librosa.load(input_path, sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        raise ValueError("输入音频为空")

    return y, sr


# 平滑数组数据
def smooth_array(x, window_size):
    if window_size <= 1:
        return x

    window = np.ones(window_size) / window_size
    return np.convolve(x, window, mode="same")


# 对齐数组长度
def align_length(x, target_length):
    if len(x) < target_length:
        return np.pad(x, (0, target_length - len(x)))

    return x[:target_length]


# 计算帧RMS
def compute_rms(y):
    rms = librosa.feature.rms(
        y=y,
        frame_length=HOP_LENGTH * 2,
        hop_length=HOP_LENGTH,
        center=True,
    )[0]

    return np.asarray(rms, dtype=np.float32)

# 对起音包络应用RMS门控
def apply_rms_gate_to_onset_envelope(onset_envelope, rms):
    rms = align_length(rms, len(onset_envelope))

    rms_p95 = float(np.percentile(rms, 95)) if len(rms) > 0 else 0.0

    gate_threshold = max(
        RMS_GATE_ABS,
        rms_p95 * RMS_GATE_RELATIVE_TO_P95,
    )

    gated = onset_envelope.copy()
    gated[rms < gate_threshold] = 0.0

    return gated


# 获取起音采样位置
def get_onset_samples(y, sr, rms):
    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    onset_envelope = np.asarray(onset_envelope, dtype=np.float32)

    onset_envelope = smooth_array(
        onset_envelope,
        ONSET_ENV_SMOOTH_FRAMES,
    )

    onset_envelope = apply_rms_gate_to_onset_envelope(
        onset_envelope=onset_envelope,
        rms=rms,
    )

    wait_frames = max(1, int(MIN_ONSET_INTERVAL_SEC * sr / HOP_LENGTH))

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        backtrack=True,
        delta=ONSET_DELTA,
        wait=wait_frames,
    )

    onset_samples = librosa.frames_to_samples(
        onset_frames,
        hop_length=HOP_LENGTH,
    )

    onset_samples = np.asarray(onset_samples, dtype=np.int64)
    onset_samples = onset_samples[(onset_samples >= 0) & (onset_samples < len(y))]

    return onset_samples


# 将采样位置转换为帧
def sample_to_frame(sample_index):
    return int(sample_index // HOP_LENGTH)


# 将帧位置转换为采样
def frame_to_sample(frame_index):
    return int(frame_index * HOP_LENGTH)


# 估算鼓点结束位置
def estimate_event_end(
    y,
    sr,
    rms,
    start_sample,
    next_start_sample,
):
    min_event_samples = int(MIN_EVENT_DURATION_SEC * sr)
    max_event_samples = int(MAX_EVENT_DURATION_SEC * sr)

    min_end_sample = min(len(y), start_sample + min_event_samples)
    max_end_sample = min(len(y), start_sample + max_event_samples)

    if next_start_sample is not None:
        max_end_sample = min(max_end_sample, next_start_sample)

    start_frame = sample_to_frame(start_sample)
    min_end_frame = sample_to_frame(min_end_sample)
    max_end_frame = min(sample_to_frame(max_end_sample), len(rms) - 1)

    if start_frame >= len(rms):
        return min_end_sample

    event_rms_region = rms[start_frame:max_end_frame + 1]

    if event_rms_region.size == 0:
        return min_end_sample

    peak_rms = float(np.max(event_rms_region))
    end_threshold = max(peak_rms * END_RELATIVE_THRESHOLD, END_ABSOLUTE_FLOOR)

    silent_count = 0

    for frame_index in range(min_end_frame, max_end_frame + 1):
        if rms[frame_index] <= end_threshold:
            silent_count += 1
        else:
            silent_count = 0

        if silent_count >= END_SILENCE_FRAMES:
            end_frame = frame_index - END_SILENCE_FRAMES + 1
            end_sample = frame_to_sample(end_frame)
            return max(end_sample, min_end_sample)

    return max_end_sample


# 计算鼓点事件RMS
def compute_event_rms(y, start_sample, end_sample):
    segment = y[start_sample:end_sample]

    if segment.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(segment ** 2)))

    return rms


# 计算鼓点事件峰值
def compute_event_peak(y, start_sample, end_sample):
    segment = y[start_sample:end_sample]

    if segment.size == 0:
        return 0.0

    peak = float(np.max(np.abs(segment)))

    return peak


# 计算鼓点前背景RMS
def compute_pre_event_rms(y, start_sample, sr):
    pre_samples = int(PRE_EVENT_SEC * sr)

    pre_start = max(0, start_sample - pre_samples)
    pre_end = start_sample

    segment = y[pre_start:pre_end]

    if segment.size == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(segment ** 2)))

    return rms

# 判断是否保留鼓点事件
def should_keep_event(y, sr, start_sample, end_sample, track_rms_p95):
    event_peak = compute_event_peak(y, start_sample, end_sample)
    event_rms = compute_event_rms(y, start_sample, end_sample)

    if event_peak < MIN_EVENT_PEAK:
        return False

    if event_rms < MIN_EVENT_RMS:
        return False

    if track_rms_p95 > 1e-8:
        if event_rms < track_rms_p95 * MIN_EVENT_RMS_RATIO_TO_TRACK_P95:
            return False

    pre_rms = compute_pre_event_rms(
        y=y,
        start_sample=start_sample,
        sr=sr,
    )

    if pre_rms >= PRE_RMS_MIN_FOR_TRANSIENT_CHECK:
        transient_ratio = event_rms / max(pre_rms, 1e-8)

        if transient_ratio < MIN_TRANSIENT_RMS_RATIO:
            return False

    return True


# 提取鼓点事件
def extract_events(y, sr):
    rms = compute_rms(y)
    track_rms_p95 = float(np.percentile(rms, 95)) if len(rms) > 0 else 0.0

    onset_samples = get_onset_samples(
        y=y,
        sr=sr,
        rms=rms,
    )

    events = []
    total = len(onset_samples)

    if total == 0:
        return events

    for index, start_sample in enumerate(onset_samples):
        next_start_sample = None

        if index + 1 < total:
            next_start_sample = int(onset_samples[index + 1])

        end_sample = estimate_event_end(
            y=y,
            sr=sr,
            rms=rms,
            start_sample=int(start_sample),
            next_start_sample=next_start_sample,
        )

        if end_sample <= start_sample:
            continue

        keep_event = should_keep_event(
            y=y,
            sr=sr,
            start_sample=int(start_sample),
            end_sample=int(end_sample),
            track_rms_p95=track_rms_p95,
        )

        if not keep_event:
            print_progress(index + 1, total)
            continue

        event_rms = compute_event_rms(
            y=y,
            start_sample=int(start_sample),
            end_sample=int(end_sample),
        )

        events.append(
            {
                "start_sec": int(start_sample) / sr,
                "end_sec": int(end_sample) / sr,
                "rms": event_rms,
            }
        )

        print_progress(index + 1, total)

    if total > 0:
        print()

    return events



# 写入鼓点CSV
def write_csv(events, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "start_sec",
        "end_sec",
        "event_type",
        "note_names",
        "midi_pitches",
        "frequencies_hz",
        "rms",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            writer.writerow(
                {
                    "start_sec": f"{event['start_sec']:.{TIME_DECIMALS}f}",
                    "end_sec": f"{event['end_sec']:.{TIME_DECIMALS}f}",
                    "event_type": "",
                    "note_names": "",
                    "midi_pitches": "",
                    "frequencies_hz": "",
                    "rms": f"{max(0.0, float(event['rms'])):.{RMS_DECIMALS}f}",
                }
            )


# 运行鼓点提取
def main():
    args = parse_args()

    pitch_overrides = load_config_overrides(args.config_json, "drums")
    apply_config_overrides(pitch_overrides)

    print("读取音频...")
    y, sr = load_audio(args.input)

    print("提取鼓点事件...")
    events = extract_events(y, sr)

    print("写入 CSV...")
    write_csv(events, args.output)

    print(f"有效事件数量：{len(events)}")
    print(f"done: {args.output}")


if __name__ == "__main__":
    main()
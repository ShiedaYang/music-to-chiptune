from __future__ import annotations

import argparse
import json
import inspect
import csv
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import librosa
import numpy as np
from basic_pitch.inference import predict

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

    path = Path(config_json_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到参数配置文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get(section_name, {})


# 应用配置覆盖项
def apply_config_overrides(config_json_path, section_name):
    overrides = load_config_overrides(config_json_path, section_name)

    for key, value in overrides.items():
        if key in globals():
            globals()[key] = value

    return overrides


STAGE_NAME = "pitch_extraction_guitar"


# 输出线性进度
def emit_linear_progress(fraction: float, message: str):
    fraction = max(0.0, min(1.0, float(fraction)))
    emit_progress(
        stage=STAGE_NAME,
        current=int(round(fraction * 10000)),
        total=10000,
        message=message,
        unit="percent",
    )


# 映射区间进度
def progress_between(start_fraction: float, end_fraction: float, local_fraction: float) -> float:
    local_fraction = max(0.0, min(1.0, float(local_fraction)))
    return start_fraction + (end_fraction - start_fraction) * local_fraction


# 估算音频时长
def estimate_audio_duration_seconds(audio_path: Path) -> float | None:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate > 0:
                return max(0.0, frame_count / float(frame_rate))
    except Exception:
        return None
    return None


# 估算Basic Pitch耗时
def estimate_basic_pitch_seconds(duration_sec: float | None) -> float:
    # 只用于 UI 平滑推进；真实完成事件会把进度校准到子区间末尾。
    if duration_sec is None:
        return 20.0
    return max(8.0, duration_sec * 0.35)


# 持续输出预估进度
def emit_estimated_progress_until_done(
    done_event: threading.Event,
    start_fraction: float,
    end_fraction: float,
    message: str,
    estimated_seconds: float,
    interval_sec: float = 0.5,
):
    started_at = time.time()
    emit_linear_progress(start_fraction, message)

    while not done_event.wait(interval_sec):
        elapsed = time.time() - started_at
        local = min(0.95, elapsed / max(estimated_seconds, 1e-6))
        emit_linear_progress(
            progress_between(start_fraction, end_fraction, local),
            message,
        )


# usage
# python pitch_extraction_guitar.py --input input_guitar.wav --output output_pitch.csv


# 参数

# 和弦聚合的起始时间容差，单位秒
# 两个或多个音符的起始时间差 <= 该值时，会被视为同一候选 note/chord 事件
ONSET_TOLERANCE_SECONDS = 0.05

# 判定为同一和弦时要求的最小重叠比例
MIN_OVERLAP_RATIO = 0.05

# 保留的最低频率，单位 Hz
# 默认标准吉他最低弦E2，约82.41Hz
MIN_FREQUENCY_HZ = 82.41

# 保留的最高频率，单位 Hz。
# 默认标准吉他最高弦E6，约1318.51Hz
MAX_FREQUENCY_HZ = 1318.51

# Basic Pitch起音阈值
# 数值越小，越容易保留弱起音；数值越大，检测越保守
BASIC_PITCH_ONSET_THRESHOLD = 0.5

# Basic Pitch帧阈值
# 数值越小，音符延续更容易被保留；数值越大，短弱音更容易被过滤
BASIC_PITCH_FRAME_THRESHOLD = 0.3

# Basic Pitch最短音符长度，单位为毫秒
# 数值越小，越容易保留短音；数值越大，碎片音符越少
BASIC_PITCH_MINIMUM_NOTE_LENGTH = 127.70

# 是否启用Basic Pitch的多音高弯音输出
BASIC_PITCH_MULTIPLE_PITCH_BENDS = False

# 是否启用 melodia trick
BASIC_PITCH_MELODIA_TRICK = True

# 归一化后再次过滤的最短音符时长，单位秒
# 用于过滤Basic Pitch输出中的极短碎片
MIN_NOTE_DURATION_SECONDS = 0.03

# 最小响度阈值
MIN_LOUDNESS = 0.0

# 单个和弦最多保留多少个音
# 0 表示不限制；大于 0 时优先保留响度较高的音
MAX_NOTES_PER_CHORD = 8



CSV_COLUMNS = [
    "start_sec",
    "end_sec",
    "event_type",
    "note_names",
    "midi_pitches",
    "frequencies_hz",
    "rms",
]


@dataclass(frozen=True)
class NoteEvent:

    start: float
    end: float
    midi: int
    loudness: float

    # 获取音符频率
    @property
    def frequency_hz(self) -> float:
        return midi_to_frequency(self.midi)

    # 获取音符名称
    @property
    def note_name(self) -> str:
        return midi_to_note_name(self.midi)

# 进度
class ProgressBar:

    # 初始化进度跟踪器
    def __init__(self, total: int, description: str = "处理进度"):
        self.tracker = ProgressTracker(
            stage=STAGE_NAME,
            total=total,
            unit="step",
        )
        self.current = 0

    # 更新进度状态
    def update(self, current=None, message="") -> None:
        if isinstance(current, str) and message == "":
            message = current
            current = None

        if current is None:
            self.current += 1
        else:
            self.current = int(current)

        self.tracker.update(
            current=self.current,
            message=message or "正在识别吉他音高",
        )

    # 完成进度状态
    def finish(self, message="完成") -> None:
        self.tracker.finish(message)

# 将MIDI音高转换为频率
def midi_to_frequency(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))

# 将MIDI音高转换为音名
def midi_to_note_name(midi_pitch: int) -> str:
    pitch_classes = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B",
    ]
    pitch_class = pitch_classes[midi_pitch % 12]
    octave = midi_pitch // 12 - 1
    return f"{pitch_class}{octave}"

# 整理Basic Pitch音符事件
def normalize_note_events(
    raw_note_events: Iterable[Sequence[float]],
    min_note_duration: float = MIN_NOTE_DURATION_SECONDS,
    min_loudness: float = MIN_LOUDNESS,
) -> List[NoteEvent]:
    notes: List[NoteEvent] = []

    for item in raw_note_events:
        if len(item) < 4:
            continue

        start, end, midi_pitch, amplitude = item[:4]

        start = float(start)
        end = float(end)
        midi_pitch = int(round(float(midi_pitch)))
        amplitude = float(amplitude)

        if end <= start:
            continue

        if (end - start) < min_note_duration:
            continue

        loudness = max(0.0, min(1.0, amplitude))

        if loudness < min_loudness:
            continue

        notes.append(
            NoteEvent(
                start=start,
                end=end,
                midi=midi_pitch,
                loudness=loudness,
            )
        )

    notes.sort(key=lambda n: (n.start, n.midi, n.end))
    return notes


# 完成音符分组
def finalize_note_group(
    group: Sequence[NoteEvent],
    max_notes_per_chord: int = MAX_NOTES_PER_CHORD,
) -> List[NoteEvent]:
    result = list(group)

    if max_notes_per_chord is not None and max_notes_per_chord > 0:
        result = sorted(result, key=lambda n: n.loudness, reverse=True)[:max_notes_per_chord]

    return sorted(result, key=lambda n: n.midi)

# 将音符聚合为事件
def group_notes_into_events(
    notes: Sequence[NoteEvent],
    onset_tolerance: float = ONSET_TOLERANCE_SECONDS,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
    max_notes_per_chord: int = MAX_NOTES_PER_CHORD,
) -> List[List[NoteEvent]]:
    if not notes:
        return []

    groups: List[List[NoteEvent]] = []

    current: List[NoteEvent] = [notes[0]]
    anchor_start = notes[0].start

    for note in notes[1:]:
        near_same_onset = abs(note.start - anchor_start) <= onset_tolerance

        overlaps_current = any(
            has_sufficient_overlap(note, other, min_overlap_ratio)
            for other in current
        )

        if near_same_onset and overlaps_current:
            current.append(note)
        else:
            groups.append(finalize_note_group(current, max_notes_per_chord=max_notes_per_chord))
            current = [note]
            anchor_start = note.start

    groups.append(finalize_note_group(current, max_notes_per_chord=max_notes_per_chord))
    return groups

# 判断音符是否充分重叠
def has_sufficient_overlap(
    a: NoteEvent,
    b: NoteEvent,
    min_overlap_ratio: float,
) -> bool:
    overlap = min(a.end, b.end) - max(a.start, b.start)

    if overlap <= 0:
        return False

    shorter_duration = min(a.end - a.start, b.end - b.start)

    if shorter_duration <= 0:
        return False

    return (overlap / shorter_duration) >= min_overlap_ratio

# 将事件转换为CSV行
def event_to_csv_row(group: Sequence[NoteEvent], event_rms: float) -> dict:
    event_type = "chord" if len(group) > 1 else "note"

    start_sec = min(n.start for n in group)
    end_sec = max(n.end for n in group)
    return {
        "start_sec": f"{start_sec:.2f}",
        "end_sec": f"{end_sec:.2f}",
        "event_type": event_type,
        "note_names": ";".join(n.note_name for n in group),
        "midi_pitches": ";".join(str(n.midi) for n in group),
        "frequencies_hz": ";".join(
            f"{n.frequency_hz:.2f}"
            for n in group
        ),
        "rms": f"{max(0.0, float(event_rms)):.8f}",
    }

# 计算音符组响度RMS
def compute_group_loudness_rms(
    grouped_events: Sequence[Sequence[NoteEvent]],
    audio_path: Path,
) -> list[float]:
    if not grouped_events:
        return []

    y_audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    y_audio = np.asarray(y_audio, dtype=np.float32)
    sample_rate = max(1, int(sample_rate))

    values: list[float] = []
    for group in grouped_events:
        start_sec = min(note.start for note in group)
        end_sec = max(note.end for note in group)
        start_sample = max(0, int(round(start_sec * sample_rate)))
        end_sample = min(len(y_audio), int(round(end_sec * sample_rate)))
        values.append(compute_segment_rms(y_audio[start_sample:end_sample]))
    return values


# 写入音高CSV
def write_csv(rows: Sequence[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# 运行Basic Pitch推理
def run_basic_pitch_predict(
    audio_path: Path,
    min_frequency: float,
    max_frequency: float,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
    multiple_pitch_bends: bool,
    melodia_trick: bool,
):
    kwargs = {
        "minimum_frequency": min_frequency,
        "maximum_frequency": max_frequency,
        "onset_threshold": onset_threshold,
        "frame_threshold": frame_threshold,
        "minimum_note_length": minimum_note_length,
        "multiple_pitch_bends": multiple_pitch_bends,
        "melodia_trick": melodia_trick,
    }

    try:
        supported = set(inspect.signature(predict).parameters.keys())
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
    except (TypeError, ValueError):
        pass

    return predict(str(audio_path), **kwargs)


# 运行带进度的Basic Pitch推理
def run_basic_pitch_predict_with_progress(
    audio_path: Path,
    min_frequency: float,
    max_frequency: float,
    onset_threshold: float,
    frame_threshold: float,
    minimum_note_length: float,
    multiple_pitch_bends: bool,
    melodia_trick: bool,
    start_progress: float = 0.08,
    end_progress: float = 0.64,
):
    done_event = threading.Event()
    estimated_seconds = estimate_basic_pitch_seconds(estimate_audio_duration_seconds(audio_path))
    progress_thread = threading.Thread(
        target=emit_estimated_progress_until_done,
        args=(
            done_event,
            start_progress,
            end_progress,
            "Basic Pitch 推理中",
            estimated_seconds,
        ),
        daemon=True,
    )
    progress_thread.start()

    try:
        return run_basic_pitch_predict(
            audio_path=audio_path,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            multiple_pitch_bends=multiple_pitch_bends,
            melodia_trick=melodia_trick,
        )
    finally:
        done_event.set()
        progress_thread.join(timeout=1.0)
        emit_linear_progress(end_progress, "Basic Pitch 推理完成")


# 提取音高并写入CSV
def extract_pitch_to_csv(
    audio_path: Path,
    output_csv: Path,
    onset_tolerance: float = ONSET_TOLERANCE_SECONDS,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
    min_frequency: float = MIN_FREQUENCY_HZ,
    max_frequency: float = MAX_FREQUENCY_HZ,
    onset_threshold: float = BASIC_PITCH_ONSET_THRESHOLD,
    frame_threshold: float = BASIC_PITCH_FRAME_THRESHOLD,
    minimum_note_length: float = BASIC_PITCH_MINIMUM_NOTE_LENGTH,
    multiple_pitch_bends: bool = BASIC_PITCH_MULTIPLE_PITCH_BENDS,
    melodia_trick: bool = BASIC_PITCH_MELODIA_TRICK,
    min_note_duration: float = MIN_NOTE_DURATION_SECONDS,
    min_loudness: float = MIN_LOUDNESS,
    max_notes_per_chord: int = MAX_NOTES_PER_CHORD,
) -> None:
    emit_linear_progress(0.02, "检查吉他输入音频")
    emit_linear_progress(0.05, "准备 Basic Pitch 参数")

    _, _, raw_note_events = run_basic_pitch_predict_with_progress(
        audio_path=audio_path,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
    )

    emit_linear_progress(0.68, f"标准化音符事件：{len(raw_note_events)} 个候选")
    notes = normalize_note_events(
        raw_note_events,
        min_note_duration=min_note_duration,
        min_loudness=min_loudness,
    )

    emit_linear_progress(0.74, f"过滤短音和弱音：{len(notes)} 个候选保留")
    notes = [
        n for n in notes
        if min_frequency <= n.frequency_hz <= max_frequency
    ]

    emit_linear_progress(0.80, f"按频率范围过滤：{len(notes)} 个音符保留")
    grouped_events = group_notes_into_events(
        notes,
        onset_tolerance=onset_tolerance,
        min_overlap_ratio=min_overlap_ratio,
        max_notes_per_chord=max_notes_per_chord,
    )

    emit_linear_progress(0.88, f"聚合 note/chord 事件：{len(grouped_events)} 个事件")
    emit_linear_progress(0.90, "计算事件 RMS")
    event_rms_values = compute_group_loudness_rms(grouped_events, audio_path)
    rows = [
        event_to_csv_row(group, event_rms)
        for group, event_rms in zip(grouped_events, event_rms_values)
    ]

    emit_linear_progress(0.94, "生成 CSV 行")
    write_csv(rows, output_csv)
    emit_linear_progress(0.99, "CSV 写入完成")
    emit_linear_progress(1.0, "吉他音高识别完成")

    print(f"完成：共导出 {len(rows)} 个事件到 {output_csv}")


# 解析命令行参数
def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用Basic Pitch提取吉他音频音高"
    )

    parser.add_argument(
        "--input",
        dest="audio_path",
        type=Path,
        required=True,
        help="输入吉他音频路径",
    )

    parser.add_argument(
        "--output",
        dest="output_csv",
        type=Path,
        required=True,
        help="输出 CSV 路径",
    )

    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=ONSET_TOLERANCE_SECONDS,
        help=f"和弦聚合的起始时间容差，单位秒",
    )

    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=MIN_OVERLAP_RATIO,
        help=f"判定为同一和弦时所需的最小重叠比例",
    )

    parser.add_argument(
        "--min-frequency",
        type=float,
        default=MIN_FREQUENCY_HZ,
        help=f"保留的最低频率Hz",
    )

    parser.add_argument(
        "--max-frequency",
        type=float,
        default=MAX_FREQUENCY_HZ,
        help=f"保留的最高频率Hz",
    )

    parser.add_argument("--config-json", type=Path, default=None, help="UI 传入的参数配置 JSON")

    return parser.parse_args(argv)


# 运行吉他音高提取
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    pitch_overrides = load_config_overrides(args.config_json, "guitar")

    extract_pitch_to_csv(
        audio_path=args.audio_path,
        output_csv=args.output_csv,
        onset_tolerance=float(pitch_overrides.get("ONSET_TOLERANCE_SECONDS", args.onset_tolerance)),
        min_overlap_ratio=float(pitch_overrides.get("MIN_OVERLAP_RATIO", args.min_overlap_ratio)),
        min_frequency=float(pitch_overrides.get("MIN_FREQUENCY_HZ", args.min_frequency)),
        max_frequency=float(pitch_overrides.get("MAX_FREQUENCY_HZ", args.max_frequency)),
        onset_threshold=float(pitch_overrides.get("BASIC_PITCH_ONSET_THRESHOLD", BASIC_PITCH_ONSET_THRESHOLD)),
        frame_threshold=float(pitch_overrides.get("BASIC_PITCH_FRAME_THRESHOLD", BASIC_PITCH_FRAME_THRESHOLD)),
        minimum_note_length=float(pitch_overrides.get("BASIC_PITCH_MINIMUM_NOTE_LENGTH", BASIC_PITCH_MINIMUM_NOTE_LENGTH)),
        multiple_pitch_bends=bool(pitch_overrides.get("BASIC_PITCH_MULTIPLE_PITCH_BENDS", BASIC_PITCH_MULTIPLE_PITCH_BENDS)),
        melodia_trick=bool(pitch_overrides.get("BASIC_PITCH_MELODIA_TRICK", BASIC_PITCH_MELODIA_TRICK)),
        min_note_duration=float(pitch_overrides.get("MIN_NOTE_DURATION_SECONDS", MIN_NOTE_DURATION_SECONDS)),
        min_loudness=float(pitch_overrides.get("MIN_LOUDNESS", MIN_LOUDNESS)),
        max_notes_per_chord=int(pitch_overrides.get("MAX_NOTES_PER_CHORD", MAX_NOTES_PER_CHORD)),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
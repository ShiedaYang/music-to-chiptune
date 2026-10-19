from __future__ import annotations

import argparse
import copy
import json
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import chiptune
from progress_utils import emit_progress

# usage
# chord
# python generator.py --channel guitar --csv guitar.csv --output guitar_chiptune.wav

# rapid arp
# python generator.py --channel guitar --csv guitar.csv --output guitar_arp.wav --arp-interval 0.04

# drums
# python generator.py --channel drums --drum kick --csv kick.csv --output kick_chiptune.wav



# 通道名称
MELODIC_CHANNELS = [
    "vocal",
    "guitar",
    "piano",
    "bass",
    "other",
]

DRUM_CHANNEL = "drums"

DRUM_NAMES = [
    "kick",
    "snare",
    "hh",
    "cymbals",
    "toms",
]

ALL_CHANNELS = MELODIC_CHANNELS + [DRUM_CHANNEL]

CROSSFADE_MELODIC_CHANNELS = {"vocal", "other"}


# 日志设置

# 设置日志格式
def setup_logger(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# 解析频率字符串
def parse_frequencies(text: str) -> list[float]:
    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    text = text.replace(",", ";")

    freqs: list[float] = []

    for part in text.split(";"):
        part = part.strip()

        if not part:
            continue

        value = float(part)

        if value > 0:
            freqs.append(value)

    return freqs

# 解析浮点数
def parse_float(value: str, default: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        if default is None:
            raise
        return default

    if math.isnan(result):
        if default is None:
            raise ValueError("数值不能是 NaN")
        return default

    return result

# 生成通用配置覆盖项
def make_common_config(
    master_volume: float | None,
    bit_depth: int | None,
) -> dict:
    config_override: dict = {
        "bit_depth": bit_depth,
    }

    if master_volume is not None:
        config_override["master_volume"] = master_volume

    return config_override

# 合并配置覆盖项
def merge_config_overrides(
    config_overrides: dict | None,
    common_config: dict,
) -> dict:
    merged = copy.deepcopy(config_overrides) if config_overrides else {}
    chiptune.deep_update(merged, common_config)
    return merged


# 读取JSON配置
def load_config_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"找不到合成参数配置文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# 扫描CSV事件范围
def scan_csv_rows(csv_path: Path, melodic: bool) -> tuple[int, float]:
    total_rows = 0
    max_end_time = 0.0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1
            start_text = str(row.get("start_sec", "")).strip()
            end_text = str(row.get("end_sec", "")).strip()

            if not start_text or not end_text:
                continue

            if melodic:
                freq_text = str(row.get("frequencies_hz", "")).strip()
                if not freq_text:
                    continue

            end_time = parse_float(row.get("end_sec"), default=0.0)
            if end_time > max_end_time:
                max_end_time = end_time

    return total_rows, max_end_time


# 创建音频累加器
def make_accumulator(total_samples: int) -> np.ndarray:
    return np.zeros(max(0, int(total_samples)), dtype=np.float32)


# 将事件音频加入累加器
def add_event_to_accumulator(
    accumulator: np.ndarray,
    event_audio: np.ndarray,
    start_time: float,
    end_time: float,
    sr: int,
) -> None:
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    if start_sample < 0:
        raise ValueError("start_time 不能小于 0")

    if end_sample <= start_sample:
        raise ValueError("end_time 必须大于 start_time")

    if start_sample >= len(accumulator):
        return

    end_sample = min(end_sample, len(accumulator))
    event_len = end_sample - start_sample

    if event_len <= 0:
        return

    event_audio = np.asarray(event_audio, dtype=np.float32)
    copy_len = min(event_len, len(event_audio))

    if copy_len > 0:
        accumulator[start_sample:start_sample + copy_len] += event_audio[:copy_len]


# 将事件采样加入累加器
def add_event_samples_to_accumulator(
    accumulator: np.ndarray,
    event_audio: np.ndarray,
    start_sample: int,
) -> None:
    start_sample = int(start_sample)
    if start_sample < 0:
        raise ValueError("start_sample 不能小于 0")
    if start_sample >= len(accumulator):
        return

    event_audio = np.asarray(event_audio, dtype=np.float32)
    copy_len = min(len(event_audio), len(accumulator) - start_sample)
    if copy_len > 0:
        accumulator[start_sample:start_sample + copy_len] += event_audio[:copy_len]


@dataclass
class MelodicRenderEvent:
    row_index: int
    start_time: float
    end_time: float
    event_type: str
    freqs: list[float]
    rms: float
    start_sample: int
    end_sample: int
    render_start_sample: int
    render_end_sample: int
    fade_in_range: tuple[int, int] | None = None
    fade_out_range: tuple[int, int] | None = None


# 读取旋律渲染事件
def load_melodic_render_events(csv_path: Path, sr: int) -> list[MelodicRenderEvent]:
    events: list[MelodicRenderEvent] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            row_index = index + 1
            start_time = parse_float(row.get("start_sec"))
            end_time = parse_float(row.get("end_sec"))
            event_type = str(row.get("event_type", "")).strip().lower()
            freqs = parse_frequencies(row.get("frequencies_hz", ""))
            event_rms = parse_float(row.get("rms"), default=1.0)

            if end_time <= start_time:
                raise ValueError(
                    f"第 {row_index} 行时间错误：end_sec 必须大于 start_sec"
                )
            if not freqs:
                raise ValueError(
                    f"第 {row_index} 行没有有效频率：frequencies_hz 为空"
                )
            if event_type not in {"note", "chord"}:
                raise ValueError(
                    f"第 {row_index} 行 event_type 只能是 note 或 chord，当前是：{event_type}"
                )

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            if end_sample <= start_sample:
                raise ValueError(
                    f"第 {row_index} 行时长小于一个采样点，无法合成"
                )

            events.append(
                MelodicRenderEvent(
                    row_index=row_index,
                    start_time=start_time,
                    end_time=end_time,
                    event_type=event_type,
                    freqs=freqs,
                    rms=event_rms,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    render_start_sample=start_sample,
                    render_end_sample=end_sample,
                )
            )

    events.sort(key=lambda event: (event.start_sample, event.end_sample, event.row_index))
    return events


# 检查交叉淡化参数
def validate_crossfade_config(channel: str, channel_params: dict) -> tuple[float, float]:
    if channel not in CROSSFADE_MELODIC_CHANNELS:
        return 0.0, 0.0

    crossfade_sec = float(channel_params.get("event_crossfade_sec", 0.0) or 0.0)
    max_gap_sec = float(channel_params.get("event_crossfade_max_gap_sec", 0.0) or 0.0)

    if not math.isfinite(crossfade_sec) or crossfade_sec < 0:
        raise ValueError(f"{channel} 的 event_crossfade_sec 必须是大于等于 0 的有限数值")
    if not math.isfinite(max_gap_sec) or max_gap_sec < 0:
        raise ValueError(f"{channel} 的 event_crossfade_max_gap_sec 必须是大于等于 0 的有限数值")

    return crossfade_sec, max_gap_sec


# 准备旋律事件交叉淡化
def prepare_melodic_crossfades(
    events: list[MelodicRenderEvent],
    sr: int,
    crossfade_sec: float,
    max_gap_sec: float,
) -> int:
    if len(events) < 2 or crossfade_sec <= 0 or max_gap_sec < 0:
        return 0

    requested_fade_samples = max(1, int(round(crossfade_sec * sr)))
    max_gap_samples = max(0, int(round(max_gap_sec * sr)))
    created = 0

    for previous, current in zip(events, events[1:]):
        gap_samples = current.start_sample - previous.end_sample

        if gap_samples < 0 or gap_samples > max_gap_samples:
            continue

        previous_duration = previous.end_sample - previous.start_sample
        current_duration = current.end_sample - current.start_sample
        fade_samples = min(
            requested_fade_samples,
            max(1, previous_duration // 2),
            max(1, current_duration // 2),
        )

        center_sample = (previous.end_sample + current.start_sample) // 2
        if previous.fade_in_range is not None:
            available = 2 * max(0, center_sample - previous.fade_in_range[1])
            fade_samples = min(fade_samples, available)

        if fade_samples < 2:
            continue

        fade_start = center_sample - fade_samples // 2
        fade_end = fade_start + fade_samples

        if fade_start < previous.start_sample or fade_end > current.end_sample:
            continue

        previous.render_end_sample = max(previous.render_end_sample, fade_end)
        current.render_start_sample = min(current.render_start_sample, fade_start)
        previous.fade_out_range = (fade_start, fade_end)
        current.fade_in_range = (fade_start, fade_end)
        created += 1

    return created


# 应用等功率淡化
def _apply_equal_power_fade(
    audio: np.ndarray,
    local_start: int,
    local_end: int,
    fade_in: bool,
) -> None:
    local_start = max(0, int(local_start))
    local_end = min(len(audio), int(local_end))
    length = local_end - local_start
    if length <= 0:
        return

    angles = np.linspace(0.0, np.pi / 2.0, length, endpoint=True, dtype=np.float32)
    weights = np.sin(angles) if fade_in else np.cos(angles)
    audio[local_start:local_end] *= weights


# 应用事件交叉淡化遮罩
def apply_event_crossfade_masks(
    event_audio: np.ndarray,
    event: MelodicRenderEvent,
) -> np.ndarray:
    event_audio = np.asarray(event_audio, dtype=np.float32)

    if event.fade_in_range is not None:
        start, end = event.fade_in_range
        _apply_equal_power_fade(
            event_audio,
            local_start=start - event.render_start_sample,
            local_end=end - event.render_start_sample,
            fade_in=True,
        )

    if event.fade_out_range is not None:
        start, end = event.fade_out_range
        _apply_equal_power_fade(
            event_audio,
            local_start=start - event.render_start_sample,
            local_end=end - event.render_start_sample,
            fade_in=False,
        )

    return event_audio


@dataclass
class VoicePhaseState:
    phase: float = 0.0
    sample_index: int = 0
    frequency: float = 0.0


class MelodicPhaseTracker:

    # 初始化旋律相位跟踪器
    def __init__(self) -> None:
        self.voices: list[VoicePhaseState] = []

    # 获取声部相位状态
    def _get_voice(self, voice_index: int) -> VoicePhaseState:
        while len(self.voices) <= voice_index:
            self.voices.append(VoicePhaseState())
        return self.voices[voice_index]

    # 计算指定位置相位
    def phase_at(self, voice_index: int, sample_index: int, sr: int) -> float:
        if sr <= 0:
            raise ValueError("sr 必须大于 0")

        voice = self._get_voice(voice_index)
        delta_samples = int(sample_index) - int(voice.sample_index)

        if voice.frequency <= 0.0 or delta_samples == 0:
            return float(voice.phase) % 1.0

        return float(
            (voice.phase + voice.frequency * delta_samples / float(sr)) % 1.0
        )

    # 更新声部相位状态
    def update(
        self,
        voice_index: int,
        sample_index: int,
        phase: float,
        frequency: float,
    ) -> None:
        voice = self._get_voice(voice_index)

        if int(sample_index) < int(voice.sample_index):
            return

        voice.phase = float(phase) % 1.0
        voice.sample_index = int(sample_index)
        voice.frequency = max(0.0, float(frequency))


# 获取琶音末尾频率
def get_arpeggio_last_frequency(
    freqs: list[float],
    total_samples: int,
    segment_samples: int,
) -> float:
    if not freqs or total_samples <= 0 or segment_samples <= 0:
        return 0.0

    last_segment_index = (total_samples - 1) // segment_samples
    return float(freqs[last_segment_index % len(freqs)])


# 生成旋律事件音频
def generate_melodic_event_audio(
    channel: str,
    event_type: str,
    freqs: list[float],
    duration: float,
    config: dict,
    start_sample: int = 0,
    phase_tracker: MelodicPhaseTracker | None = None,
) -> np.ndarray:
    if phase_tracker is None:
        phase_tracker = MelodicPhaseTracker()

    sr = int(config["sample_rate"])

    if event_type == "note":
        initial_phase = phase_tracker.phase_at(0, start_sample, sr)
        event_audio, final_phase = chiptune.generate_raw_note_audio(
            channel=channel,
            freq=freqs[0],
            duration=duration,
            config=config,
            initial_phase=initial_phase,
            return_final_phase=True,
        )
        phase_tracker.update(
            voice_index=0,
            sample_index=start_sample + len(event_audio),
            phase=final_phase,
            frequency=freqs[0],
        )

    elif event_type == "chord":
        params = chiptune.get_channel_params(channel, config)
        arpeggio_interval = params.get("arpeggio_interval", 0.0)

        if arpeggio_interval < 0:
            raise ValueError("arpeggio_interval 不能小于 0")

        if arpeggio_interval == 0:
            initial_phases = [
                phase_tracker.phase_at(voice_index, start_sample, sr)
                for voice_index in range(len(freqs))
            ]
            event_audio, final_phases = chiptune.generate_normal_chord_audio(
                channel=channel,
                freqs=freqs,
                duration=duration,
                config=config,
                initial_phases=initial_phases,
                return_final_phases=True,
            )

            end_sample = start_sample + len(event_audio)
            for voice_index, (freq, final_phase) in enumerate(zip(freqs, final_phases)):
                phase_tracker.update(
                    voice_index=voice_index,
                    sample_index=end_sample,
                    phase=final_phase,
                    frequency=freq,
                )
        else:
            initial_phase = phase_tracker.phase_at(0, start_sample, sr)
            event_audio, final_phase = chiptune.generate_arpeggio_audio(
                channel=channel,
                freqs=freqs,
                duration=duration,
                arpeggio_interval=arpeggio_interval,
                config=config,
                initial_phase=initial_phase,
                return_final_phase=True,
            )

            segment_samples = int(arpeggio_interval * sr)
            last_frequency = get_arpeggio_last_frequency(
                freqs=freqs,
                total_samples=len(event_audio),
                segment_samples=segment_samples,
            )
            phase_tracker.update(
                voice_index=0,
                sample_index=start_sample + len(event_audio),
                phase=final_phase,
                frequency=last_frequency,
            )
    else:
        raise ValueError(f"event_type 只能是 note 或 chord，当前是：{event_type}")

    return np.asarray(event_audio, dtype=np.float32)


# 旋律声部渲染


# 从CSV渲染旋律音轨
def render_melodic_score_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    channel: str = "guitar",
    arpeggio_interval: float | None = None,
    use_event_rms: bool = True,
    master_volume: float | None = None,
    bit_depth: int | None = 8,
    log_interval: int = 20,
    config_overrides: dict | None = None,
) -> np.ndarray:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    logging.info("开始读取乐谱文件：%s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    if channel not in MELODIC_CHANNELS:
        raise ValueError(
            "旋律通道只能是 vocal、guitar、piano、bass 或 other"
        )

    config_override = merge_config_overrides(
        config_overrides=config_overrides,
        common_config=make_common_config(
            master_volume=master_volume,
            bit_depth=bit_depth,
        ),
    )

    if arpeggio_interval is not None:
        if channel not in {"guitar", "piano"}:
            raise ValueError("快速琶音间隔只适用于 guitar 和 piano 通道")
        config_override.setdefault("channels", {})
        config_override["channels"].setdefault(channel, {})
        config_override["channels"][channel]["arpeggio_interval"] = arpeggio_interval

    config = chiptune.make_config(config_override)
    sr = config["sample_rate"]

    channel_params = chiptune.get_channel_params(channel, config)

    logging.info("采样率：%s Hz", sr)
    logging.info("当前通道：%s", channel)
    logging.info("通道波形：%s", channel_params.get("waveform"))
    logging.info("通道音量：%s", channel_params.get("volume"))
    logging.info("总音量：%s", config["master_volume"])
    logging.info("位深压缩：%s", "关闭" if bit_depth is None else bit_depth)
    logging.info("是否使用事件 RMS：%s", "是" if use_event_rms else "否")

    current_arp_interval = channel_params.get("arpeggio_interval", 0.0)

    if current_arp_interval == 0:
        logging.info("和弦模式：普通和弦")
    else:
        logging.info(
            "和弦模式：快速琶音，arpeggio_interval = %.4f 秒",
            current_arp_interval,
        )

    events = load_melodic_render_events(csv_path=csv_path, sr=sr)
    total_events = len(events)

    if total_events == 0:
        logging.warning("CSV 文件为空，输出静音音频：%s", output_path)
        empty_audio = np.zeros(int(0.1 * sr), dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chiptune.save_wav(filename=str(output_path), audio=empty_audio, sr=sr)
        emit_progress(stage=f"generator:{channel}", current=1, total=1, message=f"正在合成{channel}音轨", unit="event")
        return empty_audio

    crossfade_sec, max_gap_sec = validate_crossfade_config(channel, channel_params)
    crossfade_count = prepare_melodic_crossfades(
        events=events,
        sr=sr,
        crossfade_sec=crossfade_sec,
        max_gap_sec=max_gap_sec,
    )

    total_samples = max(
        max(event.render_end_sample for event in events),
        int(0.1 * sr),
    )
    mixed_audio = make_accumulator(total_samples)

    logging.info("读取完成，共 %d 个事件，输出轨长度 %.2f 秒", total_events, total_samples / sr)
    if channel in CROSSFADE_MELODIC_CHANNELS and crossfade_sec > 0:
        logging.info(
            "事件交叉淡化：重叠区 %.1f ms，最大桥接间隔 %.1f ms，已创建 %d 处",
            crossfade_sec * 1000.0,
            max_gap_sec * 1000.0,
            crossfade_count,
        )
    elif channel in CROSSFADE_MELODIC_CHANNELS:
        logging.info("事件交叉淡化：关闭")
    logging.info("开始渲染音符事件...")

    valid_events = 0
    phase_tracker = MelodicPhaseTracker()

    for index, event in enumerate(events, start=1):
        duration = (event.render_end_sample - event.render_start_sample) / sr
        event_audio = generate_melodic_event_audio(
            channel=channel,
            event_type=event.event_type,
            freqs=event.freqs,
            duration=duration,
            config=config,
            start_sample=event.render_start_sample,
            phase_tracker=phase_tracker,
        )

        if use_event_rms:
            event_rms = float(event.rms)
            if not math.isfinite(event_rms) or event_rms < 0:
                raise ValueError(f"第 {event.row_index} 行 rms 必须是大于等于 0 的有限数值")
            event_gain = event_rms
            event_audio = event_audio * event_gain
        else:
            event_gain = 1.0

        event_audio = apply_event_crossfade_masks(event_audio, event)
        add_event_samples_to_accumulator(
            accumulator=mixed_audio,
            event_audio=event_audio,
            start_sample=event.render_start_sample,
        )
        valid_events += 1

        should_log = (
            index == 1
            or index == total_events
            or index % log_interval == 0
        )

        if should_log:
            progress = index / total_events * 100
            logging.info(
                "处理进度：%d/%d，%.1f%%，通道：%s，类型：%s，时间：%.2f-%.2f 秒，频率数量：%d",
                index,
                total_events,
                progress,
                channel,
                event.event_type,
                event.start_time,
                event.end_time,
                len(event.freqs),
            )
            emit_progress(
                stage=f"generator:{channel}",
                current=index,
                total=total_events,
                message=f"正在合成{channel}音轨",
                unit="event",
            )

    if valid_events == 0:
        logging.warning("CSV 文件中没有有效旋律事件，输出静音音频：%s", output_path)
        mixed_audio = np.zeros(int(0.1 * sr), dtype=np.float32)

    logging.info("所有事件渲染完成，开始整轨 RMS 响度控制%s...", "；随后进行位深量化" if bit_depth is not None else "；中间轨不进行位深量化")
    target_rms_db = float(channel_params.get("target_rms_db", -20.0))
    mixed_audio, rms_stats = chiptune.finalize_track_audio(
        audio=mixed_audio,
        config=config,
        target_rms_db=target_rms_db,
        return_stats=True,
    )
    if rms_stats is not None:
        logging.info(
            "整轨后处理：当前 RMS %.2f dBFS，目标 %.2f dBFS，增益 %.2f dB，结果 RMS %.2f dBFS，峰值 %.2f dBFS%s",
            rms_stats["current_rms_db"],
            rms_stats["target_rms_db"],
            rms_stats["gain_db"],
            rms_stats.get("result_rms_db", rms_stats["current_rms_db"]),
            rms_stats.get("result_peak_db", rms_stats["current_peak_db"]),
            "（受峰值上限限制）" if rms_stats.get("peak_limited") else "",
        )

    duration = len(mixed_audio) / sr if len(mixed_audio) > 0 else 0.0

    logging.info("混音完成，音频总时长：%.2f 秒", duration)
    logging.info("开始保存 wav 文件：%s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chiptune.save_wav(
        filename=str(output_path),
        audio=mixed_audio,
        sr=sr,
    )

    logging.info("保存完成：%s", output_path)

    return mixed_audio


# 鼓组渲染

# 从CSV渲染鼓组音轨
def render_drum_score_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    drum: str = "kick",
    use_event_rms: bool = True,
    master_volume: float | None = None,
    bit_depth: int | None = 8,
    seed: int | None = None,
    log_interval: int = 20,
    config_overrides: dict | None = None,
) -> np.ndarray:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    logging.info("开始读取乐谱文件：%s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    if drum not in DRUM_NAMES:
        raise ValueError(
            "drum 只能是 kick、snare、hh、cymbals 或 toms"
        )

    config_override = merge_config_overrides(
        config_overrides=config_overrides,
        common_config=make_common_config(
            master_volume=master_volume,
            bit_depth=bit_depth,
        ),
    )

    config = chiptune.make_config(config_override)
    sr = config["sample_rate"]

    drum_params = chiptune.get_drum_params(drum, config)

    logging.info("采样率：%s Hz", sr)
    logging.info("当前通道：drums")
    logging.info("当前鼓件：%s", drum)
    logging.info("鼓件音量：%s", drum_params.get("volume"))
    logging.info("三角波比例：%s", drum_params.get("triangle_mix"))
    logging.info("方波比例：%s", drum_params.get("square_mix"))
    logging.info("噪声比例：%s", drum_params.get("noise_mix"))
    logging.info("总音量：%s", config["master_volume"])
    logging.info("位深压缩：%s", "关闭" if bit_depth is None else bit_depth)
    logging.info("是否使用事件 RMS：%s", "是" if use_event_rms else "否")

    total_events, max_end_time = scan_csv_rows(csv_path, melodic=False)

    if total_events == 0:
        logging.warning("CSV 文件为空，输出静音音频：%s", output_path)
        empty_audio = np.zeros(int(0.1 * sr), dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chiptune.save_wav(filename=str(output_path), audio=empty_audio, sr=sr)
        emit_progress(stage=f"generator:drums:{drum}", current=1, total=1, message=f"正在合成{drum}鼓组音轨", unit="event")
        return empty_audio

    total_samples = max(int(max_end_time * sr), int(0.1 * sr))
    mixed_audio = make_accumulator(total_samples)

    logging.info("读取完成，共 %d 个事件，输出轨长度 %.2f 秒", total_events, total_samples / sr)
    logging.info("开始渲染鼓点事件...")

    valid_events = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(reader, start=1):
            row_index = index + 1

            start_text = str(row.get("start_sec", "")).strip()
            end_text = str(row.get("end_sec", "")).strip()

            if not start_text and not end_text:
                continue

            start_time = parse_float(row.get("start_sec"))
            end_time = parse_float(row.get("end_sec"))
            event_rms = parse_float(row.get("rms"), default=1.0)

            if end_time <= start_time:
                raise ValueError(
                    f"第 {row_index} 行时间错误：end_sec 必须大于 start_sec"
                )

            if use_event_rms:
                if not math.isfinite(event_rms) or event_rms < 0:
                    raise ValueError(f"第 {row_index} 行 rms 必须是大于等于 0 的有限数值")
                event_gain = event_rms
            else:
                event_rms = 1.0
                event_gain = 1.0

            duration = end_time - start_time

            if seed is None:
                event_seed = None
            else:
                event_seed = seed + valid_events

            event_audio = chiptune.generate_raw_drum_audio(
                drum=drum,
                duration=duration,
                loudness=event_gain,
                config=config,
                seed=event_seed,
            )

            add_event_to_accumulator(
                accumulator=mixed_audio,
                event_audio=event_audio,
                start_time=start_time,
                end_time=end_time,
                sr=sr,
            )

            valid_events += 1

            should_log = (
                valid_events == 1
                or index == total_events
                or valid_events % log_interval == 0
            )

            if should_log:
                progress = index / total_events * 100

                logging.info(
                    "处理进度：%d/%d，%.1f%%，通道：drums，鼓件：%s，时间：%.2f-%.2f 秒，RMS：%.6f，事件增益：%.6f",
                    index,
                    total_events,
                    progress,
                    drum,
                    start_time,
                    end_time,
                    event_rms,
                    event_gain,
                )
                emit_progress(
                    stage=f"generator:drums:{drum}",
                    current=index,
                    total=total_events,
                    message=f"正在合成{drum}鼓组音轨",
                    unit="event",
                )

    if valid_events == 0:
        logging.warning("CSV 文件中没有有效鼓点事件，输出静音音频：%s", output_path)
        mixed_audio = np.zeros(int(0.1 * sr), dtype=np.float32)

    logging.info("所有事件渲染完成，开始整轨 RMS 响度控制%s...", "；随后进行位深量化" if bit_depth is not None else "；中间轨不进行位深量化")

    target_rms_db = float(drum_params.get("target_rms_db", -20.0))
    mixed_audio, rms_stats = chiptune.finalize_track_audio(
        audio=mixed_audio,
        config=config,
        target_rms_db=target_rms_db,
        return_stats=True,
    )
    if rms_stats is not None:
        logging.info(
            "整轨后处理：当前 RMS %.2f dBFS，目标 %.2f dBFS，增益 %.2f dB，结果 RMS %.2f dBFS，峰值 %.2f dBFS%s",
            rms_stats["current_rms_db"],
            rms_stats["target_rms_db"],
            rms_stats["gain_db"],
            rms_stats.get("result_rms_db", rms_stats["current_rms_db"]),
            rms_stats.get("result_peak_db", rms_stats["current_peak_db"]),
            "（受峰值上限限制）" if rms_stats.get("peak_limited") else "",
        )

    duration = len(mixed_audio) / sr if len(mixed_audio) > 0 else 0.0

    logging.info("混音完成，音频总时长：%.2f 秒", duration)
    logging.info("开始保存 wav 文件：%s", output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chiptune.save_wav(
        filename=str(output_path),
        audio=mixed_audio,
        sr=sr,
    )

    logging.info("保存完成：%s", output_path)

    return mixed_audio


# 统一渲染接口

# 从CSV渲染音轨
def render_score_from_csv(
    csv_path: str | Path,
    output_path: str | Path,
    channel: str = "guitar",
    drum: str | None = None,
    arpeggio_interval: float | None = None,
    use_event_rms: bool = True,
    master_volume: float | None = None,
    bit_depth: int | None = 8,
    seed: int | None = None,
    log_interval: int = 20,
    config_overrides: dict | None = None,
) -> np.ndarray:
    if channel == DRUM_CHANNEL:
        if drum is None:
            raise ValueError("当 channel='drums' 时，必须提供 drum")

        if arpeggio_interval is not None:
            raise ValueError("当 channel='drums' 时，不能使用 arpeggio_interval")

        return render_drum_score_from_csv(
            csv_path=csv_path,
            output_path=output_path,
            drum=drum,
            use_event_rms=use_event_rms,
            master_volume=master_volume,
            bit_depth=bit_depth,
            seed=seed,
            log_interval=log_interval,
            config_overrides=config_overrides,
        )

    if channel in MELODIC_CHANNELS:
        if drum is not None:
            raise ValueError("只有当 channel='drums' 时，才能提供 drum")

        return render_melodic_score_from_csv(
            csv_path=csv_path,
            output_path=output_path,
            channel=channel,
            arpeggio_interval=arpeggio_interval,
            use_event_rms=use_event_rms,
            master_volume=master_volume,
            bit_depth=bit_depth,
            log_interval=log_interval,
            config_overrides=config_overrides,
        )

    raise ValueError(
        "channel 只能是 vocal、guitar、piano、bass、other 或 drums"
    )


# 解析默认CSV路径
def resolve_default_csv_path(
    csv_path: str | None,
    channel: str,
    drum: str | None,
) -> str:
    if csv_path is not None:
        return csv_path

    if channel == DRUM_CHANNEL:
        if drum is None:
            raise ValueError("当 --channel drums 时，必须提供 --drum")
        return f"{drum}.csv"

    return "guitar.csv"


# 解析默认输出路径
def resolve_default_output_path(
    output_path: str | None,
    channel: str,
    drum: str | None,
) -> str:
    if output_path is not None:
        return output_path

    if channel == DRUM_CHANNEL:
        if drum is None:
            raise ValueError("当 --channel drums 时，必须提供 --drum")
        return f"{drum}_chiptune.wav"

    return "guitar_chiptune.wav"


# 运行命令行入口
def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据CSV乐谱调用chiptune.py生成芯片音乐音频。"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "输入乐谱 CSV 文件路径。"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "输出 wav 文件路径。"
        ),
    )

    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        choices=ALL_CHANNELS,
        help="选择通道：vocal / guitar / piano / bass / other / drums",
    )

    parser.add_argument(
        "--drum",
        type=str,
        default=None,
        choices=DRUM_NAMES,
        help="当 --channel drums 时，指定鼓件：kick / snare / hh / cymbals / toms",
    )

    parser.add_argument(
        "--arp-interval",
        type=float,
        default=None,
        help=(
            "覆盖吉他或钢琴通道的快速琶音间隔，单位秒"
            "0 表示普通和弦；大于 0 表示快速琶音"
            "默认使用普通和弦"
            "仅适用于 guitar / piano"
        ),
    )

    parser.add_argument(
        "--master-volume",
        type=float,
        default=None,
        help="总音量。默认使用 chiptune.py 中的 master_volume",
    )

    parser.add_argument(
        "--bit-depth",
        type=int,
        default=8,
        help="位深压缩，默认 8。",
    )

    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="合成参数 JSON 文件，由 UI 生成。",
    )

    parser.add_argument(
        "--ignore-rms",
        action="store_true",
        help="忽略 CSV 中的事件 RMS，所有事件使用相同音量",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="鼓组调试用：固定噪声随机种子。普通旋律声部不会使用该参数",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="每处理多少个事件显示一次进度日志，默认20",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志等级，默认 INFO",
    )

    args = parser.parse_args()

    setup_logger(args.log_level)

    if args.channel == DRUM_CHANNEL:
        if args.drum is None:
            raise ValueError("当 --channel drums 时，必须提供 --drum")

        if args.arp_interval is not None:
            raise ValueError("当 --channel drums 时，不能使用 --arp-interval")

    else:
        if args.drum is not None:
            raise ValueError("只有当 --channel drums 时，才需要提供 --drum")
        if args.arp_interval is not None and args.channel not in {"guitar", "piano"}:
            raise ValueError("--arp-interval 只适用于 guitar 和 piano")

    bit_depth = None if args.bit_depth == 0 else args.bit_depth

    csv_path = resolve_default_csv_path(
        csv_path=args.csv,
        channel=args.channel,
        drum=args.drum,
    )

    output_path = resolve_default_output_path(
        output_path=args.output,
        channel=args.channel,
        drum=args.drum,
    )

    config_overrides = load_config_json(args.config_json)

    audio = render_score_from_csv(
        csv_path=csv_path,
        output_path=output_path,
        channel=args.channel,
        drum=args.drum,
        arpeggio_interval=args.arp_interval,
        use_event_rms=not args.ignore_rms,
        master_volume=args.master_volume,
        bit_depth=bit_depth,
        seed=args.seed,
        log_interval=args.log_interval,
        config_overrides=config_overrides,
    )

    config_override = merge_config_overrides(
        config_overrides=config_overrides,
        common_config=make_common_config(
            master_volume=args.master_volume,
            bit_depth=bit_depth,
        ),
    )

    if args.channel in {"guitar", "piano"} and args.arp_interval is not None:
        config_override.setdefault("channels", {})
        config_override["channels"].setdefault(args.channel, {})
        config_override["channels"][args.channel]["arpeggio_interval"] = args.arp_interval

    config = chiptune.make_config(config_override)
    sr = config["sample_rate"]
    duration = len(audio) / sr if len(audio) > 0 else 0.0

    print()
    print("生成完成")
    print(f"输出文件：{output_path}")
    print(f"当前通道：{args.channel}")
    print(f"音频时长：{duration:.2f} 秒")

    if args.channel == DRUM_CHANNEL:
        drum_params = chiptune.get_drum_params(args.drum, config)

        print(f"当前鼓件：{args.drum}")
        print(f"鼓件音量：{drum_params.get('volume')}")
        print(f"三角波比例：{drum_params.get('triangle_mix')}")
        print(f"方波比例：{drum_params.get('square_mix')}")
        print(f"噪声比例：{drum_params.get('noise_mix')}")

        if args.ignore_rms:
            print("事件音量模式：忽略 RMS")
        else:
            print("事件音量模式：直接使用 RMS")

    else:
        channel_params = chiptune.get_channel_params(args.channel, config)
        current_arp_interval = channel_params.get("arpeggio_interval", 0.0)

        print(f"通道波形：{channel_params.get('waveform')}")

        if current_arp_interval == 0:
            print("和弦模式：普通和弦")
        else:
            print(f"和弦模式：快速琶音，arpeggio_interval={current_arp_interval} 秒")

        if args.ignore_rms:
            print("事件音量模式：忽略 RMS")
        else:
            print("事件音量模式：直接使用 RMS")


if __name__ == "__main__":
    main()

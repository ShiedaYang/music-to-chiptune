from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
import wave
import gc
from pathlib import Path
from typing import Iterable

import numpy as np

import chiptune
import generator
from progress_utils import emit_progress


BASE_DIR = Path(__file__).resolve().parent


# 格式化耗时
def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(seconds, 3600.0)
    minutes, secs = divmod(remainder, 60.0)

    if hours >= 1.0:
        return f"{int(hours)} 小时 {int(minutes)} 分 {secs:.2f} 秒"
    if minutes >= 1.0:
        return f"{int(minutes)} 分 {secs:.2f} 秒"
    return f"{secs:.2f} 秒"


# 记录阶段耗时
def log_stage_elapsed(stage_name: str, started_at: float) -> None:
    elapsed = time.perf_counter() - float(started_at)
    logging.info("%s完成，用时：%s", stage_name, format_elapsed(elapsed))


MELODIC_STEMS = {
    "vocals": {
        "channel": "vocal",
        "script_candidates": ["pitch_extraction_vocals.py", "pitch_extraction_vocal.py"],
        "aliases": ["vocals", "vocal"],
    },
    "guitar": {
        "channel": "guitar",
        "script_candidates": ["pitch_extraction_guitar.py"],
        "aliases": ["guitar"],
    },
    "bass": {
        "channel": "bass",
        "script_candidates": ["pitch_extraction_bass.py"],
        "aliases": ["bass"],
    },
    "piano": {
        "channel": "piano",
        "script_candidates": ["pitch_extraction_piano.py"],
        "aliases": ["piano"],
    },
    "other": {
        "channel": "other",
        "script_candidates": ["pitch_extraction_other.py"],
        "aliases": ["other", "others"],
    },
}

# 鼓组二次分离脚本内部会严格输出这些文件名：
# cymbals.wav / hh.wav / snare.wav / kick.wav / toms.wav
# 因此这里不再使用别名模糊匹配，避免无鼓件时出现误导性的“目标别名”报错。
DRUM_STEMS = ["cymbals", "hh", "snare", "kick", "toms"]

CSV_COLUMNS = [
    "start_sec",
    "end_sec",
    "event_type",
    "note_names",
    "midi_pitches",
    "frequencies_hz",
    "rms",
]


# 设置日志格式
def setup_logger(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

# 运行子进程，并实时转发 stdout/stderr，供 UI 解析 PROGRESS
PROGRESS_BAR_LINE_RE = re.compile(r"(\d{1,3}%\|)|(保存进度:)|(Processing audio chunks:)|(it/s\])")


# 判断库进度输出
def is_library_progress_line(line: str) -> bool:
    return bool(PROGRESS_BAR_LINE_RE.search(line))


# 运行子进程命令
def run_command(cmd: list[str], cwd: Path = BASE_DIR, env_overrides: dict[str, str] | None = None) -> None:
    logging.info("运行命令：%s", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        if is_library_progress_line(line):
            continue
        print(line, end="", flush=True)

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


# 查找脚本文件
def resolve_script(candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = BASE_DIR / name
        if path.exists():
            return path

    candidate_text = ", ".join(candidates)
    raise FileNotFoundError(f"找不到脚本文件，候选名称：{candidate_text}")


# 规范化文件名
def normalize_name(text: str) -> str:
    return text.lower().replace(" ", "_")


# 按别名查找WAV文件
def find_wav_by_aliases(root: Path, aliases: Iterable[str]) -> Path:
    root = Path(root)
    aliases_norm = [normalize_name(alias) for alias in aliases]

    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")

    wav_files = sorted(root.rglob("*.wav"))

    for wav_path in wav_files:
        stem = normalize_name(wav_path.stem)
        if stem in aliases_norm:
            return wav_path

    for wav_path in wav_files:
        stem = normalize_name(wav_path.stem)
        if any(alias in stem for alias in aliases_norm):
            return wav_path

    found = "\n".join(str(path) for path in wav_files[:30])
    raise FileNotFoundError(
        "没有找到目标 wav。\n"
        f"搜索目录：{root}\n"
        f"目标别名：{', '.join(aliases)}\n"
        f"当前找到的 wav：\n{found}"
    )


# 按名称查找WAV文件
def find_strict_wav(root: Path, stem_name: str) -> Path | None:
    """按内部约定的严格文件名查找 wav；找不到时返回 None。"""
    root = Path(root)
    if not root.exists():
        return None

    path = root / f"{stem_name}.wav"
    return path if path.exists() else None


# 判断WAV是否为空
def is_empty_wav(path: Path | None) -> bool:
    """判断 wav 是否缺失或没有可用音频帧。

    这里不把“读取元信息失败”的文件直接当作空文件，避免把损坏文件静默吞掉；
    非空但格式异常的文件仍交给后续脚本报出更明确的错误。
    """
    if path is None:
        return True

    path = Path(path)

    if not path.exists():
        return True

    if path.stat().st_size == 0:
        return True

    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() <= 0
    except Exception as exc:
        logging.warning("无法读取 wav 元信息，继续尝试处理：%s（%s）", path, exc)
        return False


# 按别名查找非空WAV文件
def try_find_non_empty_wav_by_aliases(root: Path, aliases: Iterable[str]) -> Path | None:
    """按别名查找 wav；找不到或文件为空时返回 None，而不是中断主流程。"""
    try:
        wav_path = find_wav_by_aliases(root=root, aliases=aliases)
    except FileNotFoundError:
        return None

    if is_empty_wav(wav_path):
        return None

    return wav_path


# 写入空事件CSV
def write_empty_score_csv(csv_path: Path) -> None:
    """写入只有表头的空 CSV，用于表示该声部/鼓件没有可渲染事件。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


# 标记跳过任务进度
def mark_skipped_progress(stage: str, message: str) -> None:
    emit_progress(stage, 1, 1, message, unit="skip")


# 检查CSV是否包含可渲染事件
def csv_has_renderable_events(csv_path: Path, melodic: bool) -> bool:
    if not csv_path.exists():
        return False

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            start_text = str(row.get("start_sec", "")).strip()
            end_text = str(row.get("end_sec", "")).strip()

            if not start_text or not end_text:
                continue

            if melodic:
                freq_text = str(row.get("frequencies_hz", "")).strip()
                if not freq_text:
                    continue

            return True

    return False


# 生成有效配置覆盖项
def make_effective_config_overrides(
    config_overrides: dict | None,
    master_volume: float | None,
    bit_depth: int | None,
) -> dict:
    return generator.merge_config_overrides(
        config_overrides=config_overrides,
        common_config=generator.make_common_config(
            master_volume=master_volume,
            bit_depth=bit_depth,
        ),
    )



# 运行主音轨分离
def run_main_separator(input_audio: Path, output_dir: Path, models_dir: Path) -> None:
    separator_script = resolve_script(["separator.py", "Separator.py"])
    run_command([
        sys.executable,
        str(separator_script),
        "--input",
        str(input_audio),
        "--output_dir",
        str(output_dir),
        "--models_dir",
        str(models_dir),
    ])


# 运行鼓组分离
def run_drum_separator(input_drums: Path, output_dir: Path) -> None:
    separator_script = resolve_script(["separator_drums.py"])
    run_command([
        sys.executable,
        str(separator_script),
        "--input",
        str(input_drums),
        "--output",
        str(output_dir),
    ])


# 运行音高提取
def run_pitch_extraction(
    script_path: Path,
    input_wav: Path,
    output_csv: Path,
    extra_args: list[str] | None = None,
    pitch_config_json: Path | None = None,
    progress_stage: str | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_wav),
        "--output",
        str(output_csv),
    ]

    if pitch_config_json is not None:
        cmd.extend(["--config-json", str(pitch_config_json)])

    if extra_args:
        cmd.extend(extra_args)

    env_overrides = {"PROGRESS_STAGE_OVERRIDE": progress_stage} if progress_stage else None
    run_command(cmd, env_overrides=env_overrides)



# 渲染旋律音轨
def render_melodic_track(
    csv_path: Path,
    output_wav: Path,
    channel: str,
    arpeggio_interval: float | None,
    use_event_rms: bool,
    master_volume: float | None,
    bit_depth: int | None,
    log_interval: int,
    config_overrides: dict | None = None,
) -> Path | None:
    if not csv_has_renderable_events(csv_path, melodic=True):
        logging.warning("跳过 %s：CSV 中没有可渲染的旋律事件", channel)
        return None

    audio = generator.render_score_from_csv(
        csv_path=csv_path,
        output_path=output_wav,
        channel=channel,
        drum=None,
        arpeggio_interval=arpeggio_interval,
        use_event_rms=use_event_rms,
        master_volume=master_volume,
        bit_depth=bit_depth,
        seed=None,
        log_interval=log_interval,
        config_overrides=config_overrides,
    )
    # 不把整条音轨留在 main.py 的列表中；音轨已经保存成 wav，释放生成阶段数组。
    del audio
    gc.collect()
    return output_wav if output_wav.exists() else None

# 渲染鼓组音轨
def render_drum_track(
    csv_path: Path,
    output_wav: Path,
    drum: str,
    use_event_rms: bool,
    master_volume: float | None,
    bit_depth: int | None,
    seed: int | None,
    log_interval: int,
    config_overrides: dict | None = None,
) -> Path | None:
    if not csv_has_renderable_events(csv_path, melodic=False):
        logging.warning("跳过 %s：CSV 中没有可渲染的鼓点事件", drum)
        return None

    audio = generator.render_score_from_csv(
        csv_path=csv_path,
        output_path=output_wav,
        channel="drums",
        drum=drum,
        arpeggio_interval=None,
        use_event_rms=use_event_rms,
        master_volume=master_volume,
        bit_depth=bit_depth,
        seed=seed,
        log_interval=log_interval,
        config_overrides=config_overrides,
    )
    del audio
    gc.collect()
    return output_wav if output_wav.exists() else None


# 读取WAV浮点音频块
def read_wav_chunk_as_float32(wf: wave.Wave_read, frames: int) -> np.ndarray:
    raw = wf.readframes(frames)
    if not raw:
        return np.zeros(0, dtype=np.float32)

    sample_width = wf.getsampwidth()
    channels = wf.getnchannels()

    if sample_width != 2:
        raise ValueError("当前最终混音只支持 16-bit PCM wav 中间音轨")

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0

    # 中间音轨原则上为单声道；若读到多声道则先等权折叠为单声道，再统一做声像定位。
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio


# 计算最终混音缩放比例
def compute_final_mix_scale(
    peak: float,
    final_volume: float,
    rms: float | None = None,
    rms_control: dict | None = None,
) -> float:
    if peak <= 1e-8:
        return 1.0

    settings = rms_control or {}
    enabled = bool(settings.get("enabled", False)) and rms is not None and rms > 1e-10

    if enabled:
        current_rms_db = chiptune.amplitude_to_db(float(rms))
        target_rms_db = float(settings.get("final_target_rms_db", -14.0))
        gain_db = target_rms_db - current_rms_db
        gain_db = max(
            float(settings.get("final_min_gain_db", -12.0)),
            min(float(settings.get("final_max_gain_db", 12.0)), gain_db),
        )
        scale = chiptune.db_to_amplitude(gain_db)
        peak_limit = chiptune.db_to_amplitude(float(settings.get("final_peak_db", -1.0)))
    else:
        scale = 1.0
        peak_limit = 1.0

    scale *= float(final_volume)

    if peak * scale > peak_limit:
        scale = peak_limit / peak

    return max(0.0, float(scale))


# 混合单轨音频块
def mix_track_chunk(
    handle_specs: list[tuple[wave.Wave_read, float, float]],
    current_frames: int,
    channel_mode: str,
    stereo_width: float,
) -> np.ndarray:
    if channel_mode == "mono":
        mixed = np.zeros(current_frames, dtype=np.float32)
        for wf, _left_gain, _right_gain in handle_specs:
            chunk = read_wav_chunk_as_float32(wf, current_frames)
            if len(chunk) > 0:
                mixed[:len(chunk)] += chunk
        return mixed

    mixed = np.zeros((current_frames, 2), dtype=np.float32)
    for wf, left_gain, right_gain in handle_specs:
        chunk = read_wav_chunk_as_float32(wf, current_frames)
        if len(chunk) > 0:
            mixed[:len(chunk), 0] += chunk * left_gain
            mixed[:len(chunk), 1] += chunk * right_gain

    return chiptune.apply_stereo_width(mixed, stereo_width)


# 混合已渲染音轨
def mix_rendered_tracks(
    tracks: list[Path | tuple[Path, float]],
    output_path: Path,
    sample_rate: int,
    final_volume: float,
    bit_depth: int | None = None,
    rms_control: dict | None = None,
    output_config: dict | None = None,
    chunk_frames: int = 262_144,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_params = chiptune.get_output_params({"output": output_config or {}})
    channel_mode = output_params["channel_mode"]
    stereo_width = output_params["stereo_width"]
    pan_law_db = output_params["pan_law_db"]
    output_channels = 2 if channel_mode == "stereo" else 1

    if bit_depth is not None:
        bit_depth = int(bit_depth)
        if bit_depth < 1 or bit_depth > 24:
            raise ValueError("bit_depth 必须在 1 到 24 之间，或使用 None 关闭量化")

    normalized_tracks: list[tuple[Path, float]] = []
    for item in tracks:
        if isinstance(item, tuple):
            path, pan = item
        else:
            path, pan = item, 0.0
        normalized_tracks.append((Path(path), chiptune.validate_pan(float(pan))))

    if not normalized_tracks:
        frames = int(0.1 * sample_rate)
        shape = (frames, 2) if output_channels == 2 else (frames,)
        mixed = np.zeros(shape, dtype=np.float32)
        chiptune.save_wav(filename=str(output_path), audio=mixed, sr=sample_rate)
        return frames / sample_rate

    infos: list[tuple[Path, int, float, float]] = []
    max_frames = 0

    for path, pan in normalized_tracks:
        with wave.open(str(path), "rb") as wf:
            framerate = wf.getframerate()
            if framerate != sample_rate:
                raise ValueError(
                    f"中间音轨采样率不一致：{path} 是 {framerate} Hz，期望 {sample_rate} Hz"
                )
            frames = wf.getnframes()
            if channel_mode == "stereo":
                left_gain, right_gain = chiptune.stereo_pan_gains(pan, pan_law_db)
            else:
                left_gain, right_gain = 1.0, 1.0
            infos.append((path, frames, left_gain, right_gain))
            max_frames = max(max_frames, frames)

    total_chunks = max(1, (max_frames + chunk_frames - 1) // chunk_frames)

    # 第一遍：按块应用声像和宽度，计算最终声道格式下的峰值与 RMS。
    peak = 0.0
    sum_squares = 0.0
    sample_count = 0
    handle_specs = [
        (wave.open(str(path), "rb"), left_gain, right_gain)
        for path, _frames, left_gain, right_gain in infos
    ]
    try:
        for chunk_index, chunk_start in enumerate(range(0, max_frames, chunk_frames), start=1):
            current_frames = min(chunk_frames, max_frames - chunk_start)
            mixed_chunk = mix_track_chunk(
                handle_specs=handle_specs,
                current_frames=current_frames,
                channel_mode=channel_mode,
                stereo_width=stereo_width,
            )

            chunk_peak = float(np.max(np.abs(mixed_chunk))) if mixed_chunk.size else 0.0
            peak = max(peak, chunk_peak)
            if mixed_chunk.size:
                mixed64 = np.asarray(mixed_chunk, dtype=np.float64).reshape(-1)
                sum_squares += float(np.dot(mixed64, mixed64))
                sample_count += int(mixed64.size)

            emit_progress(
                "final_mix",
                chunk_index,
                total_chunks * 2,
                "正在分析最终混音 RMS 与峰值",
                unit="chunk",
            )
    finally:
        for wf, _left_gain, _right_gain in handle_specs:
            wf.close()

    rms = float(np.sqrt(sum_squares / sample_count)) if sample_count > 0 else 0.0
    scale = compute_final_mix_scale(
        peak=peak,
        rms=rms,
        final_volume=final_volume,
        rms_control=rms_control,
    )
    logging.info(
        "最终混音：%s，宽度 %.2f，RMS %.2f dBFS，峰值 %.2f dBFS，输出增益 %.2f dB，预计结果 RMS %.2f dBFS / 峰值 %.2f dBFS，最终总线量化：%s",
        "双声道" if channel_mode == "stereo" else "单声道",
        stereo_width if channel_mode == "stereo" else 0.0,
        chiptune.amplitude_to_db(rms),
        chiptune.amplitude_to_db(peak),
        chiptune.amplitude_to_db(scale),
        chiptune.amplitude_to_db(rms * scale),
        chiptune.amplitude_to_db(peak * scale),
        "关闭" if bit_depth is None else f"{bit_depth}-bit（仅一次）",
    )

    # 第二遍：按最终声道格式混音并应用总线增益；随后在最终总线上只做一次位深量化，
    # 最后写入交错 16-bit PCM WAV 容器。中间音轨不再进行 8-bit 量化。
    handle_specs = [
        (wave.open(str(path), "rb"), left_gain, right_gain)
        for path, _frames, left_gain, right_gain in infos
    ]
    try:
        with wave.open(str(output_path), "w") as out_wav:
            out_wav.setnchannels(output_channels)
            out_wav.setsampwidth(2)
            out_wav.setframerate(sample_rate)

            for chunk_index, chunk_start in enumerate(range(0, max_frames, chunk_frames), start=1):
                current_frames = min(chunk_frames, max_frames - chunk_start)
                mixed_chunk = mix_track_chunk(
                    handle_specs=handle_specs,
                    current_frames=current_frames,
                    channel_mode=channel_mode,
                    stereo_width=stereo_width,
                )

                mixed_chunk *= scale

                # 位深压缩只在最终单双声道总线上执行一次。
                # 此时声像、立体声宽度、总线 RMS 和峰值增益均已完成。
                if bit_depth is not None:
                    chiptune.bit_crush_inplace(mixed_chunk, bit_depth)

                mixed_chunk = np.clip(mixed_chunk, -1.0, 1.0)
                mixed_int16 = (mixed_chunk * 32767).astype(np.int16)
                out_wav.writeframes(mixed_int16.reshape(-1).tobytes())

                emit_progress(
                    "final_mix",
                    total_chunks + chunk_index,
                    total_chunks * 2,
                    "正在写入最终混音",
                    unit="chunk",
                )
    finally:
        for wf, _left_gain, _right_gain in handle_specs:
            wf.close()

    return max_frames / sample_rate



# 解析命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完整流程：音频分离 -> 音高/鼓点提取 -> 芯片音色合成 -> 最终混音。"
    )

    parser.add_argument("--input", required=True, type=Path, help="输入音频文件，例如 song.mp3 / song.wav")
    parser.add_argument("--output", required=True, type=Path, help="最终输出 wav 文件路径")
    parser.add_argument("--work-dir", type=Path, default=Path("output/main_work"), help="中间文件目录")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="separator.py 使用的模型目录")
    parser.add_argument("--master-volume", type=float, default=None, help="传给 generator.py 的总音量")
    parser.add_argument("--bit-depth", type=int, default=8, help="位深压缩；0 表示关闭")
    parser.add_argument("--config-json", type=Path, default=None, help="UI 生成的合成参数 JSON 文件")
    parser.add_argument("--pitch-config-json", type=Path, default=None, help="UI 生成的音高/鼓点提取参数 JSON 文件")
    parser.add_argument("--final-volume", type=float, default=1.0, help="最终总混音音量倍率")
    parser.add_argument("--ignore-rms", action="store_true", help="忽略 CSV 中的事件 RMS，所有事件使用相同音量")
    parser.add_argument("--seed", type=int, default=None, help="鼓组噪声随机种子")
    parser.add_argument("--log-interval", type=int, default=20, help="generator.py 每处理多少事件输出一次日志")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--skip-existing", action="store_true", help="如果中间文件已存在则跳过对应阶段")
    parser.add_argument("--rmvpe-model", type=str, default=None, help="传给 pitch_extraction_vocals.py 的 --model")
    parser.add_argument("--rmvpe-threshold", type=float, default=None, help="传给 pitch_extraction_vocals.py 的 --thred")
    parser.add_argument("--guitar-arp", type=float, default=None, help="guitar 通道快速琶音间隔")
    parser.add_argument("--piano-arp", type=float, default=None, help="piano 通道快速琶音间隔")

    return parser.parse_args()


# 运行完整处理流程
def run_pipeline() -> int:
    args = parse_args()
    setup_logger(args.log_level)
    generator.setup_logger(args.log_level)

    input_audio = args.input.resolve()
    output_audio = args.output.resolve()
    work_dir = args.work_dir.resolve()
    models_dir = args.models_dir.resolve()

    if not input_audio.exists():
        raise FileNotFoundError(f"输入音频不存在：{input_audio}")

    bit_depth = None if args.bit_depth == 0 else args.bit_depth
    use_event_rms = not args.ignore_rms
    config_overrides = generator.load_config_json(args.config_json)
    pitch_config = generator.load_config_json(args.pitch_config_json) or {}

    final_config_override = make_effective_config_overrides(
        config_overrides=config_overrides,
        master_volume=args.master_volume,
        bit_depth=bit_depth,
    )
    config = chiptune.make_config(final_config_override)

    stems_dir = work_dir / "separated_stems"
    drum_stems_dir = work_dir / "separated_drums"
    csv_dir = work_dir / "csv"
    rendered_dir = work_dir / "rendered"

    for directory in [stems_dir, drum_stems_dir, csv_dir, rendered_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    emit_progress("main", 0, 6, "开始处理")

    stage_started_at = time.perf_counter()
    logging.info("阶段 1/6：分离主音频")
    emit_progress("main", 0, 6, "正在分离音频")
    if args.skip_existing and any(stems_dir.rglob("*.wav")):
        logging.info("检测到已存在的主分离结果，跳过 separator.py")
    else:
        run_main_separator(input_audio=input_audio, output_dir=stems_dir, models_dir=models_dir)

    log_stage_elapsed("阶段 1/6：分离主音频", stage_started_at)

    rendered_tracks: list[tuple[Path, float]] = []

    arpeggio_map = {
        "guitar": args.guitar_arp,
        "piano": args.piano_arp,
    }

    stage_started_at = time.perf_counter()
    logging.info("阶段 2/6：旋律声部音高提取")
    emit_progress("main", 1, 6, "正在提取旋律声部音高")
    melodic_wavs: dict[str, Path] = {}

    for stem_name, stem_config in MELODIC_STEMS.items():
        csv_path = csv_dir / f"{stem_name}.csv"
        progress_stage = f"pitch_extraction_{stem_name}"

        if args.skip_existing and csv_path.exists():
            logging.info("跳过音高提取，CSV 已存在：%s", csv_path)
            continue

        wav_path = try_find_non_empty_wav_by_aliases(
            root=stems_dir,
            aliases=stem_config["aliases"],
        )

        if wav_path is None:
            logging.warning(
                "未检测到可用的 %s 音频，按空声部处理并写入空 CSV：%s",
                stem_name,
                csv_path,
            )
            write_empty_score_csv(csv_path)
            mark_skipped_progress(progress_stage, f"未检测到 {stem_name} 音频，已跳过")
            continue

        melodic_wavs[stem_name] = wav_path
        script_path = resolve_script(stem_config["script_candidates"])

        extra_args: list[str] = []
        if stem_name == "vocals":
            if args.rmvpe_model is not None:
                extra_args.extend(["--model", args.rmvpe_model])
            if args.rmvpe_threshold is not None:
                extra_args.extend(["--thred", str(args.rmvpe_threshold)])

        logging.info("提取 %s -> %s", stem_name, csv_path)
        run_pitch_extraction(
            script_path=script_path,
            input_wav=wav_path,
            output_csv=csv_path,
            extra_args=extra_args,
            pitch_config_json=args.pitch_config_json,
            progress_stage=progress_stage,
        )

    log_stage_elapsed("阶段 2/6：旋律声部音高提取", stage_started_at)

    stage_started_at = time.perf_counter()
    logging.info("阶段 3/6：鼓组二次分离")
    emit_progress("main", 2, 6, "正在分离鼓组音频")

    drums_wav = try_find_non_empty_wav_by_aliases(root=stems_dir, aliases=["drums", "drum"])
    if drums_wav is None:
        logging.warning("主分离结果中没有可用的 drums.wav，按全曲无鼓组处理")
        emit_progress("separator_drums:demix", 1, 1, "未检测到鼓组音频，跳过鼓组二次分离", unit="skip")
        emit_progress("separator_drums:save", 1, 1, "未检测到鼓组音频，跳过鼓件保存", unit="skip")

    if drums_wav is None:
        pass
    elif args.skip_existing and any(drum_stems_dir.rglob("*.wav")):
        logging.info("检测到已存在的鼓组分离结果，跳过 separator_drums.py")
        emit_progress("separator_drums:demix", 1, 1, "已存在鼓组分离结果，跳过鼓组二次分离", unit="skip")
        emit_progress("separator_drums:save", 1, 1, "已存在鼓组分离结果，跳过鼓件保存", unit="skip")
    else:
        run_drum_separator(input_drums=drums_wav, output_dir=drum_stems_dir)

    log_stage_elapsed("阶段 3/6：鼓组二次分离", stage_started_at)

    stage_started_at = time.perf_counter()
    logging.info("阶段 4/6：鼓点事件提取")
    emit_progress("main", 3, 6, "正在提取鼓点")
    drum_pitch_script = resolve_script(["pitch_extraction_drums.py"])

    for drum_name in DRUM_STEMS:
        drum_wav = find_strict_wav(drum_stems_dir, drum_name)
        csv_path = csv_dir / f"{drum_name}.csv"
        progress_stage = f"pitch_extraction_drums:{drum_name}"

        if args.skip_existing and csv_path.exists():
            logging.info("跳过鼓点提取，CSV 已存在：%s", csv_path)
            mark_skipped_progress(progress_stage, f"已存在 {drum_name}.csv，跳过鼓点提取")
            continue

        if is_empty_wav(drum_wav):
            logging.warning(
                "未检测到可用的鼓件 %s 音频，按无鼓点处理并写入空 CSV：%s",
                drum_name,
                csv_path,
            )
            write_empty_score_csv(csv_path)
            mark_skipped_progress(progress_stage, f"未检测到 {drum_name} 音频，已跳过")
            continue

        logging.info("提取 %s -> %s", drum_name, csv_path)
        run_pitch_extraction(
            script_path=drum_pitch_script,
            input_wav=drum_wav,
            output_csv=csv_path,
            pitch_config_json=args.pitch_config_json,
            progress_stage=progress_stage,
        )

    log_stage_elapsed("阶段 4/6：鼓点事件提取", stage_started_at)

    stage_started_at = time.perf_counter()
    logging.info("阶段 5/6：根据 CSV 合成芯片音轨")
    emit_progress("main", 4, 6, "正在合成芯片音轨")

    for stem_name, stem_config in MELODIC_STEMS.items():
        channel = stem_config["channel"]
        csv_path = csv_dir / f"{stem_name}.csv"
        rendered_path = rendered_dir / f"{channel}.wav"

        rendered_track = render_melodic_track(
            csv_path=csv_path,
            output_wav=rendered_path,
            channel=channel,
            arpeggio_interval=arpeggio_map.get(channel),
            use_event_rms=use_event_rms,
            master_volume=args.master_volume,
            # 中间旋律轨保持未做 8-bit 量化；量化只在最终总线执行一次。
            bit_depth=None,
            log_interval=args.log_interval,
            config_overrides=config_overrides,
        )
        if rendered_track is not None:
            pan = float(chiptune.get_channel_params(channel, config).get("pan", 0.0))
            rendered_tracks.append((rendered_track, pan))

    for drum_index, drum_name in enumerate(DRUM_STEMS):
        csv_path = csv_dir / f"{drum_name}.csv"
        rendered_path = rendered_dir / f"{drum_name}.wav"
        drum_seed = None if args.seed is None else args.seed + drum_index * 10000

        rendered_track = render_drum_track(
            csv_path=csv_path,
            output_wav=rendered_path,
            drum=drum_name,
            use_event_rms=use_event_rms,
            master_volume=args.master_volume,
            # 中间鼓轨保持未做 8-bit 量化；量化只在最终总线执行一次。
            bit_depth=None,
            seed=drum_seed,
            log_interval=args.log_interval,
            config_overrides=config_overrides,
        )
        if rendered_track is not None:
            pan = float(chiptune.get_drum_params(drum_name, config).get("pan", 0.0))
            rendered_tracks.append((rendered_track, pan))

    log_stage_elapsed("阶段 5/6：根据 CSV 合成芯片音轨", stage_started_at)

    stage_started_at = time.perf_counter()
    logging.info("阶段 6/6：最终混音")
    emit_progress("main", 5, 6, "正在混合最终音频")
    emit_progress("final_mix", 0, 1, "正在混合最终音频")

    sample_rate = config["sample_rate"]

    duration = mix_rendered_tracks(
        tracks=rendered_tracks,
        output_path=output_audio,
        sample_rate=sample_rate,
        final_volume=args.final_volume,
        bit_depth=config.get("bit_depth"),
        rms_control=config.get("rms_control", {}),
        output_config=config.get("output", {}),
    )
    emit_progress("final_mix", 1, 1, "最终混音完成")
    emit_progress("main", 6, 6, "处理完成")
    log_stage_elapsed("阶段 6/6：最终混音", stage_started_at)

    print()
    print("全部处理完成")
    print(f"最终输出：{output_audio}")
    print(f"缓存目录：{work_dir}")
    print(f"最终时长：{duration:.2f} 秒")
    print(f"参与混音轨数：{len(rendered_tracks)}")

    return 0


# 运行主程序
def main() -> int:
    total_started_at = time.perf_counter()
    try:
        return_code = run_pipeline()
    except BaseException:
        logging.error(
            "处理未完成，总用时：%s",
            format_elapsed(time.perf_counter() - total_started_at),
        )
        raise

    logging.info(
        "全部处理完成，总用时：%s",
        format_elapsed(time.perf_counter() - total_started_at),
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import gc
import sys
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from ml_collections import ConfigDict

from progress_utils import emit_progress


# usage:
# python separator_drums.py --input test\test_drums.wav --output output\drums


# 固定路径
BASE_DIR = Path(__file__).resolve().parent

MSS_ROOT = BASE_DIR / "models" / "drum_separation" / "mss"
CONFIG_PATH = BASE_DIR / "models" / "drum_separation" / "drumsep" / "config_mdx23c.yaml"
CHECKPOINT_PATH = BASE_DIR / "models" / "drum_separation" / "drumsep" / "drumsep_5stems_mdx23c_jarredou.ckpt"


STRICT_DRUM_STEMS = ["cymbals", "hh", "snare", "kick", "toms"]


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入鼓组音频")
    parser.add_argument("--output", required=True, help="输出目录")
    return parser.parse_args()


# 加载鼓组分离配置
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))


# 加载鼓组分离模型
def load_model(config, checkpoint_path, device):
    model = TFC_TDF_net(config)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        ckpt = ckpt.get("state", ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)))

    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    return model


# 加载鼓组音频
def load_audio(input_path, config):
    sample_rate = int(getattr(config.audio, "sample_rate", 44100))
    num_channels = int(getattr(config.audio, "num_channels", 2))

    mix, _ = librosa.load(input_path, sr=sample_rate, mono=False)
    mix = np.asarray(mix, dtype=np.float32)

    if mix.ndim == 1:
        mix = np.stack([mix, mix], axis=0)

    if mix.shape[0] < num_channels:
        mix = np.tile(mix[:1], (num_channels, 1))
    elif mix.shape[0] > num_channels:
        mix = mix[:num_channels]

    return mix, sample_rate


# 规范化鼓组音频
def normalize_audio(mix, config):
    norm_mean = 0.0
    norm_std = 1.0

    if bool(getattr(config.inference, "normalize", False)):
        mono = mix.mean(axis=0)
        norm_mean = float(mono.mean())
        norm_std = float(mono.std())

        if norm_std > 1e-8:
            mix = (mix - norm_mean) / norm_std

    return mix, norm_mean, norm_std


# 恢复音频幅度
def restore_normalization(audio, config, norm_mean, norm_std):
    if bool(getattr(config.inference, "normalize", False)) and norm_std > 1e-8:
        audio = audio * norm_std + norm_mean

    return audio


# 估算鼓组分离耗时
def estimate_demix_seconds(duration_sec: float, device: torch.device) -> float:
    # 粗略估计：不同机器速度差异很大，因此只用于让 UI 进度条平滑推进。
    # 结束时会用真实完成事件校准到 100%。
    realtime_factor = 0.06 if device.type == "cuda" else 0.25
    return max(8.0, duration_sec * realtime_factor)


# 持续输出预估进度
def emit_estimated_progress_until_done(
    done_event: threading.Event,
    stage: str,
    message: str,
    estimated_seconds: float,
    interval_sec: float = 0.5,
):
    started_at = time.time()
    emit_progress(stage, 0, 1000, message, unit="percent", extra={"eta_seconds": estimated_seconds})

    while not done_event.wait(interval_sec):
        elapsed = time.time() - started_at
        ratio = min(0.95, elapsed / max(estimated_seconds, 1e-6))
        eta = max(0.0, estimated_seconds - elapsed)
        emit_progress(
            stage,
            int(ratio * 1000),
            1000,
            message,
            unit="percent",
            extra={"eta_seconds": eta},
        )


# 运行带进度的鼓组分离
def demix_with_estimated_progress(config, model, mix, device):
    duration_sec = mix.shape[-1] / float(getattr(config.audio, "sample_rate", 44100))
    estimated_seconds = estimate_demix_seconds(duration_sec, device)
    done_event = threading.Event()
    progress_thread = threading.Thread(
        target=emit_estimated_progress_until_done,
        args=(done_event, "separator_drums:demix", "正在执行鼓组二次分离", estimated_seconds),
        daemon=True,
    )
    progress_thread.start()

    try:
        with torch.no_grad():
            return demix(
                config=config,
                model=model,
                mix=mix,
                device=device,
                model_type="mdx23c",
                pbar=False,
            )
    finally:
        done_event.set()
        progress_thread.join(timeout=1.0)
        emit_progress("separator_drums:demix", 1000, 1000, "鼓组二次分离完成", unit="percent")


# 获取输出鼓件名称
def get_output_stem_names(config) -> list[str]:
    try:
        names = list(prefer_target_instrument(config))
    except Exception:
        names = STRICT_DRUM_STEMS

    # 只保留本项目内部约定的严格名称，并保持固定顺序，避免别名和模型内部顺序影响后续主流程。
    name_set = set(names)
    strict_names = [name for name in STRICT_DRUM_STEMS if name in name_set]
    return strict_names or STRICT_DRUM_STEMS


# 保存鼓件音轨
def save_stems(stems, config, output_dir, sample_rate, norm_mean, norm_std):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_names = get_output_stem_names(config)
    total = len(stem_names)

    for index, stem_name in enumerate(stem_names, start=1):
        if stem_name not in stems:
            emit_progress(
                "separator_drums:save",
                index,
                total,
                f"未生成 {stem_name}.wav，跳过保存",
                unit="stem",
            )
            continue

        audio = np.asarray(stems[stem_name], dtype=np.float32)
        audio = restore_normalization(audio, config, norm_mean, norm_std)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        if audio.ndim == 1:
            audio = audio[None, :]

        output_path = output_dir / f"{stem_name}.wav"
        sf.write(output_path, audio.T, sample_rate, subtype="FLOAT")

        # 不再打印文本进度条，改为结构化进度事件给 UI 使用。
        emit_progress(
            "separator_drums:save",
            index,
            total,
            f"正在保存鼓件音频：{stem_name}",
            unit="stem",
        )

        del audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 运行鼓组分离
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(MSS_ROOT))

    global TFC_TDF_net
    global demix
    global prefer_target_instrument

    from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
    from utils.model_utils import demix, prefer_target_instrument

    emit_progress("separator_drums:load", 0, 4, "正在读取鼓组分离配置", unit="step")
    print("读取配置...")
    config = load_config(CONFIG_PATH)

    emit_progress("separator_drums:load", 1, 4, "正在加载鼓组分离模型", unit="step")
    print("加载模型...")
    model = load_model(config, CHECKPOINT_PATH, device)

    emit_progress("separator_drums:load", 2, 4, "正在读取鼓组音频", unit="step")
    print("读取音频...")
    mix, sample_rate = load_audio(args.input, config)

    emit_progress("separator_drums:load", 3, 4, "正在归一化鼓组音频", unit="step")
    print("归一化音频...")
    mix, norm_mean, norm_std = normalize_audio(mix, config)
    emit_progress("separator_drums:load", 4, 4, "鼓组分离准备完成", unit="step")

    print("开始分离...")
    stems = demix_with_estimated_progress(config, model, mix, device)

    print("保存分离结果...")
    save_stems(
        stems=stems,
        config=config,
        output_dir=args.output,
        sample_rate=sample_rate,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    del stems, mix, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("done")


if __name__ == "__main__":
    main()

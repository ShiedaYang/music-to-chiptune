from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

from progress_utils import emit_progress


# 计算音频片段RMS
def compute_segment_rms(audio) -> float:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


# usage:
# python pitch_extraction_vocals.py --input input_vocals.wav --output output_vocals.csv --model models/rmvpe/rmvpe.pt


# =========================
# 与 UI / main.py 对齐的参数
# =========================

# RMVPE 固定推理采样率
SR = 16000

# RMVPE 默认帧移。RVC 常用 rmvpe.pt 为 160 samples，即 10 ms/frame。
HOP_LENGTH = 160

# RMVPE salience 解码阈值；越小越灵敏，越大越保守。
DEFAULT_RMVPE_THRESHOLD = 0.03

# 最短音符时长，单位秒。
MIN_NOTE_SEC = 0.05

# MIDI 轨迹中音高变化超过该值时切分音符，单位半音。
PITCH_SPLIT_SEMITONES = 0.35

# 音高变化需要连续确认的帧数，避免颤音被切得过碎。
PITCH_HOLD_FRAMES = 2

# 起音切分的最小间隔帧数。
ONSET_MIN_GAP_FRAMES = 4

# MIDI 轨迹中值平滑窗口，必须为奇数；1 表示不平滑。
SMOOTH_KERNEL_SIZE = 3

# 可填补的短暂无声帧数。
MAX_UNVOICED_GAP_FRAMES = 1

# onset 检测参数。
ONSET_PRE_MAX = 3
ONSET_POST_MAX = 3
ONSET_PRE_AVG = 6
ONSET_POST_AVG = 6
ONSET_DELTA = 0.05
ONSET_WAIT = 3

# 长音频分块推理长度。RMVPE 支持整段推理，但分块可降低显存/内存峰值并让 UI 有进度反馈。
PITCH_ANALYSIS_CHUNK_SEC = 30.0

# 分块重叠，单位秒；用于减轻块边缘伪影。
PITCH_ANALYSIS_OVERLAP_SEC = 1.0

# 默认权重下载地址。脚本不会自动下载，只在缺少模型时提示。
RMVPE_WEIGHT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"

CSV_COLUMNS = [
    "start_sec",
    "end_sec",
    "event_type",
    "note_names",
    "midi_pitches",
    "frequencies_hz",
    "rms",
]


# =========================
# 配置覆盖：与其他 pitch_extraction_*.py 保持一致
# =========================


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


# =========================
# 进度输出：由 progress_utils 统一格式化，可被 main.py / UI 解析
# =========================


# 输出线性进度
def emit_linear_progress(fraction: float, message: str):
    fraction = max(0.0, min(1.0, float(fraction)))
    emit_progress(
        stage="pitch_extraction_vocals",
        current=int(round(fraction * 10000)),
        total=10000,
        message=message,
        unit="percent",
    )



# 映射区间进度
def progress_between(start_fraction: float, end_fraction: float, local_fraction: float) -> float:
    local_fraction = max(0.0, min(1.0, float(local_fraction)))
    return start_fraction + (end_fraction - start_fraction) * local_fraction


# =========================
# RMVPE PyTorch 模型结构
# 说明：此处内嵌的是 RVC 常用 rmvpe.pt 对应的 PyTorch 推理结构，不需要 clone 仓库。
# =========================

N_MELS = 128
N_CLASS = 360
MEL_FMIN = 30
MEL_FMAX = 8000
WINDOW_LENGTH = 1024
CENTS_OFFSET = 1997.3794084376191


class BiGRU(nn.Module):
    # 初始化双向GRU模块
    def __init__(self, input_features: int, hidden_features: int, num_layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    # 执行双向GRU前向计算
    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    # 初始化残差卷积模块
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else None

    # 执行残差卷积前向计算
    def forward(self, x):
        if self.shortcut is None:
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    # 初始化残差编码模块
    def __init__(self, in_channels: int, out_channels: int, kernel_size, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    # 执行残差编码前向计算
    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    # 初始化编码器
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size,
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []

        for _ in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2

        self.out_size = in_size
        self.out_channel = out_channels

    # 执行编码器前向计算
    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            tensor, x = layer(x)
            concat_tensors.append(tensor)
        return x, concat_tensors


class Intermediate(nn.Module):
    # 初始化中间层模块
    def __init__(self, in_channels: int, out_channels: int, n_inters: int, n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    # 执行中间层前向计算
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    # 初始化残差解码模块
    def __init__(self, in_channels: int, out_channels: int, stride, n_blocks: int = 1, momentum: float = 0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    # 执行残差解码前向计算
    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        if x.shape[-2:] != concat_tensor.shape[-2:]:
            x = x[..., : concat_tensor.shape[-2], : concat_tensor.shape[-1]]
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    # 初始化解码器
    def __init__(self, in_channels: int, n_decoders: int, stride, n_blocks: int, momentum: float = 0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    # 执行解码器前向计算
    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    # 初始化DeepUnet网络
    def __init__(self, kernel_size, n_blocks: int, en_de_layers: int = 5, inter_layers: int = 4, in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    # 执行DeepUnet前向计算
    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    # 初始化端到端模型
    def __init__(self, n_blocks: int = 4, n_gru: int = 1, kernel_size=(2, 2), en_de_layers: int = 5, inter_layers: int = 4, in_channels: int = 1, en_out_channels: int = 16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, kernel_size=(3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    # 执行端到端模型前向计算
    def forward(self, mel):
        # mel: [B, n_mels, frames]
        mel = mel.transpose(-1, -2).unsqueeze(1)  # [B, 1, frames, n_mels]
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    # 初始化梅尔频谱模块
    def __init__(
        self,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: int | None = None,
        mel_fmin: float = 0.0,
        mel_fmax: float | None = None,
        clamp: float = 1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa_mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self.register_buffer("hann_window", torch.hann_window(win_length).float())

    # 计算梅尔频谱
    def forward(self, audio, center: bool = True):
        window = self.hann_window.to(device=audio.device, dtype=audio.dtype)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=center,
            return_complex=True,
        )
        magnitude = torch.abs(fft)
        mel_basis = self.mel_basis.to(device=audio.device, dtype=magnitude.dtype)
        mel_output = torch.matmul(mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPE:
    # 初始化RMVPE模型
    def __init__(self, model_path: str | Path, device: str | torch.device | None = None, is_half: bool = False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device) == "cuda":
            device = "cuda:0"

        self.device = torch.device(device)
        self.is_half = bool(is_half and self.device.type == "cuda")

        self.mel_extractor = MelSpectrogram(
            n_mel_channels=N_MELS,
            sampling_rate=SR,
            win_length=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            n_fft=None,
            mel_fmin=MEL_FMIN,
            mel_fmax=MEL_FMAX,
        ).to(self.device)

        self.model = E2E(4, 1, (2, 2))
        self._load_weights(model_path)
        self.model.eval()
        self.model.to(self.device)
        self.model.half() if self.is_half else self.model.float()

        cents_mapping = 20 * np.arange(N_CLASS, dtype=np.float64) + CENTS_OFFSET
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    # 加载RMVPE模型权重
    def _load_weights(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到 RMVPE 权重文件：{model_path}\n"
                f"请下载 rmvpe.pt：{RMVPE_WEIGHT_URL}\n"
                "推荐放到项目目录 models/rmvpe/rmvpe.pt，或用 --model 指定路径。"
            )

        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(model_path, map_location="cpu")
        except Exception:
            # 兼容旧版 PyTorch 或非 weights_only 可读的 checkpoint。
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break

        if not isinstance(ckpt, (dict, OrderedDict)):
            raise TypeError(f"无法识别的 RMVPE checkpoint 格式：{type(ckpt)!r}")

        state_dict = OrderedDict()
        for key, value in ckpt.items():
            new_key = str(key)
            for prefix in ("module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            state_dict[new_key] = value

        self.model.load_state_dict(state_dict, strict=True)

    # 将梅尔频谱转换为隐藏特征
    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")

        mel = mel.half() if self.is_half else mel.float()
        with torch.inference_mode():
            hidden = self.model(mel)
        return hidden[:, :n_frames]

    # 解码音高结果
    def decode(self, hidden: np.ndarray, thred: float = DEFAULT_RMVPE_THRESHOLD) -> np.ndarray:
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0.astype(np.float32)

    # 从音频推理音高
    def infer_from_audio(self, audio: np.ndarray, thred: float = DEFAULT_RMVPE_THRESHOLD) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)

        audio_tensor = torch.from_numpy(audio.astype(np.float32, copy=False)).to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio_tensor, center=True)
        hidden = self.mel2hidden(mel).squeeze(0).detach().cpu().numpy()
        if hidden.dtype != np.float32:
            hidden = hidden.astype(np.float32)
        return self.decode(hidden, thred=thred)

    # 计算局部平均音分
    def to_local_average_cents(self, salience: np.ndarray, thred: float = 0.05) -> np.ndarray:
        if salience.ndim != 2:
            raise ValueError("salience 必须是二维数组 [frames, bins]")

        center = np.argmax(salience, axis=1) + 4
        salience_padded = np.pad(salience, ((0, 0), (4, 4)))

        cents = np.zeros(salience.shape[0], dtype=np.float64)
        for idx in range(salience.shape[0]):
            start = center[idx] - 4
            end = center[idx] + 5
            local_salience = salience_padded[idx, start:end]
            weight_sum = float(np.sum(local_salience))
            if weight_sum <= 1e-12 or np.max(salience_padded[idx]) <= thred:
                cents[idx] = 0.0
            else:
                cents[idx] = float(np.sum(local_salience * self.cents_mapping[start:end]) / weight_sum)
        return cents


# =========================
# 音频、RMVPE 分块推理、音符切分与 CSV 输出
# =========================


# 解析模型权重路径
def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg:
        return Path(model_arg).expanduser()

    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "models" / "rmvpe" / "rmvpe.pt",
        base_dir / "models" / "rmvpe.pt",
        base_dir / "rmvpe.pt",
        Path.cwd() / "models" / "rmvpe" / "rmvpe.pt",
        Path.cwd() / "models" / "rmvpe.pt",
        Path.cwd() / "rmvpe.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]



# 加载人声音频
def load_audio(input_path: str | Path) -> np.ndarray:
    y, _ = librosa.load(input_path, sr=SR, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32, copy=False)



# 分块运行RMVPE推理
def run_rmvpe_in_chunks(rmvpe: RMVPE, y: np.ndarray, thred: float, start_progress: float = 0.30, end_progress: float = 0.70) -> np.ndarray:
    total_samples = len(y)
    if total_samples == 0:
        return np.zeros(0, dtype=np.float32)

    chunk_samples = max(int(float(PITCH_ANALYSIS_CHUNK_SEC) * SR), HOP_LENGTH * 64)
    overlap_samples = max(0, int(float(PITCH_ANALYSIS_OVERLAP_SEC) * SR))
    overlap_samples = min(overlap_samples, chunk_samples // 4)

    if total_samples <= chunk_samples:
        f0 = rmvpe.infer_from_audio(y, thred=thred)
        emit_linear_progress(end_progress, "运行 RMVPE 音高检测")
        return f0

    parts: list[np.ndarray] = []
    frame_cursor = 0
    step_samples = max(HOP_LENGTH, chunk_samples - overlap_samples * 2)
    chunk_index = 0

    for start_sample in range(0, total_samples, step_samples):
        chunk_index += 1
        end_sample = min(total_samples, start_sample + chunk_samples)
        chunk = y[start_sample:end_sample]
        f0_chunk = rmvpe.infer_from_audio(chunk, thred=thred)

        # 丢弃非首尾分块的重叠区域，按全局帧位置拼接，避免块边缘重复。
        chunk_start_frame_global = start_sample // HOP_LENGTH
        keep_start_frame_global = frame_cursor
        keep_end_frame_global = end_sample // HOP_LENGTH + 1

        if end_sample < total_samples:
            keep_end_frame_global = max(keep_start_frame_global, (end_sample - overlap_samples) // HOP_LENGTH)

        local_start = max(0, keep_start_frame_global - chunk_start_frame_global)
        local_end = max(local_start, keep_end_frame_global - chunk_start_frame_global)
        parts.append(f0_chunk[local_start:local_end])
        frame_cursor = keep_end_frame_global

        local_fraction = min(1.0, end_sample / total_samples)
        emit_linear_progress(
            progress_between(start_progress, end_progress, local_fraction),
            f"运行 RMVPE 音高检测（分块 {chunk_index}）",
        )

        if end_sample >= total_samples:
            break

    if not parts:
        return np.zeros(0, dtype=np.float32)

    expected_frames = total_samples // HOP_LENGTH + 1
    f0 = np.concatenate(parts).astype(np.float32, copy=False)
    if len(f0) > expected_frames:
        f0 = f0[:expected_frames]
    elif len(f0) < expected_frames:
        f0 = np.pad(f0, (0, expected_frames - len(f0)), mode="constant")
    return f0



# 确保滤波窗口为奇数
def ensure_odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size



# 执行一维中值滤波
def median_filter_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = ensure_odd_kernel(kernel_size)
    if kernel_size <= 1 or len(values) < kernel_size:
        return values.copy()

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    result = np.empty_like(values)
    for i in range(len(values)):
        result[i] = np.median(padded[i:i + kernel_size])
    return result



# 平滑MIDI音高轨迹
def smooth_midi_contour(f0_hz: np.ndarray) -> np.ndarray:
    midi = np.full(len(f0_hz), np.nan, dtype=np.float32)
    valid = f0_hz > 0
    midi[valid] = librosa.hz_to_midi(f0_hz[valid]).astype(np.float32)

    kernel_size = ensure_odd_kernel(SMOOTH_KERNEL_SIZE)
    if kernel_size <= 1:
        return midi

    starts = np.flatnonzero(np.diff(np.r_[False, valid, False]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, valid, False]) == -1)

    for start, end in zip(starts, ends):
        if end - start >= kernel_size:
            midi[start:end] = median_filter_1d(midi[start:end], kernel_size)

    return midi



# 填补短暂无声间隙
def fill_short_unvoiced_gaps(midi: np.ndarray) -> np.ndarray:
    result = midi.copy()
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
            gap_len <= int(MAX_UNVOICED_GAP_FRAMES)
            and left_idx >= 0
            and right_idx < n
            and not np.isnan(result[left_idx])
            and not np.isnan(result[right_idx])
            and abs(float(result[left_idx]) - float(result[right_idx])) <= PITCH_SPLIT_SEMITONES
        ):
            result[gap_start:gap_end] = (float(result[left_idx]) + float(result[right_idx])) / 2.0

    return result



# 检测起音位置
def detect_onsets(y: np.ndarray) -> set[int]:
    if len(y) == 0:
        return set()

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=SR,
        hop_length=HOP_LENGTH,
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=SR,
        hop_length=HOP_LENGTH,
        units="frames",
        pre_max=int(ONSET_PRE_MAX),
        post_max=int(ONSET_POST_MAX),
        pre_avg=int(ONSET_PRE_AVG),
        post_avg=int(ONSET_POST_AVG),
        delta=float(ONSET_DELTA),
        wait=int(ONSET_WAIT),
    )

    if len(onset_frames) <= 1:
        return set(onset_frames.tolist())

    filtered: list[int] = []
    min_gap = max(1, int(ONSET_MIN_GAP_FRAMES))
    for frame in onset_frames.tolist():
        if not filtered or frame - filtered[-1] >= min_gap:
            filtered.append(int(frame))

    return set(filtered)



# 判断是否按音高切分
def should_split_by_pitch(midi: np.ndarray, start: int, current: int) -> bool:
    if current - start < max(1, int(PITCH_HOLD_FRAMES)):
        return False

    if np.isnan(midi[current]):
        return True

    history = midi[start:current]
    history = history[~np.isnan(history)]
    if history.size == 0:
        return False

    base_pitch = float(np.median(history))
    if abs(float(midi[current]) - base_pitch) < float(PITCH_SPLIT_SEMITONES):
        return False

    confirm_end = min(len(midi), current + max(1, int(PITCH_HOLD_FRAMES)))
    future = midi[current:confirm_end]
    future = future[~np.isnan(future)]
    if len(future) < max(1, int(PITCH_HOLD_FRAMES)):
        return False

    return abs(float(np.median(future)) - base_pitch) >= float(PITCH_SPLIT_SEMITONES)



# 提取人声音符事件
def extract_note_events(f0_hz: np.ndarray, midi_contour: np.ndarray, y_audio: np.ndarray, onset_frames: set[int]) -> list[list]:
    events: list[list] = []
    total_duration = len(y_audio) / SR if SR > 0 else 0.0
    n = min(len(f0_hz), len(midi_contour))
    i = 0

    while i < n:
        if np.isnan(midi_contour[i]):
            i += 1
            continue

        start = i
        j = i + 1

        while j < n:
            if np.isnan(midi_contour[j]):
                break

            if j in onset_frames and (j - start) >= max(1, int(ONSET_MIN_GAP_FRAMES)):
                break

            if should_split_by_pitch(midi_contour, start, j):
                break

            j += 1

        end = j
        start_sec = start * HOP_LENGTH / SR
        end_sec = min(total_duration, end * HOP_LENGTH / SR)
        if end_sec <= start_sec:
            end_sec = min(total_duration, start_sec + HOP_LENGTH / SR)
        duration = end_sec - start_sec

        if duration < float(MIN_NOTE_SEC):
            i = j
            continue

        segment_f0 = f0_hz[start:end]
        segment_f0 = segment_f0[segment_f0 > 0]
        if segment_f0.size == 0:
            i = j
            continue

        median_f0 = float(np.median(segment_f0))
        if not math.isfinite(median_f0) or median_f0 <= 0:
            i = j
            continue

        midi_pitch = int(round(float(librosa.hz_to_midi(median_f0))))
        note_name = librosa.midi_to_note(midi_pitch, unicode=False)

        start_sample = max(0, int(round(start_sec * SR)))
        end_sample = min(len(y_audio), int(round(end_sec * SR)))
        event_rms = compute_segment_rms(y_audio[start_sample:end_sample])

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



# 将音符事件写入CSV
def write_events_to_csv(events: list[list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for event in events:
            start_sec, end_sec, event_type, note_names, midi_pitches, frequency_hz, event_rms = event
            writer.writerow([
                f"{start_sec:.2f}",
                f"{end_sec:.2f}",
                event_type,
                note_names,
                midi_pitches,
                f"{frequency_hz:.2f}",
                f"{max(0.0, float(event_rms)):.8f}",
            ])


# =========================
# CLI
# =========================


# 解析命令行参数
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 RMVPE 提取人声音高并输出项目统一 CSV。")
    parser.add_argument("--input", required=True, help="输入人声 wav/audio 文件")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    parser.add_argument("--config-json", type=Path, default=None, help="UI 传入的音高提取参数 JSON")
    parser.add_argument("--model", type=str, default=None, help="RMVPE 权重路径，默认查找 models/rmvpe/rmvpe.pt")
    parser.add_argument("--thred", "--threshold", dest="threshold", type=float, default=None, help="RMVPE salience 阈值，默认使用 DEFAULT_RMVPE_THRESHOLD")
    parser.add_argument("--device", type=str, default=None, help="推理设备，例如 cuda、cuda:0 或 cpu；默认自动选择")
    parser.add_argument("--half", action="store_true", help="CUDA 上使用 FP16 推理；CPU 会自动回退 FP32")
    return parser.parse_args(argv)



# 运行人声音高提取
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    pitch_overrides = load_config_overrides(args.config_json, "vocals")
    apply_config_overrides(pitch_overrides)

    threshold = float(DEFAULT_RMVPE_THRESHOLD if args.threshold is None else args.threshold)
    model_path = resolve_model_path(args.model)

    emit_linear_progress(0.02, "读取人声音频")
    y = load_audio(args.input)
    emit_linear_progress(0.05, f"人声音频读取完成：{len(y) / max(SR, 1):.2f} 秒")

    if y.size == 0:
        emit_linear_progress(0.97, "写入空 CSV")
        write_events_to_csv([], Path(args.output))
        emit_linear_progress(1.0, "人声音高识别完成")
        print(f"完成：{Path(args.output)}")
        return 0

    emit_linear_progress(0.08, "解析 RMVPE 参数")
    emit_linear_progress(0.12, "加载 RMVPE 模型")
    rmvpe = RMVPE(model_path=model_path, device=args.device, is_half=args.half)
    emit_linear_progress(0.28, "RMVPE 模型加载完成")

    emit_linear_progress(0.30, "运行 RMVPE 音高检测")
    f0 = run_rmvpe_in_chunks(rmvpe, y, threshold)

    emit_linear_progress(0.72, "转换 Hz 音高到 MIDI 轮廓")
    midi_contour = smooth_midi_contour(f0)
    emit_linear_progress(0.76, "平滑 MIDI 音高轮廓")
    midi_contour = fill_short_unvoiced_gaps(midi_contour)
    emit_linear_progress(0.79, "填补短暂无声间隙")

    emit_linear_progress(0.82, "检测人声起音")
    onset_frames = detect_onsets(y)
    emit_linear_progress(0.86, f"起音检测完成：{len(onset_frames)} 个候选")

    emit_linear_progress(0.90, "切分人声音符事件")
    events = extract_note_events(f0, midi_contour, y, onset_frames)
    emit_linear_progress(0.94, f"人声音符事件整理完成：{len(events)} 个事件")

    emit_linear_progress(0.97, "写入 CSV 文件")
    output_path = Path(args.output)
    write_events_to_csv(events, output_path)
    emit_linear_progress(0.99, "CSV 写入完成")

    emit_linear_progress(1.0, "人声音高识别完成")
    print(f"完成：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

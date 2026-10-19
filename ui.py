from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
import queue
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import chiptune
except Exception:  
    chiptune = None

try:
    import vlc  # type: ignore
except Exception:
    vlc = None

try:
    import winsound  # type: ignore
except Exception:
    winsound = None


PROGRESS_PREFIX = "PROGRESS "

CHANNEL_DISPLAY_NAMES = {
    "vocal": "人声",
    "guitar": "吉他",
    "piano": "钢琴",
    "bass": "贝斯",
    "other": "其他乐器",
}

DRUM_DISPLAY_NAMES = {
    "kick": "底鼓",
    "snare": "军鼓",
    "hh": "踩镲",
    "cymbals": "镲片",
    "toms": "嗵鼓",
}

WAVEFORM_OPTIONS = ["square", "triangle"]
CHANNEL_MODE_OPTIONS = ["mono", "stereo"]
NOISE_FILTER_OPTIONS = ["none", "lowpass", "highpass", "bandpass"]

# 测试音参数
CENTRAL_C_FREQUENCY_HZ = 261.625565
MELODIC_TEST_DURATION_SEC = 1.0
DRUM_TEST_DURATION_SEC = 0.2
CYMBALS_TEST_DURATION_SEC = 1.0
SYNTH_TEST_CACHE_VERSION = 2

# 进度
PHASE_RANGES = {
    "main_separator": (0.0, 100.0 / 6.0),
    "melodic_pitch": (100.0 / 6.0, 200.0 / 6.0),
    "drum_separator": (200.0 / 6.0, 300.0 / 6.0),
    "drum_pitch": (300.0 / 6.0, 400.0 / 6.0),
    "synthesis": (400.0 / 6.0, 500.0 / 6.0),
    "final_mix": (500.0 / 6.0, 100.0),
}

MAIN_MESSAGE_PHASE_FLOORS = {
    "开始处理": 0.0,
    "正在分离音频": PHASE_RANGES["main_separator"][0],
    "正在提取旋律声部音高": PHASE_RANGES["melodic_pitch"][0],
    "正在分离鼓组音频": PHASE_RANGES["drum_separator"][0],
    "正在提取鼓点": PHASE_RANGES["drum_pitch"][0],
    "正在合成芯片音轨": PHASE_RANGES["synthesis"][0],
    "正在混合最终音频": PHASE_RANGES["final_mix"][0],
    "处理完成": 100.0,
}

MELODIC_PROGRESS_ORDER = ["vocals", "guitar", "bass", "piano", "other"]
DRUM_PROGRESS_ORDER = ["cymbals", "hh", "snare", "kick", "toms"]
SYNTH_PROGRESS_ORDER = ["vocal", "guitar", "bass", "piano", "other", "cymbals", "hh", "snare", "kick", "toms"]

LIBRARY_PROGRESS_LINE_RE = re.compile(r"(\d{1,3}%\|)|(保存进度:)|(Processing audio chunks:)|(it/s\])")

PARAM_LABELS = {
    "sample_rate": "采样率",
    "master_volume": "主音量",
    "bit_depth": "最终总线位深",
    "max_attack_duration_sec": "Attack最长（秒）",
    "max_decay_duration_sec": "Decay最长（秒）",
    "max_release_duration_sec": "Release最长（秒）",
    "event_crossfade_sec": "事件交叉淡化时间（秒，0=关闭）",
    "event_crossfade_max_gap_sec": "允许交叉淡化的最大间隔（秒）",
    "channel_mode": "输出声道",
    "stereo_width": "立体声宽度",
    "pan_law_db": "中央声像衰减",
    "pan": "声像位置",
    "enabled": "启用RMS控制",
    "block_ms": "RMS块长度",
    "gate_db": "有效片段门限",
    "track_peak_db": "单轨峰值上限",
    "track_min_gain_db": "单轨最小增益",
    "track_max_gain_db": "单轨最大增益",
    "final_target_rms_db": "最终目标RMS",
    "final_peak_db": "最终峰值上限",
    "final_min_gain_db": "最终最小增益",
    "final_max_gain_db": "最终最大增益",
    "target_rms_db": "目标RMS",
    "waveform": "波形",
    "volume": "音量",
    "duty": "占空比",
    "attack": "Attack",
    "decay": "Decay",
    "sustain": "Sustain",
    "release": "Release",
    "arpeggio_interval": "快速琶音间隔",
    "steps": "阶梯数",
    "start_freq": "起始频率",
    "end_freq": "结束频率",
    "pitch_curve": "音高曲线",
    "triangle_mix": "三角波比例",
    "square_mix": "方波比例",
    "noise_mix": "噪声比例",
    "noise_filter": "噪声滤波",
    "noise_low": "噪声低频",
    "noise_high": "噪声高频",
}

GLOBAL_PARAM_ORDER = [
    "sample_rate", "master_volume", "bit_depth",
    "max_attack_duration_sec", "max_decay_duration_sec", "max_release_duration_sec",
]
OUTPUT_PARAM_ORDER = ["channel_mode", "stereo_width", "pan_law_db"]

RMS_PARAM_ORDER = [
    "enabled", "block_ms", "gate_db", "track_peak_db",
    "track_min_gain_db", "track_max_gain_db",
    "final_target_rms_db", "final_peak_db",
    "final_min_gain_db", "final_max_gain_db",
]

CHANNEL_PARAM_ORDER = [
    "pan", "waveform", "volume", "target_rms_db", "duty", "attack", "decay", "sustain",
    "release", "arpeggio_interval", "steps",
    "event_crossfade_sec", "event_crossfade_max_gap_sec",
]

DRUM_PARAM_ORDER = [
    "pan", "volume", "target_rms_db", "start_freq", "end_freq", "pitch_curve", "duty",
    "triangle_mix", "square_mix", "noise_mix", "noise_filter", "noise_low", "noise_high",
    "attack", "decay", "sustain", "release",
]

PITCH_CONFIG_DEFAULTS = {
    "vocals": {
        "DEFAULT_RMVPE_THRESHOLD": 0.03,
        "MIN_NOTE_SEC": 0.05,
        "PITCH_SPLIT_SEMITONES": 0.35,
        "PITCH_HOLD_FRAMES": 2,
        "ONSET_MIN_GAP_FRAMES": 4,
        "SMOOTH_KERNEL_SIZE": 3,
        "MAX_UNVOICED_GAP_FRAMES": 1,
        "ONSET_PRE_MAX": 3,
        "ONSET_POST_MAX": 3,
        "ONSET_PRE_AVG": 6,
        "ONSET_POST_AVG": 6,
        "ONSET_DELTA": 0.05,
        "ONSET_WAIT": 3,
    },
    "guitar": {
        "BASIC_PITCH_ONSET_THRESHOLD": 0.5,
        "BASIC_PITCH_FRAME_THRESHOLD": 0.3,
        "BASIC_PITCH_MINIMUM_NOTE_LENGTH": 127.70,
        "MIN_FREQUENCY_HZ": 82.41,
        "MAX_FREQUENCY_HZ": 1318.51,
        "MIN_NOTE_DURATION_SECONDS": 0.03,
        "ONSET_TOLERANCE_SECONDS": 0.05,
        "MIN_OVERLAP_RATIO": 0.05,
        "MAX_NOTES_PER_CHORD": 8,
    },
    "piano": {
        "BASIC_PITCH_ONSET_THRESHOLD": 0.5,
        "BASIC_PITCH_FRAME_THRESHOLD": 0.3,
        "BASIC_PITCH_MINIMUM_NOTE_LENGTH": 127.70,
        "MIN_FREQUENCY_HZ": 27.50,
        "MAX_FREQUENCY_HZ": 4186.01,
        "MIN_NOTE_DURATION_SECONDS": 0.03,
        "ONSET_TOLERANCE_SECONDS": 0.05,
        "MIN_OVERLAP_RATIO": 0.05,
        "MAX_NOTES_PER_CHORD": 8,
    },
    "bass": {
        "FRAME_LENGTH": 2048,
        "FMIN": 30.87,
        "FMAX": 392.00,
        "TRIM_TOP_DB": 70,
        "HARMONIC_MARGIN": 3.0,
        "SMOOTH_KERNEL_SIZE": 5,
        "MAX_GAP_FRAMES": 2,
        "MIN_NOTE_DURATION_SEC": 0.04,
        "PITCH_ANALYSIS_CHUNK_SEC": 30.0,
    },
    "other": {
        "FMIN": 80.0,
        "FMAX": 2000.0,
        "TRIM_TOP_DB": 70,
        "HARMONIC_MARGIN": 3.0,
        "TORCHCREPE_MODEL": "full",
        "TORCHCREPE_BATCH_SIZE": 2048,
        "PERIODICITY_MEDIAN_KERNEL": 3,
        "SILENCE_THRESHOLD_DB": -60.0,
        "PITCH_CONFIDENCE_THRESHOLD": 0.28,
        "PITCH_MEAN_FILTER_KERNEL": 3,
        "MIDI_SMOOTH_KERNEL_SIZE": 7,
        "MAX_UNVOICED_GAP_FRAMES": 3,
        "MIN_NOTE_SEC": 0.04,
        "ONSET_PRE_MAX": 4,
        "ONSET_POST_MAX": 4,
        "ONSET_PRE_AVG": 8,
        "ONSET_POST_AVG": 8,
        "ONSET_DELTA": 0.04,
        "ONSET_WAIT": 6,
        "MIN_FRAMES_BEFORE_ONSET_SPLIT": 8,
        "PITCH_CHANGE_SEMITONES": 1,
        "PITCH_CHANGE_LOOKAHEAD_FRAMES": 4,
        "PITCH_CHANGE_CONFIRM_FRAMES": 3,
        "MIN_MEAN_PERIODICITY": 0.35,
        "PITCH_ANALYSIS_CHUNK_SEC": 30.0,
    },
    "drums": {
        "HOP_LENGTH": 512,
        "ONSET_DELTA": 0.22,
        "MIN_ONSET_INTERVAL_SEC": 0.06,
        "ONSET_ENV_SMOOTH_FRAMES": 3,
        "RMS_GATE_ABS": 0.002,
        "RMS_GATE_RELATIVE_TO_P95": 0.15,
        "MIN_EVENT_DURATION_SEC": 0.03,
        "MAX_EVENT_DURATION_SEC": 1.20,
        "END_RELATIVE_THRESHOLD": 0.12,
        "END_ABSOLUTE_FLOOR": 1e-5,
        "END_SILENCE_FRAMES": 3,
        "MIN_EVENT_PEAK": 0.015,
        "MIN_EVENT_RMS": 0.002,
        "MIN_EVENT_RMS_RATIO_TO_TRACK_P95": 0.18,
        "PRE_EVENT_SEC": 0.05,
        "MIN_TRANSIENT_RMS_RATIO": 1.60,
        "PRE_RMS_MIN_FOR_TRANSIENT_CHECK": 0.001,
    },
}

PITCH_PARAM_LABELS = {
    "HOP_LENGTH": "帧移长度",
    "FRAME_LENGTH": "帧长度",
    "DEFAULT_RMVPE_THRESHOLD": "检测阈值",
    "BASIC_PITCH_ONSET_THRESHOLD": "起音阈值",
    "BASIC_PITCH_FRAME_THRESHOLD": "帧阈值",
    "BASIC_PITCH_MINIMUM_NOTE_LENGTH": "最短音符长度",
    "MIN_NOTE_SEC": "最短音符",
    "MIN_NOTE_DURATION_SEC": "最短音符",
    "MIN_NOTE_DURATION_SECONDS": "最短音符",
    "PITCH_ANALYSIS_CHUNK_SEC": "分块分析长度",
    "PITCH_SPLIT_SEMITONES": "音高切分阈值",
    "PITCH_HOLD_FRAMES": "音高保持帧",
    "SMOOTH_KERNEL_SIZE": "平滑窗口",
    "MIDI_SMOOTH_KERNEL_SIZE": "MIDI平滑窗口",
    "MAX_UNVOICED_GAP_FRAMES": "可补无声帧",
    "MAX_GAP_FRAMES": "可补间隙帧",
    "ONSET_TOLERANCE_SECONDS": "起音容差",
    "MIN_OVERLAP_RATIO": "最小重叠比例",
    "MAX_NOTES_PER_CHORD": "和弦最多音数",
    "MIN_FREQUENCY_HZ": "最低频率",
    "MAX_FREQUENCY_HZ": "最高频率",
    "FMIN": "最低频率",
    "FMAX": "最高频率",
    "TRIM_TOP_DB": "静音裁剪",
    "HARMONIC_MARGIN": "谐波增强",
    "ONSET_DELTA": "起音阈值",
    "ONSET_WAIT": "起音等待帧",
    "ONSET_MIN_GAP_FRAMES": "起音最小间隔",
    "ONSET_PRE_MAX": "起音前峰值窗",
    "ONSET_POST_MAX": "起音后峰值窗",
    "ONSET_PRE_AVG": "起音前均值窗",
    "ONSET_POST_AVG": "起音后均值窗",
    "TORCHCREPE_MODEL": "模型类型",
    "TORCHCREPE_BATCH_SIZE": "批大小",
    "PERIODICITY_MEDIAN_KERNEL": "周期性平滑",
    "SILENCE_THRESHOLD_DB": "静音阈值",
    "PITCH_CONFIDENCE_THRESHOLD": "音高置信度",
    "PITCH_MEAN_FILTER_KERNEL": "音高均值滤波",
    "MIN_FRAMES_BEFORE_ONSET_SPLIT": "切分前最短帧",
    "PITCH_CHANGE_SEMITONES": "音高变化阈值",
    "PITCH_CHANGE_LOOKAHEAD_FRAMES": "变化观察帧",
    "PITCH_CHANGE_CONFIRM_FRAMES": "变化确认帧",
    "MIN_MEAN_PERIODICITY": "最小周期性",
    "MIN_ONSET_INTERVAL_SEC": "鼓点最小间隔",
    "ONSET_ENV_SMOOTH_FRAMES": "起音包络平滑",
    "RMS_GATE_ABS": "RMS绝对门限",
    "RMS_GATE_RELATIVE_TO_P95": "RMS相对门限",
    "MIN_EVENT_DURATION_SEC": "最短事件",
    "MAX_EVENT_DURATION_SEC": "最长事件",
    "END_RELATIVE_THRESHOLD": "结束相对阈值",
    "END_ABSOLUTE_FLOOR": "结束绝对下限",
    "END_SILENCE_FRAMES": "结束静音帧",
    "MIN_EVENT_PEAK": "最小峰值",
    "MIN_EVENT_RMS": "最小RMS",
    "MIN_EVENT_RMS_RATIO_TO_TRACK_P95": "事件RMS比例",
    "PRE_EVENT_SEC": "前置背景时长",
    "MIN_TRANSIENT_RMS_RATIO": "瞬态RMS比例",
    "PRE_RMS_MIN_FOR_TRANSIENT_CHECK": "瞬态检查下限",
}

PARAM_DESCRIPTIONS = {
    "sample_rate": "输出合成音频的采样率，数值越高音频细节越多，但文件更大",
    "master_volume": "单轨合成阶段的总音量倍率",
    "bit_depth": "位深量化；空值或0表示关闭。",
    "max_attack_duration_sec": "Attack 按事件比例计算后允许的最长实际时间；0 表示不限制",
    "max_decay_duration_sec": "Decay 按事件比例计算后允许的最长实际时间；0 表示不限制",
    "max_release_duration_sec": "Release 按事件比例计算后允许的最长实际时间；0 表示不限制",
    "event_crossfade_sec": "相邻事件使用的等功率交叉淡化重叠时间；0 表示关闭",
    "event_crossfade_max_gap_sec": "仅当相邻事件原始间隔不超过该值时才建立交叉淡化",
    "channel_mode": "最终输出声道：mono 为单声道，stereo 为双声道",
    "stereo_width": "双声道宽度：0 为左右相同，1 为标准宽度，2 为增强宽度",
    "pan_law_db": "声像居中时左右声道各自的衰减，通常使用 -3 dB",
    "pan": "双声道声像位置：-1 为最左，0 为中央，1 为最右；单声道模式下忽略",
    "enabled": "是否启用单轨与最终总线的RMS响度控制",
    "block_ms": "计算单轨有效RMS时的分析块长度，单位毫秒",
    "gate_db": "低于该dBFS门限的块不参与单轨RMS计算，避免静音导致过度放大",
    "track_peak_db": "单条中间音轨允许的最高峰值，单位dBFS",
    "track_min_gain_db": "单轨RMS控制允许的最大衰减量，单位dB",
    "track_max_gain_db": "单轨RMS控制允许的最大提升量，单位dB",
    "final_target_rms_db": "最终混音希望达到的整体RMS，单位dBFS",
    "final_peak_db": "最终输出允许的最高峰值，单位dBFS",
    "final_min_gain_db": "最终RMS控制允许的最大衰减量，单位dB",
    "final_max_gain_db": "最终RMS控制允许的最大提升量，单位dB",
    "target_rms_db": "当前声部有效片段的目标RMS，单位dBFS",
    "waveform": "使用的基础波形",
    "volume": "基础音量",
    "duty": "方波占空比",
    "attack": "包络Attack阶段比例，控制起音从无到有的速度",
    "decay": "包络Decay阶段比例，控制起音后衰减到Sustain的速度",
    "sustain": "包络Sustain音量比例，控制持续阶段音量",
    "release": "包络Release阶段比例，控制音符结束时的衰减速度",
    "arpeggio_interval": "和弦快速琶音间隔，0表示普通和弦",
    "steps": "三角波量化阶数，数值越小颗粒感越强",
    "start_freq": "鼓点音高滑动的起始频率",
    "end_freq": "鼓点音高滑动的结束频率",
    "pitch_curve": "鼓点频率滑动曲线，小于1前段变化更快，大于1后段变化更明显。",
    "triangle_mix": "鼓件中三角波成分比例",
    "square_mix": "鼓件中方波成分比例",
    "noise_mix": "鼓件中噪声成分比例",
    "noise_filter": "鼓件噪声滤波类型：none、lowpass、highpass、bandpass。",
    "noise_low": "噪声滤波低频边界，空值表示不限制",
    "noise_high": "噪声滤波高频边界，空值表示不限制",
}

PITCH_PARAM_DESCRIPTIONS = {
    "HOP_LENGTH": "分析帧移长度，数值越小时间精度越高，但处理更慢",
    "FRAME_LENGTH": "分析帧长度，数值越大频率更稳定，但响应更慢",
    "DEFAULT_RMVPE_THRESHOLD": "人声音高检测阈值，越低越灵敏，越高越保守",
    "PITCH_SPLIT_SEMITONES": "人声音高变化达到该半音数时切分音符；越小越容易切分颤音和滑音",
    "PITCH_HOLD_FRAMES": "音高变化需连续保持的帧数；越大越能抑制短暂抖动，但快速换音响应更慢",
    "BASIC_PITCH_ONSET_THRESHOLD": "Basic Pitch起音阈值，越低越容易保留弱起音",
    "BASIC_PITCH_FRAME_THRESHOLD": "Basic Pitch帧阈值，影响音符持续区域的保留",
    "BASIC_PITCH_MINIMUM_NOTE_LENGTH": "Basic Pitch内部最短音符长度，单位毫秒",
    "MIN_NOTE_SEC": "最短音符时长，短于该值的片段会被过滤",
    "MIN_NOTE_DURATION_SEC": "最短音符时长，短于该值的片段会被过滤",
    "MIN_NOTE_DURATION_SECONDS": "二次过滤的最短音符时长，单位秒",
    "PITCH_ANALYSIS_CHUNK_SEC": "长音频分块分析长度，单位秒；越小单块占用更低、进度更新更频繁，但分块边界更多",
    "MAX_NOTES_PER_CHORD": "单个和弦最多保留的音符数量；0 表示不限制",
    "ONSET_TOLERANCE_SECONDS": "和弦聚合起始时间容差",
    "MIN_OVERLAP_RATIO": "判定同一和弦时要求的最小重叠比例",
    "MIN_FREQUENCY_HZ": "保留的最低频率",
    "MAX_FREQUENCY_HZ": "保留的最高频率",
    "FMIN": "检测最低频率",
    "FMAX": "检测最高频率",
    "TRIM_TOP_DB": "静音裁剪阈值，数值越大保留弱音越多",
    "HARMONIC_MARGIN": "谐波增强强度，用于削弱噪声和打击成分",
    "SMOOTH_KERNEL_SIZE": "音高平滑窗口大小，通常为奇数",
    "MIDI_SMOOTH_KERNEL_SIZE": "MIDI音高轨迹平滑窗口大小",
    "MAX_UNVOICED_GAP_FRAMES": "允许填补的短暂无声间隙帧数",
    "MAX_GAP_FRAMES": "允许填补的短暂无声间隙帧数",
    "ONSET_DELTA": "起音检测阈值，越低越灵敏",
    "ONSET_WAIT": "相邻起音之间的最小等待帧数",
    "ONSET_MIN_GAP_FRAMES": "人声切分时允许起音切分的最小间隔",
    "ONSET_PRE_MAX": "起音峰值检测的前向窗口",
    "ONSET_POST_MAX": "起音峰值检测的后向窗口",
    "ONSET_PRE_AVG": "起音局部均值的前向窗口",
    "ONSET_POST_AVG": "起音局部均值的后向窗口",
    "TORCHCREPE_MODEL": "torchcrepe模型类型，full更准但更慢，tiny更快",
    "TORCHCREPE_BATCH_SIZE": "torchcrepe推理批大小",
    "PERIODICITY_MEDIAN_KERNEL": "周期性置信度中值滤波窗口",
    "SILENCE_THRESHOLD_DB": "低于该能量的区域会被认为不可靠",
    "PITCH_CONFIDENCE_THRESHOLD": "音高置信度阈值，越高越保守",
    "PITCH_MEAN_FILTER_KERNEL": "音高均值滤波窗口",
    "MIN_FRAMES_BEFORE_ONSET_SPLIT": "遇到新起音前当前音符至少持续的帧数",
    "PITCH_CHANGE_SEMITONES": "音高变化切分阈值，单位半音",
    "PITCH_CHANGE_LOOKAHEAD_FRAMES": "判断音高变化是否稳定时向后观察的帧数",
    "PITCH_CHANGE_CONFIRM_FRAMES": "确认音高变化所需的最少有效帧数",
    "MIN_MEAN_PERIODICITY": "音符片段最低平均周期性",
    "MIN_ONSET_INTERVAL_SEC": "两个鼓点之间的最小间隔",
    "ONSET_ENV_SMOOTH_FRAMES": "鼓点起音包络平滑帧数",
    "RMS_GATE_ABS": "鼓点检测前的RMS绝对门限",
    "RMS_GATE_RELATIVE_TO_P95": "鼓点检测前相对于整轨能量的RMS门限",
    "MIN_EVENT_DURATION_SEC": "鼓点事件最短持续时间",
    "MAX_EVENT_DURATION_SEC": "鼓点事件最长持续时间",
    "END_RELATIVE_THRESHOLD": "判断鼓点结束的相对能量阈值",
    "END_ABSOLUTE_FLOOR": "判断鼓点结束的绝对能量下限",
    "END_SILENCE_FRAMES": "连续多少帧低于结束阈值后确认结束",
    "MIN_EVENT_PEAK": "保留鼓点事件所需的最小峰值",
    "MIN_EVENT_RMS": "保留鼓点事件所需的最小 RMS",
    "MIN_EVENT_RMS_RATIO_TO_TRACK_P95": "事件RMS相对整轨有效能量的最低比例",
    "PRE_EVENT_SEC": "起音前用于计算背景能量的时间长度",
    "MIN_TRANSIENT_RMS_RATIO": "事件RMS至少要比前景背景RMS高出的倍数",
    "PRE_RMS_MIN_FOR_TRANSIENT_CHECK": "前景背景RMS太小时不启用瞬态比判断",
}


class ToolTip:
    # 初始化工具提示
    def __init__(self, widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.after_id = None
        self.window = None

        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)

    # 安排显示工具提示
    def schedule(self, _event=None):
        self.cancel()
        self.after_id = self.widget.after(self.delay_ms, self.show)

    # 取消工具提示任务
    def cancel(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    # 显示工具提示
    def show(self):
        if not self.text or self.window is not None:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.window,
            text=self.text,
            justify=tk.LEFT,
            wraplength=360,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padding=(8, 5),
        )
        label.pack()

    # 隐藏工具提示
    def hide(self, _event=None):
        self.cancel()
        if self.window is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None



class ScrollableFrame(ttk.Frame):
    # 初始化可滚动容器
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<Configure>", self._resize_inner)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    # 调整滚动区域宽度
    def _resize_inner(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    # 处理鼠标滚轮事件
    def _on_mousewheel(self, event):
        if self.winfo_containing(event.x_root, event.y_root) is None:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class ChiptuneUI(tk.Tk):
    # 初始化主界面
    def __init__(self):
        super().__init__()

        self.title("Chiptune Audio Processor")
        self.geometry("980x760")
        self.minsize(860, 600)

        self.process: subprocess.Popen | None = None
        self.worker_thread: threading.Thread | None = None
        self.input_player_process: subprocess.Popen | None = None
        self.output_player_process: subprocess.Popen | None = None
        self.synth_test_player_process: subprocess.Popen | None = None
        self.player_buttons: dict[str, ttk.Button] = {}
        self.player_progress_vars: dict[str, tk.DoubleVar] = {}
        self.player_time_vars: dict[str, tk.StringVar] = {}
        self.player_durations: dict[str, float] = {}
        self.player_started_at: dict[str, float] = {}
        self.player_elapsed_before_pause: dict[str, float] = {}
        self.player_paused: dict[str, bool] = {}
        self.player_volume_vars: dict[str, tk.DoubleVar] = {}
        self.player_seek_scales: dict[str, ttk.Scale] = {}
        self.player_is_seeking: dict[str, bool] = {}
        self.player_seek_after_ids: dict[str, str | None] = {}
        self.player_source_signatures: dict[str, tuple[str, int, int] | None] = {}
        self.player_ready: dict[str, bool] = {"input": True, "output": True}
        self.player_applied_volume: dict[str, int] = {}
        self.volume_restart_after_ids: dict[str, str | None] = {}

        self.ui_event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.duration_request_tokens: dict[str, int] = {"input": 0, "output": 0}
        self.synth_test_task_token = 0
        self.synth_test_buttons: dict[tuple[str, ...], ttk.Button] = {}
        self.synth_test_generation_lock = threading.Lock()
        self._closing = False

        self.vlc_instance = self.create_vlc_instance()
        self.vlc_players: dict[str, object | None] = {}
        self.vlc_active_media_signatures: dict[str, tuple[str, int, int] | None] = {}

        base_dir = Path(__file__).resolve().parent

        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar(value=str(base_dir / "output" / "final_chiptune.wav"))
        self.work_dir_var = tk.StringVar(value=str(base_dir / "output" / "cache"))

        self.overall_progress_var = tk.DoubleVar(value=0.0)
        self.current_progress_var = tk.DoubleVar(value=0.0)
        self.overall_text_var = tk.StringVar(value="总进度：0.00%")
        self.current_text_var = tk.StringVar(value="当前任务：等待开始")
        self.status_var = tk.StringVar(value="请选择音频文件")
        self.log_button_text_var = tk.StringVar(value="显示日志")

        self.default_config = self.load_default_config()
        self.pitch_config_defaults = copy.deepcopy(PITCH_CONFIG_DEFAULTS)
        self.config_vars: dict[tuple[str, ...], tk.StringVar] = {}
        self.config_original_values: dict[tuple[str, ...], object] = {}
        self.pitch_config_vars: dict[tuple[str, ...], tk.StringVar] = {}
        self.pitch_config_original_values: dict[tuple[str, ...], object] = {}
        self.parameter_widgets: list[tuple[tk.Widget, str]] = []
        self.stereo_parameter_widgets: list[tuple[tk.Widget, str]] = []
        self.mono_parameter_frame: ttk.Widget | None = None
        self.stereo_parameter_frame: ttk.Widget | None = None
        self.processing_running = False
        self.processing_started_at = 0.0

        self.log_visible = False
        self.log_panel_height = 200
        self.window_height_before_log: int | None = None
        self.current_bar_is_indeterminate = False
        self.max_overall_percent = 0.0
        self.main_progress_message = ""
        self.last_progress_log: dict[str, tuple[float, float]] = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(40, self.process_ui_event_queue)

    # 构建主界面
    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        self.main_area = ttk.Frame(root)
        self.main_area.pack(fill=tk.BOTH, expand=True)

        self.left_area = ttk.Frame(self.main_area)
        self.left_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_area = ttk.Frame(self.main_area, width=430)
        self.right_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        self.right_area.pack_propagate(False)

        file_frame = ttk.LabelFrame(self.left_area, text="文件设置", padding=10)
        file_frame.pack(fill=tk.X)

        self._add_path_row(file_frame, 0, "输入音频", self.input_path_var, "选择音频...", self.choose_input_file)
        self._add_path_row(file_frame, 1, "输出文件", self.output_path_var, "选择输出...", self.choose_output_file)
        self._add_path_row(file_frame, 2, "缓存目录", self.work_dir_var, "选择目录...", self.choose_work_dir)
        file_frame.columnconfigure(1, weight=1)

        self._build_audio_preview(self.left_area)

        control_frame = ttk.Frame(self.left_area, padding=(0, 10, 0, 0))
        control_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(control_frame, text="开始处理", command=self.start_process)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))

        self.reset_params_button = ttk.Button(control_frame, text="重置参数", command=self.reset_synth_params)
        self.reset_params_button.pack(side=tk.LEFT, padx=(8, 0))

        self.log_toggle_button = ttk.Button(
            control_frame,
            textvariable=self.log_button_text_var,
            command=self.toggle_log_panel,
        )
        self.log_toggle_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(16, 0))

        progress_frame = ttk.LabelFrame(self.left_area, text="进度", padding=10)
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(progress_frame, textvariable=self.overall_text_var).pack(anchor=tk.W)
        self.overall_bar = ttk.Progressbar(progress_frame, maximum=100, variable=self.overall_progress_var)
        self.overall_bar.pack(fill=tk.X, pady=(4, 10))

        ttk.Label(progress_frame, textvariable=self.current_text_var).pack(anchor=tk.W)
        self.current_bar = ttk.Progressbar(progress_frame, maximum=100, variable=self.current_progress_var)
        self.current_bar.pack(fill=tk.X, pady=(4, 0))

        self._build_parameter_panel(self.right_area)
        self._build_log_panel(self.left_area)

    # 构建参数面板
    def _build_parameter_panel(self, root):
        self.param_frame = ttk.LabelFrame(root, text="参数", padding=10)
        self.param_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        notebook = ttk.Notebook(self.param_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        global_tab = ScrollableFrame(notebook)
        output_tab = ScrollableFrame(notebook)
        melodic_tab = ScrollableFrame(notebook)
        drums_tab = ScrollableFrame(notebook)
        pitch_tab = ScrollableFrame(notebook)

        notebook.add(global_tab, text="全局")
        notebook.add(output_tab, text="声道设置")
        notebook.add(pitch_tab, text="音高提取参数")
        notebook.add(melodic_tab, text="旋律合成参数")
        notebook.add(drums_tab, text="鼓组合成参数")

        self._add_config_section(
            parent=global_tab.inner,
            title="全局参数",
            base_path=(),
            values={
                "sample_rate": self.default_config.get("sample_rate"),
                "master_volume": self.default_config.get("master_volume"),
                "bit_depth": self.default_config.get("bit_depth"),
            },
        )

        self._build_output_channel_panel(output_tab.inner)

        channels = self.default_config.get("channels", {})
        for channel, values in channels.items():
            display = CHANNEL_DISPLAY_NAMES.get(channel, channel)
            synth_values = {
                key: value
                for key, value in values.items()
                if key not in {"pan", "target_rms_db"}
                and not (key == "arpeggio_interval" and channel not in {"guitar", "piano"})
            }
            self._add_config_section(
                parent=melodic_tab.inner,
                title=f"{display}（{channel}）",
                base_path=("channels", channel),
                values=synth_values,
            )

        self._add_config_section(
            parent=melodic_tab.inner,
            title="ADR包络最长时间",
            base_path=(),
            values={
                "max_attack_duration_sec": self.default_config.get("max_attack_duration_sec", 0.05),
                "max_decay_duration_sec": self.default_config.get("max_decay_duration_sec", 0.05),
                "max_release_duration_sec": self.default_config.get("max_release_duration_sec", 0.05),
            },
        )

        drums = self.default_config.get("drums", {})
        for drum, values in drums.items():
            display = DRUM_DISPLAY_NAMES.get(drum, drum)
            synth_values = {
                key: value
                for key, value in values.items()
                if key not in {"pan", "target_rms_db"}
            }
            self._add_config_section(
                parent=drums_tab.inner,
                title=f"{display}（{drum}）",
                base_path=("drums", drum),
                values=synth_values,
            )

        pitch_titles = {
            "vocals": "人声音高提取",
            "guitar": "吉他音高识别",
            "piano": "钢琴音高识别",
            "bass": "贝斯音高识别",
            "other": "其他乐器音高识别",
            "drums": "鼓点提取",
        }
        for section, values in self.pitch_config_defaults.items():
            self._add_pitch_config_section(
                parent=pitch_tab.inner,
                title=pitch_titles.get(section, section),
                section=section,
                values=values,
            )

        self.update_stereo_parameter_state()

    # 构建输出声道面板
    def _build_output_channel_panel(self, parent):
        output_values = self.default_config.get("output", {})

        self._add_config_section(
            parent=parent,
            title="输出模式",
            base_path=("output",),
            values={
                "channel_mode": output_values.get("channel_mode", "mono"),
            },
        )

        mode_content = ttk.Frame(parent)
        mode_content.pack(fill=tk.X, padx=2, pady=(0, 4))

        self.mono_parameter_frame = ttk.LabelFrame(
            mode_content,
            text="单声道参数",
            padding=10,
        )
        ttk.Label(
            self.mono_parameter_frame,
            text="所有声部将居中合并为单声道；立体声宽度、中央衰减和各声部声像不参与处理。",
            wraplength=360,
            justify=tk.LEFT,
        ).pack(fill=tk.X)

        self.stereo_parameter_frame = ttk.Frame(mode_content)

        stereo_values = {
            key: output_values[key]
            for key in ("stereo_width", "pan_law_db")
            if key in output_values
        }
        self._add_config_section(
            parent=self.stereo_parameter_frame,
            title="双声道全局参数",
            base_path=("output",),
            values=stereo_values,
        )

        channel_pan_items = []
        for channel, values in self.default_config.get("channels", {}).items():
            if "pan" not in values:
                continue
            channel_pan_items.append(
                (
                    CHANNEL_DISPLAY_NAMES.get(channel, channel),
                    ("channels", channel, "pan"),
                    values["pan"],
                )
            )
        self._add_pan_parameter_section(
            parent=self.stereo_parameter_frame,
            title="旋律声部声像",
            items=channel_pan_items,
        )

        drum_pan_items = []
        for drum, values in self.default_config.get("drums", {}).items():
            if "pan" not in values:
                continue
            drum_pan_items.append(
                (
                    DRUM_DISPLAY_NAMES.get(drum, drum),
                    ("drums", drum, "pan"),
                    values["pan"],
                )
            )
        self._add_pan_parameter_section(
            parent=self.stereo_parameter_frame,
            title="鼓组声像",
            items=drum_pan_items,
        )

    # 添加声像参数区域
    def _add_pan_parameter_section(
        self,
        parent,
        title: str,
        items: list[tuple[str, tuple[str, ...], object]],
    ):
        if not items:
            return None

        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, padx=2, pady=4)
        frame.columnconfigure(1, weight=1)

        description = PARAM_DESCRIPTIONS.get("pan", "")
        for row, (display_name, path, value) in enumerate(items):
            label_widget = ttk.Label(
                frame,
                text=display_name,
                width=12,
                anchor=tk.W,
            )
            label_widget.grid(
                row=row,
                column=0,
                sticky=tk.W,
                padx=(0, 4),
                pady=2,
            )
            if description:
                ToolTip(label_widget, description)

            var = tk.StringVar(value=self.value_to_text(value))
            self.config_vars[path] = var
            self.config_original_values[path] = value

            widget = ttk.Entry(frame, textvariable=var, width=9)
            widget.grid(
                row=row,
                column=1,
                sticky=tk.EW,
                padx=(0, 2),
                pady=2,
            )
            if description:
                ToolTip(widget, description)

            self.parameter_widgets.append((widget, "normal"))
            self.stereo_parameter_widgets.append((widget, "normal"))

        return frame

    # 构建日志面板
    def _build_log_panel(self, root):
        self.log_frame = ttk.LabelFrame(root, text="日志", padding=10)
        self.log_frame.configure(height=self.log_panel_height)
        self.log_frame.pack_propagate(False)

        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    # 构建音频预览区域
    def _build_audio_preview(self, root):
        preview_frame = ttk.LabelFrame(root, text="音频预览", padding=10)
        preview_frame.pack(fill=tk.X, pady=(10, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)

        self._add_audio_player(preview_frame, 0, "输入音频", "input")
        self._add_audio_player(preview_frame, 1, "输出音频", "output")

    # 添加音频播放器
    def _add_audio_player(self, parent, column: int, title: str, kind: str):
        frame = ttk.Frame(parent, padding=(0, 0, 12 if column == 0 else 0, 0))
        frame.grid(row=0, column=column, sticky=tk.EW, padx=(0, 8 if column == 0 else 0))
        frame.columnconfigure(2, weight=1)

        self.player_progress_vars[kind] = tk.DoubleVar(value=0.0)
        self.player_time_vars[kind] = tk.StringVar(value="00:00 / 00:00")
        self.player_volume_vars[kind] = tk.DoubleVar(value=80.0)
        self.player_durations[kind] = 0.0
        self.player_started_at[kind] = 0.0
        self.player_elapsed_before_pause[kind] = 0.0
        self.player_paused[kind] = False
        self.player_is_seeking[kind] = False
        self.player_seek_after_ids[kind] = None
        self.player_source_signatures[kind] = None
        self.player_applied_volume[kind] = int(max(0, min(100, self.player_volume_vars[kind].get())))
        self.volume_restart_after_ids[kind] = None
        self.vlc_players[kind] = None
        self.vlc_active_media_signatures[kind] = None

        ttk.Label(frame, text=title).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))

        button = ttk.Button(frame, text="播放", command=lambda: self.toggle_audio(kind))
        button.grid(row=0, column=1, sticky=tk.W, padx=(0, 8))
        self.player_buttons[kind] = button

        ttk.Label(frame, textvariable=self.player_time_vars[kind]).grid(row=0, column=2, sticky=tk.E)

        progress = ttk.Scale(
            frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.player_progress_vars[kind],
            command=lambda value, player_kind=kind: self.on_seek_preview(player_kind, value),
        )
        progress.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))
        progress.bind("<ButtonPress-1>", lambda _event, player_kind=kind: self.begin_seek(player_kind))
        progress.bind("<ButtonRelease-1>", lambda _event, player_kind=kind: self.end_seek(player_kind))
        self.player_seek_scales[kind] = progress

        volume_frame = ttk.Frame(frame)
        volume_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(6, 0))
        volume_frame.columnconfigure(1, weight=1)

        ttk.Label(volume_frame, text="音量").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        volume_scale = ttk.Scale(
            volume_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.player_volume_vars[kind],
            command=lambda _value, player_kind=kind: self.on_volume_change(player_kind),
        )
        volume_scale.grid(row=0, column=1, sticky=tk.EW)

    # 获取播放器音频路径
    def get_audio_path_for_player(self, kind: str) -> Path:
        if kind == "input":
            return Path(self.input_path_var.get().strip())
        if kind == "output":
            return Path(self.output_path_var.get().strip())
        raise ValueError(f"未知播放器类型：{kind}")

    # 获取播放器进程
    def get_player_process(self, kind: str) -> subprocess.Popen | None:
        if kind == "input":
            return self.input_player_process
        if kind == "output":
            return self.output_player_process
        return None

    # 设置播放器进程
    def set_player_process(self, kind: str, process: subprocess.Popen | None):
        if kind == "input":
            self.input_player_process = process
        elif kind == "output":
            self.output_player_process = process

    # 发送界面事件
    def post_ui_event(self, event_type: str, payload=None) -> None:
        if self._closing:
            return
        self.ui_event_queue.put((event_type, payload))

    # 发送界面回调
    def post_ui_callback(self, callback, *args) -> None:
        self.post_ui_event("callback", (callback, args))

    # 在后台执行任务
    def run_in_background(self, worker, on_success=None, on_error=None) -> None:
        # 执行后台任务
        def target():
            try:
                result = worker()
            except Exception as exc:
                if on_error is not None:
                    self.post_ui_callback(on_error, exc)
                return
            if on_success is not None:
                self.post_ui_callback(on_success, result)

        threading.Thread(target=target, daemon=True).start()

    # 处理界面事件队列
    def process_ui_event_queue(self):
        if self._closing:
            return

        log_parts: list[str] = []
        latest_progress: dict[str, dict] = {}
        callbacks: list[tuple[object, tuple]] = []

        # 每轮限制处理数量，避免子进程瞬间输出很多内容时饿死鼠标和键盘事件。
        for _ in range(500):
            try:
                event_type, payload = self.ui_event_queue.get_nowait()
            except queue.Empty:
                break

            if event_type == "log":
                log_parts.append(str(payload))
            elif event_type == "progress" and isinstance(payload, dict):
                latest_progress[str(payload.get("stage", "unknown"))] = payload
            elif event_type == "callback" and isinstance(payload, tuple):
                callback, args = payload
                callbacks.append((callback, tuple(args)))

        if log_parts:
            self.append_log("".join(log_parts))

        for progress_payload in latest_progress.values():
            self.update_progress_from_payload(progress_payload)

        for callback, args in callbacks:
            try:
                callback(*args)
            except Exception as exc:
                self.append_log(f"\nUI 回调异常：{exc}\n")

        delay = 10 if not self.ui_event_queue.empty() else 40
        self.after(delay, self.process_ui_event_queue)

    # 获取文件签名
    @staticmethod
    def get_file_signature(path: Path) -> tuple[str, int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))

    # 请求音频时长
    def request_player_duration(
        self,
        kind: str,
        *,
        reset_position: bool = False,
        expected_signature: tuple[str, int, int] | None = None,
    ) -> None:
        path = self.get_audio_path_for_player(kind)
        signature = expected_signature or self.get_file_signature(path)
        self.duration_request_tokens[kind] = self.duration_request_tokens.get(kind, 0) + 1
        token = self.duration_request_tokens[kind]

        if signature is None:
            self.player_durations[kind] = 0.0
            if reset_position:
                self.reset_player_ui(kind)
            return

        if reset_position:
            self.player_durations[kind] = 0.0
            self.reset_player_ui(kind)

        # 读取音频时长
        def worker():
            return self.get_audio_duration(path)

        # 应用音频时长
        def apply_duration(duration: float):
            if token != self.duration_request_tokens.get(kind):
                return
            if self.get_file_signature(path) != signature:
                return
            self.player_durations[kind] = max(0.0, float(duration))
            elapsed = self.get_audio_elapsed(kind)
            total = self.player_durations[kind]
            percent = max(0.0, min(100.0, elapsed / total * 100.0)) if total > 0 else 0.0
            if not self.player_is_seeking.get(kind, False):
                self.player_progress_vars[kind].set(percent)
            self.player_time_vars[kind].set(
                f"{self.format_time(elapsed)} / {self.format_time(total)}"
            )

        self.run_in_background(worker, on_success=apply_duration)

    # 刷新播放器音频源
    def refresh_player_source(self, kind: str, *, reset_position: bool = True) -> bool:
        path = self.get_audio_path_for_player(kind)
        signature = self.get_file_signature(path)
        self.player_source_signatures[kind] = signature

        if signature is None:
            self.duration_request_tokens[kind] = self.duration_request_tokens.get(kind, 0) + 1
            self.player_durations[kind] = 0.0
            if reset_position:
                self.reset_player_ui(kind)
            else:
                self.player_time_vars[kind].set("00:00 / 00:00")
            return False

        self.request_player_duration(
            kind,
            reset_position=reset_position,
            expected_signature=signature,
        )
        return True

    # 确保播放器音频源最新
    def ensure_player_source_fresh(self, kind: str, *, reset_if_changed: bool = True) -> bool:
        path = self.get_audio_path_for_player(kind)
        signature = self.get_file_signature(path)
        if signature is None:
            return False

        changed = signature != self.player_source_signatures.get(kind)
        if changed:
            self.player_source_signatures[kind] = signature
            self.request_player_duration(
                kind,
                reset_position=reset_if_changed,
                expected_signature=signature,
            )
        elif self.player_durations.get(kind, 0.0) <= 0.0:
            self.request_player_duration(
                kind,
                reset_position=False,
                expected_signature=signature,
            )
        return True

    # 设置播放器就绪状态
    def set_player_ready(self, kind: str, ready: bool):
        self.player_ready[kind] = bool(ready)
        button = self.player_buttons.get(kind)
        if button is not None:
            button.configure(state=tk.NORMAL if ready else tk.DISABLED)

    # 准备播放器开始新任务
    def prepare_players_for_new_processing(self, input_path: Path, output_path: Path):
        self.stop_audio("output")
        self.player_source_signatures["output"] = None
        self.player_durations["output"] = 0.0
        self.reset_player_ui("output")
        self.set_player_ready("output", False)

        self.player_source_signatures["input"] = self.get_file_signature(input_path)
        if self.player_source_signatures["input"] is not None:
            self.request_player_duration(
                "input",
                reset_position=self.get_player_process("input") is None,
                expected_signature=self.player_source_signatures["input"],
            )

    # 标记输出播放器完成
    def mark_output_player_finished(self):
        self.set_player_ready("output", True)
        self.refresh_player_source("output", reset_position=True)

    # 创建VLC实例
    def create_vlc_instance(self):
        if vlc is None:
            return None

        try:
            return vlc.Instance("--quiet", "--no-video")
        except Exception:
            return None

    # 检查VLC播放后端
    def has_vlc_backend(self) -> bool:
        return self.vlc_instance is not None

    # 获取VLC播放器
    def get_vlc_player(self, kind: str):
        return self.vlc_players.get(kind)

    # 设置VLC播放器
    def set_vlc_player(self, kind: str, player):
        self.vlc_players[kind] = player

    # 获取VLC播放状态
    def get_vlc_state_name(self, kind: str) -> str:
        player = self.get_vlc_player(kind)
        if player is None:
            return "None"
        try:
            state = player.get_state()
            return str(state).split(".")[-1]
        except Exception:
            return "Unknown"

    # 判断VLC是否停止播放
    def is_vlc_terminal_state(self, kind: str) -> bool:
        return self.get_vlc_state_name(kind) in {"Ended", "Error", "Stopped"}

    # 判断音频是否正在播放
    def is_audio_playing(self, kind: str) -> bool:
        if self.has_vlc_backend():
            return self.get_vlc_state_name(kind) in {"Opening", "Buffering", "Playing"}

        process = self.get_player_process(kind)
        return process is not None and process.poll() is None

    # 切换音频播放状态
    def toggle_audio(self, kind: str):
        if not self.player_ready.get(kind, True):
            if kind == "output":
                messagebox.showwarning("无法播放", "新一轮处理尚未完成，输出音频会在处理完成后刷新。")
            return

        if self.is_audio_playing(kind):
            self.pause_audio(kind)
            return

        if self.player_paused.get(kind, False):
            self.resume_audio(kind)
        else:
            self.start_audio(kind)

    # 开始播放音频
    def start_audio(self, kind: str):
        path = self.get_audio_path_for_player(kind)

        if not path.exists():
            if kind == "output":
                messagebox.showwarning("无法播放", "输出音频还不存在，请先完成处理。")
            else:
                messagebox.showwarning("无法播放", "请选择有效的输入音频文件。")
            return

        if not self.has_vlc_backend():
            ffplay = shutil.which("ffplay")
            if ffplay is None:
                messagebox.showerror(
                    "缺少播放器",
                    "未找到 VLC/python-vlc 或 ffplay。请安装 VLC 并执行：pip install python-vlc，或安装 FFmpeg。",
                )
                return

        self.stop_audio(kind, reset_ui=False)
        self.ensure_player_source_fresh(kind, reset_if_changed=True)

        duration = self.player_durations.get(kind, 0.0)
        self.player_source_signatures[kind] = self.get_file_signature(path)
        self.player_elapsed_before_pause[kind] = 0.0
        self.player_started_at[kind] = time.time()
        self.player_paused[kind] = False
        self.player_progress_vars[kind].set(0.0)
        self.player_time_vars[kind].set(f"00:00 / {self.format_time(duration)}")

        if self.has_vlc_backend():
            ok = self.start_vlc_audio(kind, offset_seconds=0.0)
        else:
            process = self.start_ffplay_process(kind, offset_seconds=0.0)
            ok = process is not None
            if process is not None:
                self.set_player_process(kind, process)

        if not ok:
            self.reset_player_ui(kind)
            return

        self.player_buttons[kind].configure(text="暂停")
        self.update_audio_progress(kind)

    # 使用VLC播放音频
    def start_vlc_audio(self, kind: str, offset_seconds: float = 0.0) -> bool:
        path = self.get_audio_path_for_player(kind)
        if self.vlc_instance is None:
            return False

        try:
            player = self.vlc_instance.media_player_new()
            media = self.vlc_instance.media_new(str(path))
            player.set_media(media)
            player.audio_set_volume(self.get_player_volume(kind))
            result = player.play()
            if result == -1:
                return False

            self.set_vlc_player(kind, player)
            self.vlc_active_media_signatures[kind] = self.get_file_signature(path)
            self.player_applied_volume[kind] = self.get_player_volume(kind)
            self.player_started_at[kind] = time.time()
            self.player_paused[kind] = False

            if offset_seconds > 0:
                self.after(80, lambda player_kind=kind, seconds=offset_seconds: self.apply_vlc_seek(player_kind, seconds))

            return True
        except Exception as exc:
            messagebox.showerror("播放失败", f"VLC 播放失败：{exc}")
            self.set_vlc_player(kind, None)
            return False

    # 应用VLC播放位置
    def apply_vlc_seek(self, kind: str, target_seconds: float):
        player = self.get_vlc_player(kind)
        if player is None:
            return
        try:
            player.set_time(int(max(0.0, float(target_seconds)) * 1000))
        except Exception:
            pass

    # 暂停播放音频
    def pause_audio(self, kind: str):
        if self.has_vlc_backend():
            player = self.get_vlc_player(kind)
            if player is None:
                self.reset_player_ui(kind)
                return

            elapsed = self.get_audio_elapsed(kind)
            try:
                player.pause()
            except Exception:
                pass
            self.player_elapsed_before_pause[kind] = elapsed
            self.player_started_at[kind] = 0.0
            self.player_paused[kind] = True
            self.player_buttons[kind].configure(text="播放")
            self.update_audio_progress(kind)
            return

        process = self.get_player_process(kind)
        if process is None or process.poll() is not None:
            self.reset_player_ui(kind)
            return

        elapsed = self.get_audio_elapsed(kind)
        self.terminate_player_process(kind)
        self.player_elapsed_before_pause[kind] = elapsed
        self.player_started_at[kind] = 0.0
        self.player_paused[kind] = True
        self.player_buttons[kind].configure(text="播放")
        self.update_audio_progress(kind)

    # 继续播放音频
    def resume_audio(self, kind: str):
        if not self.ensure_player_source_fresh(kind, reset_if_changed=False):
            self.reset_player_ui(kind)
            return

        duration = self.player_durations.get(kind, 0.0)
        offset = self.player_elapsed_before_pause.get(kind, 0.0)

        if duration > 0 and offset >= duration:
            offset = 0.0
            self.player_elapsed_before_pause[kind] = 0.0
            self.player_progress_vars[kind].set(0.0)

        if self.has_vlc_backend():
            player = self.get_vlc_player(kind)
            current_signature = self.get_file_signature(self.get_audio_path_for_player(kind))

            if player is not None and self.vlc_active_media_signatures.get(kind) == current_signature:
                try:
                    player.audio_set_volume(self.get_player_volume(kind))
                    player.set_time(int(max(0.0, offset) * 1000))
                    player.play()
                except Exception:
                    if not self.start_vlc_audio(kind, offset_seconds=offset):
                        return
            else:
                if not self.start_vlc_audio(kind, offset_seconds=offset):
                    return
        else:
            process = self.start_ffplay_process(kind, offset_seconds=offset)
            if process is None:
                return
            self.set_player_process(kind, process)

        self.player_started_at[kind] = time.time()
        self.player_paused[kind] = False
        self.player_buttons[kind].configure(text="暂停")
        self.update_audio_progress(kind)

    # 启动ffplay播放器
    def start_ffplay_process(self, kind: str, offset_seconds: float) -> subprocess.Popen | None:
        path = self.get_audio_path_for_player(kind)
        ffplay = shutil.which("ffplay")
        if ffplay is None:
            messagebox.showerror("缺少 ffplay", "未找到 ffplay。请确认 FFmpeg 已安装并加入 PATH。")
            return None

        volume = self.get_player_volume(kind)
        cmd = [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-volume",
            str(volume),
        ]

        if offset_seconds > 0:
            cmd.extend(["-ss", f"{offset_seconds:.3f}"])

        cmd.append(str(path))

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.player_applied_volume[kind] = volume
            return process
        except Exception as exc:
            messagebox.showerror("播放失败", str(exc))
            return None

    # 停止VLC播放器
    def stop_vlc_player(self, kind: str):
        player = self.get_vlc_player(kind)
        if player is None:
            return

        self.set_vlc_player(kind, None)
        self.vlc_active_media_signatures[kind] = None

        # 释放VLC播放器
        def dispose_player():
            try:
                player.stop()
            except Exception:
                pass
            try:
                player.release()
            except Exception:
                pass

        threading.Thread(target=dispose_player, daemon=True).start()

    # 异步结束播放器进程
    @staticmethod
    def terminate_process_without_blocking(process: subprocess.Popen | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except Exception:
            return

        # 等待并强制结束进程
        def wait_and_kill():
            try:
                process.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
            except Exception:
                pass

        threading.Thread(target=wait_and_kill, daemon=True).start()

    # 结束播放器进程
    def terminate_player_process(self, kind: str):
        if self.has_vlc_backend():
            self.stop_vlc_player(kind)

        process = self.get_player_process(kind)
        self.set_player_process(kind, None)
        self.terminate_process_without_blocking(process)

    # 处理音量变化
    def on_volume_change(self, kind: str):
        if self.has_vlc_backend():
            desired_volume = self.get_player_volume(kind)
            player = self.get_vlc_player(kind)
            if player is not None:
                try:
                    player.audio_set_volume(desired_volume)
                except Exception:
                    pass
            self.player_applied_volume[kind] = desired_volume
            return


        after_id = self.volume_restart_after_ids.get(kind)
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass

        self.volume_restart_after_ids[kind] = self.after(
            300,
            lambda player_kind=kind: self.apply_player_volume(player_kind),
        )

    # 获取播放器音量
    def get_player_volume(self, kind: str) -> int:
        try:
            value = float(self.player_volume_vars[kind].get())
        except Exception:
            value = 80.0
        return int(max(0, min(100, round(value))))

    # 应用播放器音量
    def apply_player_volume(self, kind: str):
        self.volume_restart_after_ids[kind] = None

        desired_volume = self.get_player_volume(kind)
        process = self.get_player_process(kind)

        if process is None or process.poll() is not None:
            self.player_applied_volume[kind] = desired_volume
            return

        if desired_volume == self.player_applied_volume.get(kind):
            return

        path = self.get_audio_path_for_player(kind)
        if not path.exists():
            return

        if not self.ensure_player_source_fresh(kind, reset_if_changed=False):
            return

        elapsed = self.get_audio_elapsed(kind)
        duration = self.player_durations.get(kind, 0.0)
        if duration > 0:
            elapsed = max(0.0, min(elapsed, duration))
            if elapsed >= duration:
                self.player_applied_volume[kind] = desired_volume
                return

        self.terminate_player_process(kind)
        self.player_elapsed_before_pause[kind] = elapsed
        self.player_started_at[kind] = 0.0

        process = self.start_ffplay_process(kind, offset_seconds=elapsed)
        if process is None:
            self.player_paused[kind] = True
            self.player_buttons[kind].configure(text="播放")
            self.update_audio_progress(kind)
            return

        self.set_player_process(kind, process)
        self.player_started_at[kind] = time.time()
        self.player_paused[kind] = False
        self.player_buttons[kind].configure(text="暂停")
        self.update_audio_progress(kind)

    # 预览播放位置
    def on_seek_preview(self, kind: str, value: str | float):
        if not self.player_is_seeking.get(kind, False):
            return
        duration = self.player_durations.get(kind, 0.0)
        if duration <= 0:
            self.player_time_vars[kind].set("00:00 / 00:00")
            return
        try:
            percent = float(value)
        except (TypeError, ValueError):
            percent = self.player_progress_vars[kind].get()
        percent = max(0.0, min(100.0, percent))
        elapsed = duration * percent / 100.0
        self.player_time_vars[kind].set(f"{self.format_time(elapsed)} / {self.format_time(duration)}")

    # 开始拖动播放位置
    def begin_seek(self, kind: str):
        if not self.player_ready.get(kind, True):
            return
        self.ensure_player_source_fresh(kind, reset_if_changed=False)
        self.player_is_seeking[kind] = True

    # 结束拖动播放位置
    def end_seek(self, kind: str):
        if not self.player_ready.get(kind, True):
            return
        duration = self.player_durations.get(kind, 0.0)
        if duration <= 0:
            self.player_is_seeking[kind] = False
            self.reset_player_ui(kind)
            return

        percent = max(0.0, min(100.0, float(self.player_progress_vars[kind].get())))
        target = duration * percent / 100.0
        self.player_is_seeking[kind] = False
        self.seek_audio(kind, target)

    # 跳转音频播放位置
    def seek_audio(self, kind: str, target_seconds: float):
        path = self.get_audio_path_for_player(kind)
        if not path.exists():
            self.reset_player_ui(kind)
            return

        self.ensure_player_source_fresh(kind, reset_if_changed=False)
        duration = self.player_durations.get(kind, 0.0)
        target_seconds = max(0.0, min(float(target_seconds), duration if duration > 0 else float(target_seconds)))

        if self.has_vlc_backend():
            player = self.get_vlc_player(kind)
            was_playing = self.is_audio_playing(kind)

            if player is None:
                self.player_elapsed_before_pause[kind] = target_seconds
                self.player_started_at[kind] = 0.0
                self.player_paused[kind] = True
                self.update_audio_progress(kind)
                return

            try:
                player.set_time(int(target_seconds * 1000))
                player.audio_set_volume(self.get_player_volume(kind))
            except Exception:
                pass

            self.player_elapsed_before_pause[kind] = target_seconds
            self.player_started_at[kind] = time.time() if was_playing else 0.0
            self.player_paused[kind] = not was_playing
            self.player_buttons[kind].configure(text="暂停" if was_playing else "播放")
            self.update_audio_progress(kind)
            return

        current_process = self.get_player_process(kind)
        was_playing = current_process is not None and current_process.poll() is None

        self.terminate_player_process(kind)
        self.player_elapsed_before_pause[kind] = target_seconds
        self.player_started_at[kind] = 0.0
        self.player_paused[kind] = True
        self.player_buttons[kind].configure(text="播放")
        self.update_audio_progress(kind)

        if was_playing:
            process = self.start_ffplay_process(kind, offset_seconds=target_seconds)
            if process is not None:
                self.set_player_process(kind, process)
                self.player_started_at[kind] = time.time()
                self.player_paused[kind] = False
                self.player_buttons[kind].configure(text="暂停")
                self.update_audio_progress(kind)

    # 停止播放音频
    def stop_audio(self, kind: str, reset_ui: bool = True):
        after_id = self.volume_restart_after_ids.get(kind)
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
            self.volume_restart_after_ids[kind] = None

        after_id = self.player_seek_after_ids.get(kind)
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass
            self.player_seek_after_ids[kind] = None

        self.player_is_seeking[kind] = False
        self.terminate_player_process(kind)

        if reset_ui:
            self.reset_player_ui(kind)

    # 停止全部音频
    def stop_all_audio(self):
        self.stop_audio("input")
        self.stop_audio("output")
        self.stop_synth_test_audio()

    # 重置播放器界面
    def reset_player_ui(self, kind: str):
        self.player_paused[kind] = False
        self.player_is_seeking[kind] = False
        self.player_elapsed_before_pause[kind] = 0.0
        self.player_started_at[kind] = 0.0
        self.player_progress_vars[kind].set(0.0)
        self.player_buttons[kind].configure(text="播放")

        duration = self.player_durations.get(kind, 0.0)
        self.player_time_vars[kind].set(f"00:00 / {self.format_time(duration)}")

    # 获取音频已播放时长
    def get_audio_elapsed(self, kind: str) -> float:
        if self.has_vlc_backend():
            player = self.get_vlc_player(kind)
            if player is not None:
                try:
                    milliseconds = player.get_time()
                    if milliseconds is not None and milliseconds >= 0:
                        elapsed = milliseconds / 1000.0
                        duration = self.player_durations.get(kind, 0.0)
                        if duration > 0:
                            elapsed = min(elapsed, duration)
                        return max(0.0, elapsed)
                except Exception:
                    pass

        elapsed = self.player_elapsed_before_pause.get(kind, 0.0)

        if not self.player_paused.get(kind, False):
            started_at = self.player_started_at.get(kind, 0.0)
            if started_at > 0:
                elapsed += max(0.0, time.time() - started_at)

        duration = self.player_durations.get(kind, 0.0)
        if duration > 0:
            elapsed = min(elapsed, duration)

        return elapsed

    # 更新音频播放进度
    def update_audio_progress(self, kind: str):
        duration = self.player_durations.get(kind, 0.0)

        if self.has_vlc_backend():
            player = self.get_vlc_player(kind)

            if self.player_is_seeking.get(kind, False):
                if player is not None and not self.is_vlc_terminal_state(kind):
                    self.after(100, lambda: self.update_audio_progress(kind))
                return

            elapsed = self.get_audio_elapsed(kind)
            percent = max(0.0, min(100.0, elapsed / duration * 100.0)) if duration > 0 else 0.0
            self.player_progress_vars[kind].set(percent)
            self.player_time_vars[kind].set(f"{self.format_time(elapsed)} / {self.format_time(duration)}")

            state_name = self.get_vlc_state_name(kind)
            if state_name in {"Ended", "Error"}:
                self.player_elapsed_before_pause[kind] = duration if state_name == "Ended" and duration > 0 else elapsed
                if state_name == "Ended" and duration > 0:
                    self.player_progress_vars[kind].set(100.0)
                self.player_buttons[kind].configure(text="播放")
                self.player_paused[kind] = True
                self.stop_vlc_player(kind)
                return

            if player is not None and state_name not in {"Stopped", "None", "Error", "Ended"}:
                self.after(100, lambda: self.update_audio_progress(kind))
            return

        process = self.get_player_process(kind)

        if self.player_is_seeking.get(kind, False):
            if process is not None and process.poll() is None:
                self.after(300, lambda: self.update_audio_progress(kind))
            return

        if process is None:
            elapsed = self.get_audio_elapsed(kind)
            if duration > 0:
                percent = max(0.0, min(100.0, elapsed / duration * 100.0))
                self.player_progress_vars[kind].set(percent)
            self.player_time_vars[kind].set(f"{self.format_time(elapsed)} / {self.format_time(duration)}")
            return

        elapsed = self.get_audio_elapsed(kind)

        if duration > 0:
            percent = max(0.0, min(100.0, elapsed / duration * 100.0))
        else:
            percent = 0.0

        self.player_progress_vars[kind].set(percent)
        self.player_time_vars[kind].set(f"{self.format_time(elapsed)} / {self.format_time(duration)}")

        if process.poll() is not None:
            self.set_player_process(kind, None)
            if duration > 0 and elapsed >= duration * 0.95:
                self.player_progress_vars[kind].set(100.0)
                self.player_elapsed_before_pause[kind] = duration
            self.player_buttons[kind].configure(text="播放")
            self.player_paused[kind] = True
            return

        self.after(300, lambda: self.update_audio_progress(kind))

    # 获取音频时长
    @staticmethod
    def get_audio_duration(path: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return 0.0

        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=4.0,
            )
            return max(0.0, float(result.stdout.strip()))
        except Exception:
            return 0.0

    # 格式化时间
    @staticmethod
    def format_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    # 添加路径选择行
    def _add_path_row(self, parent, row, label, variable, button_text, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=(8, 8), pady=4)
        button = ttk.Button(parent, text=button_text, command=command)
        button.grid(row=row, column=2, sticky=tk.E, pady=4)

    # 添加合成参数区域
    def _add_config_section(self, parent, title: str, base_path: tuple[str, ...], values: dict):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, padx=2, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        has_synth_test_button = self.is_synth_test_section(base_path)
        group_row_offset = 0

        if has_synth_test_button:
            group_row_offset = 1
            test_frame = ttk.Frame(frame)
            test_frame.grid(
                row=0,
                column=0,
                columnspan=2,
                sticky=tk.EW,
                padx=3,
                pady=(0, 6),
            )
            test_frame.columnconfigure(0, weight=1)

            button = ttk.Button(
                test_frame,
                text="播放测试音",
                command=lambda path=base_path: self.play_synth_test_audio(path),
            )
            button.grid(row=0, column=1, sticky=tk.E, padx=(8, 0))
            self.synth_test_buttons[base_path] = button
            self.parameter_widgets.append((button, "normal"))

        groups = self.get_config_param_groups(base_path, values)

        single_column_groups = base_path == ()
        for group_index, (group_title, keys) in enumerate(groups):
            if single_column_groups:
                row = group_index + group_row_offset
                column = 0
                columnspan = 2
            else:
                row = group_index // 2 + group_row_offset
                column = group_index % 2
                columnspan = 1

            group_frame = ttk.LabelFrame(frame, text=group_title, padding=6)
            group_frame.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky=tk.NSEW,
                padx=3,
                pady=3,
            )
            group_frame.columnconfigure(1, weight=1)

            for param_row, key in enumerate(keys):
                if key not in values:
                    continue
                self.add_config_param_row(
                    parent=group_frame,
                    row=param_row,
                    key=key,
                    value=values[key],
                    path=base_path + (key,),
                )

    # 判断是否为测试音区域
    @staticmethod
    def is_synth_test_section(base_path: tuple[str, ...]) -> bool:
        return (
            len(base_path) == 2
            and base_path[0] in {"channels", "drums"}
        )

    # 添加合成参数行
    def add_config_param_row(self, parent, row: int, key: str, value, path: tuple[str, ...]):
        label_widget = ttk.Label(
            parent,
            text=self.get_param_label(key),
            width=12,
            anchor=tk.W,
        )
        label_widget.grid(
            row=row,
            column=0,
            sticky=tk.W,
            padx=(0, 4),
            pady=2,
        )

        description = PARAM_DESCRIPTIONS.get(key, "")
        if description:
            ToolTip(label_widget, description)

        var = tk.StringVar(value=self.value_to_text(value))
        self.config_vars[path] = var
        self.config_original_values[path] = value

        if key == "channel_mode":
            widget = ttk.Combobox(
                parent,
                textvariable=var,
                values=CHANNEL_MODE_OPTIONS,
                state="readonly",
                width=8,
            )
        elif key == "waveform":
            widget = ttk.Combobox(
                parent,
                textvariable=var,
                values=WAVEFORM_OPTIONS,
                state="readonly",
                width=8,
            )
        elif key == "noise_filter":
            widget = ttk.Combobox(
                parent,
                textvariable=var,
                values=NOISE_FILTER_OPTIONS,
                state="readonly",
                width=8,
            )
        else:
            widget = ttk.Entry(parent, textvariable=var, width=9)

        widget.grid(
            row=row,
            column=1,
            sticky=tk.EW,
            padx=(0, 2),
            pady=2,
        )
        if description:
            ToolTip(widget, description)
        normal_state = "readonly" if isinstance(widget, ttk.Combobox) else "normal"
        self.parameter_widgets.append((widget, normal_state))

        if path == ("output", "channel_mode"):
            var.trace_add("write", lambda *_args: self.update_stereo_parameter_state())
        elif path in {("output", "stereo_width"), ("output", "pan_law_db")} or key == "pan":
            self.stereo_parameter_widgets.append((widget, normal_state))

    # 更新立体声参数状态
    def update_stereo_parameter_state(self):
        mode_var = self.config_vars.get(("output", "channel_mode"))
        stereo_enabled = mode_var is not None and mode_var.get().strip().lower() == "stereo"

        mono_frame = self.mono_parameter_frame
        stereo_frame = self.stereo_parameter_frame
        if mono_frame is not None and stereo_frame is not None:
            try:
                if stereo_enabled:
                    if mono_frame.winfo_manager():
                        mono_frame.pack_forget()
                    if not stereo_frame.winfo_manager():
                        stereo_frame.pack(fill=tk.X)
                else:
                    if stereo_frame.winfo_manager():
                        stereo_frame.pack_forget()
                    if not mono_frame.winfo_manager():
                        mono_frame.pack(fill=tk.X, padx=2, pady=4)
            except Exception:
                pass

        for widget, normal_state in self.stereo_parameter_widgets:
            try:
                if self.processing_running or not stereo_enabled:
                    widget.configure(state="disabled")
                else:
                    widget.configure(state=normal_state)
            except Exception:
                pass

    # 添加音高参数区域
    def _add_pitch_config_section(self, parent, title: str, section: str, values: dict):
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, padx=2, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        groups = self.get_pitch_param_groups(section, values)

        for group_index, (group_title, keys) in enumerate(groups):
            row = group_index // 2
            column = group_index % 2

            group_frame = ttk.LabelFrame(frame, text=group_title, padding=6)
            group_frame.grid(
                row=row,
                column=column,
                sticky=tk.NSEW,
                padx=3,
                pady=3,
            )
            group_frame.columnconfigure(1, weight=1)

            for param_row, key in enumerate(keys):
                if key not in values:
                    continue
                self.add_pitch_param_row(
                    parent=group_frame,
                    row=param_row,
                    section=section,
                    key=key,
                    value=values[key],
                )

    # 添加音高参数行
    def add_pitch_param_row(self, parent, row: int, section: str, key: str, value):
        path = (section, key)
        label_widget = ttk.Label(
            parent,
            text=PITCH_PARAM_LABELS.get(key, key),
            width=15,
            anchor=tk.W,
        )
        label_widget.grid(
            row=row,
            column=0,
            sticky=tk.W,
            padx=(0, 4),
            pady=2,
        )

        description = PITCH_PARAM_DESCRIPTIONS.get(key, "")
        if description:
            ToolTip(label_widget, description)

        var = tk.StringVar(value=self.value_to_text(value))
        self.pitch_config_vars[path] = var
        self.pitch_config_original_values[path] = value

        widget = ttk.Entry(parent, textvariable=var, width=9)
        widget.grid(
            row=row,
            column=1,
            sticky=tk.EW,
            padx=(0, 2),
            pady=2,
        )
        if description:
            ToolTip(widget, description)
        self.parameter_widgets.append((widget, "normal"))

    # 构建参数分组
    @staticmethod
    def build_param_groups(values: dict, candidates: list[tuple[str, list[str]]], ordered_keys: list[str]) -> list[tuple[str, list[str]]]:
        used: set[str] = set()
        groups: list[tuple[str, list[str]]] = []

        for group_title, keys in candidates:
            present_keys = [key for key in keys if key in values]
            if present_keys:
                groups.append((group_title, present_keys))
                used.update(present_keys)

        remaining = [key for key in ordered_keys if key in values and key not in used]
        if remaining:
            groups.append(("其他参数", remaining))

        return groups

    # 获取合成参数分组
    def get_config_param_groups(self, base_path: tuple[str, ...], values: dict) -> list[tuple[str, list[str]]]:
        ordered_keys = [key for key, _value in self.get_ordered_config_items(base_path, values)]

        if base_path == ():
            candidates = [
                ("输出设置", ["sample_rate", "master_volume", "bit_depth"]),
                ("ADR包络最长时间", ["max_attack_duration_sec", "max_decay_duration_sec", "max_release_duration_sec"]),
            ]
        elif base_path == ("output",):
            candidates = [
                ("声道模式", ["channel_mode"]),
                ("立体声参数", ["stereo_width", "pan_law_db"]),
            ]
        elif base_path == ("rms_control",):
            candidates = [
                ("单轨平衡", ["enabled", "block_ms", "gate_db", "track_peak_db", "track_min_gain_db", "track_max_gain_db"]),
                ("最终总线", ["final_target_rms_db", "final_peak_db", "final_min_gain_db", "final_max_gain_db"]),
            ]
        elif len(base_path) >= 1 and base_path[0] == "channels":
            source_keys = ["waveform", "volume", "target_rms_db", "duty", "steps"]
            if len(base_path) >= 2 and base_path[1] in {"guitar", "piano"}:
                source_keys.append("arpeggio_interval")
            candidates = [
                ("音源参数", source_keys),
                ("包络参数", ["attack", "decay", "sustain", "release"]),
            ]
            if len(base_path) >= 2 and base_path[1] in {"vocal", "other"}:
                candidates.append(
                    ("事件交叉淡化", ["event_crossfade_sec", "event_crossfade_max_gap_sec"])
                )
        elif len(base_path) >= 1 and base_path[0] == "drums":
            candidates = [
                ("基础与音高", ["volume", "target_rms_db", "start_freq", "end_freq", "pitch_curve", "duty"]),
                ("波形混合", ["triangle_mix", "square_mix", "noise_mix"]),
                ("噪声滤波", ["noise_filter", "noise_low", "noise_high"]),
                ("包络参数", ["attack", "decay", "sustain", "release"]),
            ]
        else:
            candidates = [("参数", ordered_keys)]

        return self.build_param_groups(values, candidates, ordered_keys)

    # 获取音高参数分组
    def get_pitch_param_groups(self, section: str, values: dict) -> list[tuple[str, list[str]]]:
        ordered_keys = list(values.keys())

        if section == "vocals":
            candidates = [
                ("分析参数", ["DEFAULT_RMVPE_THRESHOLD"]),
                ("平滑与切分", ["MIN_NOTE_SEC", "PITCH_SPLIT_SEMITONES", "PITCH_HOLD_FRAMES", "SMOOTH_KERNEL_SIZE", "MAX_UNVOICED_GAP_FRAMES"]),
                ("起音检测", ["ONSET_MIN_GAP_FRAMES", "ONSET_PRE_MAX", "ONSET_POST_MAX", "ONSET_PRE_AVG", "ONSET_POST_AVG", "ONSET_DELTA", "ONSET_WAIT"]),
            ]
        elif section in {"guitar", "piano"}:
            candidates = [
                ("模型阈值", ["BASIC_PITCH_ONSET_THRESHOLD", "BASIC_PITCH_FRAME_THRESHOLD", "BASIC_PITCH_MINIMUM_NOTE_LENGTH"]),
                ("频率与过滤", ["MIN_FREQUENCY_HZ", "MAX_FREQUENCY_HZ", "MIN_NOTE_DURATION_SECONDS"]),
                ("和弦聚合", ["ONSET_TOLERANCE_SECONDS", "MIN_OVERLAP_RATIO", "MAX_NOTES_PER_CHORD"]),
            ]
        elif section == "bass":
            candidates = [
                ("分析范围", ["FRAME_LENGTH", "FMIN", "FMAX", "PITCH_ANALYSIS_CHUNK_SEC"]),
                ("预处理", ["TRIM_TOP_DB", "HARMONIC_MARGIN"]),
                ("平滑与过滤", ["SMOOTH_KERNEL_SIZE", "MAX_GAP_FRAMES", "MIN_NOTE_DURATION_SEC"]),
            ]
        elif section == "other":
            candidates = [
                ("分析范围", ["FMIN", "FMAX", "PITCH_ANALYSIS_CHUNK_SEC"]),
                ("模型与置信度", ["TORCHCREPE_MODEL", "TORCHCREPE_BATCH_SIZE", "PERIODICITY_MEDIAN_KERNEL", "SILENCE_THRESHOLD_DB", "PITCH_CONFIDENCE_THRESHOLD", "PITCH_MEAN_FILTER_KERNEL"]),
                ("预处理与平滑", ["TRIM_TOP_DB", "HARMONIC_MARGIN", "MIDI_SMOOTH_KERNEL_SIZE", "MAX_UNVOICED_GAP_FRAMES", "MIN_NOTE_SEC", "MIN_MEAN_PERIODICITY"]),
                ("起音与切分", ["ONSET_PRE_MAX", "ONSET_POST_MAX", "ONSET_PRE_AVG", "ONSET_POST_AVG", "ONSET_DELTA", "ONSET_WAIT", "MIN_FRAMES_BEFORE_ONSET_SPLIT", "PITCH_CHANGE_SEMITONES", "PITCH_CHANGE_LOOKAHEAD_FRAMES", "PITCH_CHANGE_CONFIRM_FRAMES"]),
            ]
        elif section == "drums":
            candidates = [
                ("起音检测", ["HOP_LENGTH", "ONSET_DELTA", "MIN_ONSET_INTERVAL_SEC", "ONSET_ENV_SMOOTH_FRAMES"]),
                ("RMS门限", ["RMS_GATE_ABS", "RMS_GATE_RELATIVE_TO_P95", "MIN_EVENT_RMS", "MIN_EVENT_RMS_RATIO_TO_TRACK_P95"]),
                ("事件长度与结束", ["MIN_EVENT_DURATION_SEC", "MAX_EVENT_DURATION_SEC", "END_RELATIVE_THRESHOLD", "END_ABSOLUTE_FLOOR", "END_SILENCE_FRAMES"]),
                ("瞬态过滤", ["MIN_EVENT_PEAK", "PRE_EVENT_SEC", "MIN_TRANSIENT_RMS_RATIO", "PRE_RMS_MIN_FOR_TRANSIENT_CHECK"]),
            ]
        else:
            candidates = [("参数", ordered_keys)]

        return self.build_param_groups(values, candidates, ordered_keys)

    # 获取有序配置项
    def get_ordered_config_items(self, base_path: tuple[str, ...], values: dict):
        if base_path == ():
            order = GLOBAL_PARAM_ORDER
        elif base_path == ("output",):
            order = OUTPUT_PARAM_ORDER
        elif base_path == ("rms_control",):
            order = RMS_PARAM_ORDER
        elif len(base_path) >= 1 and base_path[0] == "channels":
            order = CHANNEL_PARAM_ORDER
        elif len(base_path) >= 1 and base_path[0] == "drums":
            order = DRUM_PARAM_ORDER
        else:
            order = list(values.keys())

        ordered = [(key, values[key]) for key in order if key in values]
        remaining = [(key, value) for key, value in values.items() if key not in order]
        return ordered + remaining

    # 获取参数标签
    @staticmethod
    def get_param_label(key: str) -> str:
        return PARAM_LABELS.get(key, key)

    # 加载默认合成配置
    def load_default_config(self) -> dict:
        if chiptune is not None and hasattr(chiptune, "DEFAULT_CONFIG"):
            return copy.deepcopy(chiptune.DEFAULT_CONFIG)

        return {
            "sample_rate": 44100,
            "master_volume": 1.0,
            "bit_depth": 8,
            "max_attack_duration_sec": 0.05,
            "max_decay_duration_sec": 0.05,
            "max_release_duration_sec": 0.05,
            "output": {
                "channel_mode": "mono",
                "stereo_width": 1.0,
                "pan_law_db": -3.0,
            },
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
            "channels": {},
            "drums": {},
        }

    # 将参数值转换为文本
    @staticmethod
    def value_to_text(value) -> str:
        if value is None:
            return ""
        return str(value)

    # 解析配置参数值
    @staticmethod
    def parse_config_value(text: str, original_value):
        text = text.strip()

        if text.lower() in {"", "none", "null"}:
            return None if original_value is None else original_value

        if isinstance(original_value, bool):
            return text.lower() in {"1", "true", "yes", "y", "是"}

        if isinstance(original_value, int) and not isinstance(original_value, bool):
            return int(float(text))

        if isinstance(original_value, float):
            return float(text)

        if original_value is None:
            try:
                return float(text)
            except ValueError:
                return text

        return text

    # 生成合成配置
    def build_synth_config(self) -> dict:
        config = copy.deepcopy(self.default_config)

        for path, var in self.config_vars.items():
            original_value = self.config_original_values[path]
            value = self.parse_config_value(var.get(), original_value)

            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

        for key, display_name in (
            ("max_attack_duration_sec", "Attack最长时间"),
            ("max_decay_duration_sec", "Decay最长时间"),
            ("max_release_duration_sec", "Release最长时间"),
        ):
            value = config.get(key)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{display_name}必须是大于等于 0 的有限数值；0 表示不限制")
            config[key] = value

        for channel in ("vocal", "other"):
            channel_config = config.get("channels", {}).get(channel, {})
            event_crossfade_sec = float(channel_config.get("event_crossfade_sec", 0.0) or 0.0)
            if not np.isfinite(event_crossfade_sec) or event_crossfade_sec < 0:
                raise ValueError(f"{CHANNEL_DISPLAY_NAMES.get(channel, channel)}交叉淡化时间必须是大于等于 0 的有限数值；0 表示关闭")
            channel_config["event_crossfade_sec"] = event_crossfade_sec

            event_crossfade_max_gap_sec = float(channel_config.get("event_crossfade_max_gap_sec", 0.0) or 0.0)
            if not np.isfinite(event_crossfade_max_gap_sec) or event_crossfade_max_gap_sec < 0:
                raise ValueError(f"{CHANNEL_DISPLAY_NAMES.get(channel, channel)}允许交叉淡化的最大间隔必须是大于等于 0 的有限数值")
            channel_config["event_crossfade_max_gap_sec"] = event_crossfade_max_gap_sec

        return config

    # 保存合成配置
    def save_synth_config(self, work_dir: Path) -> Path:
        config = self.build_synth_config()
        config_path = work_dir / "ui_synth_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config_path


    # 生成音高配置
    def build_pitch_config(self) -> dict:
        config = copy.deepcopy(self.pitch_config_defaults)

        for path, var in self.pitch_config_vars.items():
            section, key = path
            original_value = self.pitch_config_original_values[path]
            value = self.parse_config_value(var.get(), original_value)
            config[section][key] = value


        return config

    # 保存音高配置
    def save_pitch_config(self, work_dir: Path) -> Path:
        config = self.build_pitch_config()
        config_path = work_dir / "ui_pitch_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config_path

    # 重置合成参数
    def reset_synth_params(self):
        items = [
            (path, var, self.config_original_values[path])
            for path, var in self.config_vars.items()
        ]
        items.extend(
            (path, var, self.pitch_config_original_values[path])
            for path, var in self.pitch_config_vars.items()
        )

        self.reset_params_button.configure(state=tk.DISABLED)
        self.status_var.set("正在重置参数...")

        # 批量应用默认参数
        def apply_batch(start: int = 0):
            end = min(len(items), start + 32)
            for _path, var, original in items[start:end]:
                var.set(self.value_to_text(original))

            if end < len(items):
                self.after_idle(apply_batch, end)
                return

            self.update_stereo_parameter_state()
            if not self.processing_running:
                self.reset_params_button.configure(state=tk.NORMAL)
            self.status_var.set("参数已重置")

        self.after_idle(apply_batch, 0)


    # 停止测试音播放
    def stop_synth_test_audio(self, *, cancel_pending: bool = True):
        if cancel_pending:
            self.synth_test_task_token += 1

        if winsound is not None and sys.platform.startswith("win"):
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

        process = self.synth_test_player_process
        self.synth_test_player_process = None
        self.terminate_process_without_blocking(process)

    # 获取鼓点测试时长
    @staticmethod
    def get_drum_test_duration(drum: str) -> float:
        return CYMBALS_TEST_DURATION_SEC if drum == "cymbals" else DRUM_TEST_DURATION_SEC

    # 生成测试音缓存键
    @staticmethod
    def make_synth_test_cache_key(base_path: tuple[str, ...], config: dict) -> str:
        payload = {
            "version": SYNTH_TEST_CACHE_VERSION,
            "base_path": list(base_path),
            "config": config,
            "melodic_duration": MELODIC_TEST_DURATION_SEC,
            "drum_duration": DRUM_TEST_DURATION_SEC,
            "cymbals_duration": CYMBALS_TEST_DURATION_SEC,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

    # 获取测试音缓存路径
    def get_synth_test_cache_path(
        self,
        base_path: tuple[str, ...],
        config: dict,
    ) -> Path:
        base_dir = Path(__file__).resolve().parent
        test_dir = base_dir / "output" / "ui_test_audio"
        cache_key = self.make_synth_test_cache_key(base_path, config)
        section, name = base_path
        return test_dir / f"test_{section}_{name}_{cache_key}.wav"

    # 检查测试音缓存
    @staticmethod
    def is_valid_cached_wav(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 44
        except OSError:
            return False

    # 设置测试音按钮状态
    def set_synth_test_button_busy(self, base_path: tuple[str, ...], busy: bool) -> None:
        button = self.synth_test_buttons.get(base_path)
        if button is None:
            return
        try:
            should_disable = busy or bool(getattr(self, "processing_running", False))
            button.configure(state=tk.DISABLED if should_disable else tk.NORMAL)
        except Exception:
            pass

    # 开始播放测试音
    def start_synth_test_playback(self, output_path: Path, description: str) -> None:
        self.stop_synth_test_audio(cancel_pending=False)


        if winsound is not None and sys.platform.startswith("win"):
            try:
                flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
                winsound.PlaySound(str(output_path), flags)
                self.status_var.set(f"正在播放{description}测试音")
                return
            except Exception:
                pass

        ffplay = shutil.which("ffplay")
        if ffplay is None:
            messagebox.showerror("缺少播放器", "未找到 ffplay，并且系统异步 WAV 播放不可用。")
            return

        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

        try:
            self.synth_test_player_process = subprocess.Popen(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    str(output_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as exc:
            messagebox.showerror("播放失败", str(exc))
            return
        self.status_var.set(f"正在播放{description}测试音")

    # 生成并播放测试音
    def play_synth_test_audio(self, base_path: tuple[str, ...]):
        if chiptune is None:
            messagebox.showerror("无法播放", "未能导入 chiptune.py，请确认 ui.py 和 chiptune.py 位于同一目录。")
            return

        try:
            # Tk 变量只能在主线程读取；音频计算和文件写入放到后台。
            config = self.build_synth_config()
        except Exception as exc:
            messagebox.showerror("参数错误", f"测试音生成失败：{exc}")
            return

        output_path = self.get_synth_test_cache_path(base_path, config)
        description = self.get_synth_test_description(base_path)

        self.synth_test_task_token += 1
        token = self.synth_test_task_token

        if self.is_valid_cached_wav(output_path):
            self.start_synth_test_playback(output_path, description)
            return

        self.set_synth_test_button_busy(base_path, True)
        self.status_var.set("正在生成测试音...")

        # 生成测试音文件
        def worker():
            with self.synth_test_generation_lock:
                if self.is_valid_cached_wav(output_path):
                    return output_path, description
                audio, sample_rate, _unused_output_path, generated_description = self.make_synth_test_audio(
                    base_path,
                    config,
                    output_path=output_path,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = output_path.with_suffix(".tmp.wav")
                chiptune.save_wav(
                    filename=str(temp_path),
                    audio=audio,
                    sr=sample_rate,
                )
                temp_path.replace(output_path)
                return output_path, generated_description

        # 播放生成的测试音
        def start_playback(result):
            self.set_synth_test_button_busy(base_path, False)
            if token != self.synth_test_task_token:
                return
            cached_path, generated_description = result
            self.start_synth_test_playback(cached_path, generated_description)

        # 显示测试音错误
        def show_error(exc: Exception):
            self.set_synth_test_button_busy(base_path, False)
            if token != self.synth_test_task_token:
                return
            messagebox.showerror("测试音失败", str(exc))
            self.status_var.set("测试音生成失败")

        self.run_in_background(worker, on_success=start_playback, on_error=show_error)

    # 获取测试音说明
    @staticmethod
    def get_synth_test_description(base_path: tuple[str, ...]) -> str:
        if len(base_path) == 2 and base_path[0] == "channels":
            return CHANNEL_DISPLAY_NAMES.get(base_path[1], base_path[1])
        if len(base_path) == 2 and base_path[0] == "drums":
            return DRUM_DISPLAY_NAMES.get(base_path[1], base_path[1])
        return ""

    # 生成测试音音频
    def make_synth_test_audio(
        self,
        base_path: tuple[str, ...],
        config: dict,
        *,
        output_path: Path | None = None,
    ):
        sample_rate = int(config.get("sample_rate", 44100))
        base_dir = Path(__file__).resolve().parent
        test_dir = base_dir / "output" / "ui_test_audio"

        if len(base_path) == 2 and base_path[0] == "channels":
            channel = base_path[1]
            display = CHANNEL_DISPLAY_NAMES.get(channel, channel)
            audio = chiptune.render_event(
                channel=channel,
                freq=CENTRAL_C_FREQUENCY_HZ,
                start_time=0.0,
                end_time=MELODIC_TEST_DURATION_SEC,
                config=config,
            )
            params = chiptune.get_channel_params(channel, config)
            audio = chiptune.format_output_audio(audio, config, pan=float(params.get("pan", 0.0)))
            if output_path is None:
                output_path = test_dir / f"test_channel_{channel}.wav"
            return audio, sample_rate, output_path, display

        if len(base_path) == 2 and base_path[0] == "drums":
            drum = base_path[1]
            display = DRUM_DISPLAY_NAMES.get(drum, drum)
            params = chiptune.get_drum_params(drum, config)
            test_duration = self.get_drum_test_duration(drum)
            event_duration = max(1.0 / sample_rate, test_duration)

            event_audio = chiptune.generate_raw_drum_audio(
                drum=drum,
                duration=event_duration,
                loudness=1.0,
                config=config,
                seed=0,
            )

            total_samples = max(1, int(round(test_duration * sample_rate)))
            audio = chiptune.place_audio_on_timeline(
                event_audio=event_audio,
                start_time=0.0,
                end_time=test_duration,
                sr=sample_rate,
            )
            audio = audio[:total_samples]
            if len(audio) < total_samples:
                audio = np.pad(audio, (0, total_samples - len(audio)))
            audio = chiptune.finalize_track_audio(
                audio,
                config,
                target_rms_db=float(params.get("target_rms_db", -20.0)),
            )

            audio = chiptune.format_output_audio(audio, config, pan=float(params.get("pan", 0.0)))
            if output_path is None:
                output_path = test_dir / f"test_drum_{drum}.wav"
            return audio, sample_rate, output_path, display

        raise ValueError("只能在旋律合成参数或鼓组合成参数中播放测试音")

    # 选择输入音频
    def choose_input_file(self):
        path = filedialog.askopenfilename(
            title="选择输入音频文件",
            filetypes=[
                ("音频文件", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return

        self.stop_audio("input")
        self.stop_audio("output")
        self.input_path_var.set(path)
        input_path = Path(path)
        base_dir = Path(__file__).resolve().parent
        self.output_path_var.set(str(base_dir / "output" / f"{input_path.stem}_chiptune.wav"))
        self.work_dir_var.set(str(base_dir / "output" / f"{input_path.stem}_cache"))
        self.player_source_signatures["input"] = self.get_file_signature(input_path)
        if self.player_source_signatures["input"] is not None:
            self.request_player_duration(
                "input",
                reset_position=True,
                expected_signature=self.player_source_signatures["input"],
            )
        self.player_source_signatures["output"] = None
        self.player_durations["output"] = 0.0
        self.reset_player_ui("output")
        self.set_player_ready("output", True)
        self.status_var.set("已选择音频文件")

    # 选择输出音频
    def choose_output_file(self):
        path = filedialog.asksaveasfilename(
            title="选择输出 wav 文件",
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")],
        )
        if path:
            self.stop_audio("output")
            self.output_path_var.set(path)
            self.player_source_signatures["output"] = None
            self.player_durations["output"] = 0.0
            self.reset_player_ui("output")
            self.set_player_ready("output", True)

    # 选择工作目录
    def choose_work_dir(self):
        path = filedialog.askdirectory(title="选择缓存目录")
        if path:
            self.work_dir_var.set(path)


    # 启动处理任务
    def start_process(self):
        input_path = Path(self.input_path_var.get().strip())
        output_path = Path(self.output_path_var.get().strip())
        work_dir = Path(self.work_dir_var.get().strip())
        main_py = Path(__file__).resolve().parent / "main.py"

        if not input_path.exists():
            messagebox.showerror("错误", "请选择有效的输入音频文件。")
            return

        if not main_py.exists():
            messagebox.showerror("错误", "找不到 main.py，请确认 main.py 和 ui.py 放在同一目录。")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            synth_config = self.build_synth_config()
            config_json = self.save_synth_config(work_dir)
            pitch_config_json = self.save_pitch_config(work_dir)
        except Exception as exc:
            messagebox.showerror("参数错误", f"参数无法保存：{exc}")
            return

        bit_depth = synth_config.get("bit_depth")
        bit_depth_arg = "0" if bit_depth is None else str(int(bit_depth))
        master_volume_arg = str(float(synth_config.get("master_volume", 1.0)))

        self.prepare_players_for_new_processing(input_path, output_path)
        self.stop_synth_test_audio()
        self.clear_log()
        self.reset_progress()
        self.set_current_indeterminate("正在启动处理流程")

        cmd = [
            sys.executable,
            str(main_py),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--work-dir",
            str(work_dir),
            "--config-json",
            str(config_json),
            "--pitch-config-json",
            str(pitch_config_json),
            "--master-volume",
            master_volume_arg,
            "--bit-depth",
            bit_depth_arg,
        ]

        self.append_log("启动命令：\n" + " ".join(cmd) + "\n")
        self.append_log(f"合成参数配置：{config_json}\n")
        self.append_log(f"音高提取参数配置：{pitch_config_json}\n")

        self.processing_started_at = time.perf_counter()
        self.set_running_state(True)
        self.status_var.set("处理中...")

        self.worker_thread = threading.Thread(
            target=self.run_subprocess,
            args=(cmd, main_py.parent),
            daemon=True,
        )
        self.worker_thread.start()

    # 运行处理子进程
    def run_subprocess(self, cmd: list[str], cwd: Path):
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            self.process = subprocess.Popen(
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

            assert self.process.stdout is not None

            for line in self.process.stdout:
                self.handle_output_line(line.rstrip("\n"))

            return_code = self.process.wait()

            if return_code == 0:
                self.post_ui_callback(self.on_process_success)
            else:
                self.post_ui_callback(self.on_process_failed, return_code)

        except Exception as exc:
            self.post_ui_callback(self.on_process_exception, exc)

    # 停止处理任务
    def stop_process(self):
        if self.process is None:
            return

        if self.process.poll() is None:
            self.terminate_process_without_blocking(self.process)
            self.append_log("\n已请求停止进程。\n")
            self.status_var.set("正在停止...")

    # 处理子进程输出
    def handle_output_line(self, line: str):
        if LIBRARY_PROGRESS_LINE_RE.search(line):
            return

        if line.startswith(PROGRESS_PREFIX):
            json_text = line[len(PROGRESS_PREFIX):]
            try:
                payload = json.loads(json_text)
            except json.JSONDecodeError:
                self.post_ui_event("log", line + "\n")
                return
            self.post_ui_event("progress", payload)
            return

        self.post_ui_event("log", line + "\n")

    # 停止不确定进度动画
    def stop_current_indeterminate(self):
        if not self.current_bar_is_indeterminate:
            return

        self.current_bar.stop()
        self.current_bar.configure(mode="determinate", variable=self.current_progress_var)
        self.current_bar_is_indeterminate = False

    # 根据进度数据更新界面
    def update_progress_from_payload(self, payload: dict):
        stage = str(payload.get("stage", "unknown"))
        percent = float(payload.get("percent", 0.0))
        current = payload.get("current", 0)
        total = payload.get("total", 0)
        message = str(payload.get("message", ""))
        unit = str(payload.get("unit", ""))
        extra = payload.get("extra", {}) or {}

        percent = max(0.0, min(100.0, percent))
        display_name = self.get_task_display_name(stage, message)

        if stage == "main":
            self.update_overall_from_main(message=message)
            self.update_current_for_main_message(message=message, percent=percent)
        else:
            self.stop_current_indeterminate()
            self.current_progress_var.set(percent)
            self.current_text_var.set(
                self.format_current_task_text(display_name, percent, current, total, unit, extra)
            )
            self.update_overall_from_child(stage=stage, percent=percent, message=message)

        if self.should_log_progress_event(stage, percent):
            self.append_log(f"[PROGRESS] {display_name} {percent:.2f}% {message}\n")

    # 判断是否记录进度事件
    def should_log_progress_event(self, stage: str, percent: float) -> bool:
        now = time.time()
        last = self.last_progress_log.get(stage)

        if last is None:
            self.last_progress_log[stage] = (percent, now)
            return True

        last_percent, last_time = last
        if percent >= 100.0 or abs(percent - last_percent) >= 5.0 or now - last_time >= 5.0:
            self.last_progress_log[stage] = (percent, now)
            return True

        return False

    # 根据主阶段更新总进度
    def update_overall_from_main(self, message: str):
        # main 只推进到大阶段起点，子任务进度会在这个区间内继续平滑推进。
        phase_floor = MAIN_MESSAGE_PHASE_FLOORS.get(message)
        if phase_floor is None:
            return

        self.main_progress_message = message or self.main_progress_message
        self.set_overall_percent(phase_floor, self.main_progress_message)

    # 根据子阶段更新总进度
    def update_overall_from_child(self, stage: str, percent: float, message: str = ""):
        overall = self.map_child_progress_to_overall(stage=stage, percent=percent)
        if overall is None:
            return

        if message:
            self.main_progress_message = self.get_phase_message_for_stage(stage, message)

        self.set_overall_percent(overall, self.main_progress_message)

    # 设置总进度百分比
    def set_overall_percent(self, percent: float, message: str = ""):
        percent = max(0.0, min(100.0, percent))
        # 允许同一阶段内连续上升，但避免偶发乱序事件让总进度倒退。
        self.max_overall_percent = max(self.max_overall_percent, percent)
        suffix = f"  当前阶段：{message}" if message else ""
        self.overall_progress_var.set(self.max_overall_percent)
        self.overall_text_var.set(f"总进度：{self.max_overall_percent:.2f}%{suffix}")

    # 插值计算进度
    def interpolate_progress(self, phase_name: str, item_index: int, item_count: int, percent: float) -> float:
        start, end = PHASE_RANGES[phase_name]
        item_count = max(1, int(item_count))
        item_index = max(0, min(int(item_index), item_count - 1))
        local = (item_index + percent / 100.0) / item_count
        return start + (end - start) * local

    # 映射子任务进度到总进度
    def map_child_progress_to_overall(self, stage: str, percent: float) -> float | None:
        if stage == "separator_drums:load":
            start, end = PHASE_RANGES["drum_separator"]
            local = 0.00 + (percent / 100.0) * 0.20
            return start + (end - start) * local

        if stage == "separator_drums:demix":
            start, end = PHASE_RANGES["drum_separator"]
            local = 0.20 + (percent / 100.0) * 0.65
            return start + (end - start) * local

        if stage == "separator_drums:save":
            start, end = PHASE_RANGES["drum_separator"]
            local = 0.85 + (percent / 100.0) * 0.15
            return start + (end - start) * local

        if stage.startswith("separator_drums"):
            start, end = PHASE_RANGES["drum_separator"]
            return start + (end - start) * percent / 100.0

        if stage == "separator:infer":
            start, end = PHASE_RANGES["main_separator"]
            local = 0.75 + (percent / 100.0) * 0.25
            return start + (end - start) * local

        if stage.startswith("separator"):
            start, end = PHASE_RANGES["main_separator"]
            return start + (end - start) * percent / 100.0

        if stage.startswith("pitch_extraction_drums"):
            parts = stage.split(":", 1)
            if len(parts) == 2 and parts[1] in DRUM_PROGRESS_ORDER:
                index = DRUM_PROGRESS_ORDER.index(parts[1])
                return self.interpolate_progress("drum_pitch", index, len(DRUM_PROGRESS_ORDER), percent)
            start, end = PHASE_RANGES["drum_pitch"]
            return start + (end - start) * percent / 100.0

        if stage.startswith("pitch_extraction_"):
            name = stage.removeprefix("pitch_extraction_").split(":", 1)[0]
            if name == "vocal":
                name = "vocals"
            if name in MELODIC_PROGRESS_ORDER:
                index = MELODIC_PROGRESS_ORDER.index(name)
                return self.interpolate_progress("melodic_pitch", index, len(MELODIC_PROGRESS_ORDER), percent)

        if stage.startswith("generator:"):
            parts = stage.split(":")
            if len(parts) >= 3 and parts[1] == "drums":
                item = parts[2]
            elif len(parts) >= 2:
                item = parts[1]
            else:
                item = ""
            if item in SYNTH_PROGRESS_ORDER:
                index = SYNTH_PROGRESS_ORDER.index(item)
                return self.interpolate_progress("synthesis", index, len(SYNTH_PROGRESS_ORDER), percent)

        if stage == "final_mix":
            start, end = PHASE_RANGES["final_mix"]
            return start + (end - start) * percent / 100.0

        return None

    # 获取阶段提示文本
    def get_phase_message_for_stage(self, stage: str, fallback: str = "") -> str:
        if stage.startswith("separator_drums"):
            return "正在分离鼓组音频"
        if stage.startswith("separator"):
            return "正在分离音频"
        if stage.startswith("pitch_extraction_drums"):
            return "正在提取鼓点"
        if stage.startswith("pitch_extraction_"):
            return "正在提取旋律声部音高"
        if stage.startswith("generator:"):
            return "正在合成芯片音轨"
        if stage == "final_mix":
            return "正在混合最终音频"
        return fallback

    # 根据主阶段更新当前任务
    def update_current_for_main_message(self, message: str, percent: float):
        if not message:
            return

        self.stop_current_indeterminate()

        if percent >= 100.0 or "完成" in message:
            self.current_progress_var.set(100.0)
            self.current_text_var.set(f"当前任务：{message}")
        else:
            self.current_progress_var.set(0.0)
            self.current_text_var.set(f"当前任务：{message}")

    # 格式化剩余时间
    def format_eta_text(self, extra: dict) -> str:
        eta = extra.get("eta_seconds") if isinstance(extra, dict) else None
        if eta is None:
            return ""

        try:
            eta_float = max(0.0, float(eta))
        except (TypeError, ValueError):
            return ""

        minutes = int(eta_float // 60)
        seconds = int(round(eta_float % 60))
        if minutes > 0:
            return f"  预计剩余：{minutes}分{seconds:02d}秒"
        return f"  预计剩余：{seconds}秒"

    # 格式化当前任务文本
    def format_current_task_text(self, display_name, percent, current, total, unit, extra=None):
        eta_text = self.format_eta_text(extra or {})
        if unit == "percent":
            return f"当前任务：{display_name}  {percent:.2f}%{eta_text}"

        try:
            total_value = int(total)
        except (TypeError, ValueError):
            total_value = 0

        if total_value <= 0:
            return f"当前任务：{display_name}  {percent:.2f}%{eta_text}"

        return f"当前任务：{display_name}  {percent:.2f}%  ({current}/{total}){eta_text}"

    # 获取任务显示名称
    def get_task_display_name(self, stage: str, message: str = "") -> str:
        if stage == "main":
            return "总流程"
        if stage.startswith("separator") and not stage.startswith("separator_drums"):
            return "正在分离音频"
        if stage.startswith("separator_drums"):
            return "正在分离鼓组音频"
        if stage in {"pitch_extraction_vocals", "pitch_extraction_vocal"}:
            return "正在提取人声音高"
        if stage.startswith("pitch_extraction_drums"):
            parts = stage.split(":", 1)
            if len(parts) == 2:
                display = DRUM_DISPLAY_NAMES.get(parts[1], parts[1])
                return f"正在提取{display}鼓点"
            return "正在提取鼓点"
        if stage.startswith("pitch_extraction_"):
            name = stage.removeprefix("pitch_extraction_").split(":", 1)[0]
            display = CHANNEL_DISPLAY_NAMES.get("vocal" if name == "vocals" else name, name)
            return f"正在识别{display}音高"
        if stage.startswith("generator:"):
            parts = stage.split(":")
            if len(parts) >= 3 and parts[1] == "drums":
                drum = parts[2]
                display = DRUM_DISPLAY_NAMES.get(drum, drum)
                return f"正在合成{display}音轨"
            if len(parts) >= 2:
                channel = parts[1]
                display = CHANNEL_DISPLAY_NAMES.get(channel, channel)
                return f"正在合成{display}音轨"
        if stage == "final_mix":
            return "正在混合最终音频"
        return message or stage


    # UI 状态

    # 切换日志面板
    def toggle_log_panel(self):
        if self.log_visible:
            self.hide_log_panel()
        else:
            self.show_log_panel()

    # 显示日志面板
    def show_log_panel(self):
        if self.log_visible:
            return

        # 日志栏只在左侧流程区域底部展开
        self.log_frame.pack(
            fill=tk.BOTH,
            expand=False,
            pady=(10, 0),
            side=tk.BOTTOM,
        )

        self.log_button_text_var.set("隐藏日志")
        self.log_visible = True

    # 隐藏日志面板
    def hide_log_panel(self):
        if not self.log_visible:
            return

        self.log_frame.pack_forget()
        self.log_button_text_var.set("显示日志")
        self.log_visible = False

    # 追加日志内容
    def append_log(self, text: str):
        if not text:
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)

        try:
            line_count = int(self.log_text.index("end-1c").split(".")[0])
            if line_count > 6000:
                self.log_text.delete("1.0", f"{line_count - 5000}.0")
        except Exception:
            pass

        if self.log_visible:
            self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # 清空日志内容
    def clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # 重置处理进度
    def reset_progress(self):
        self.overall_progress_var.set(0.0)
        self.current_progress_var.set(0.0)
        self.overall_text_var.set("总进度：0.00%")
        self.current_text_var.set("当前任务：等待开始")
        self.current_bar.configure(mode="determinate", variable=self.current_progress_var)
        self.max_overall_percent = 0.0
        self.main_progress_message = ""
        self.last_progress_log: dict[str, tuple[float, float]] = {}
        self.current_bar_is_indeterminate = False

    # 设置不确定进度状态
    def set_current_indeterminate(self, message: str):
        self.current_text_var.set(f"当前任务：{message}")
        self.current_progress_var.set(0.0)
        if not self.current_bar_is_indeterminate:
            self.current_bar.configure(mode="indeterminate")
            self.current_bar.start(10)
            self.current_bar_is_indeterminate = True

    # 设置参数编辑状态
    def set_parameter_editing_enabled(self, enabled: bool):
        for widget, normal_state in self.parameter_widgets:
            try:
                widget.configure(state=normal_state if enabled else "disabled")
            except Exception:
                pass

        if hasattr(self, "reset_params_button"):
            self.reset_params_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.update_stereo_parameter_state()

    # 设置任务运行状态
    def set_running_state(self, running: bool):
        self.processing_running = bool(running)
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.set_parameter_editing_enabled(not running)

    # 获取处理耗时文本
    def get_processing_elapsed_text(self) -> str:
        if self.processing_started_at <= 0.0:
            return "0.00 秒"

        elapsed = max(0.0, time.perf_counter() - self.processing_started_at)
        hours, remainder = divmod(elapsed, 3600.0)
        minutes, seconds = divmod(remainder, 60.0)

        if hours >= 1.0:
            return f"{int(hours)} 小时 {int(minutes)} 分 {seconds:.2f} 秒"
        if minutes >= 1.0:
            return f"{int(minutes)} 分 {seconds:.2f} 秒"
        return f"{seconds:.2f} 秒"

    # 结束处理计时
    def finish_processing_timer(self) -> str:
        elapsed_text = self.get_processing_elapsed_text()
        self.processing_started_at = 0.0
        return elapsed_text

    # 处理任务成功
    def on_process_success(self):
        elapsed_text = self.finish_processing_timer()
        self.stop_current_indeterminate()
        self.set_running_state(False)
        self.status_var.set(f"处理完成（用时：{elapsed_text}）")
        self.overall_progress_var.set(100.0)
        self.current_progress_var.set(100.0)
        self.overall_text_var.set("总进度：100.00%  处理完成")
        self.mark_output_player_finished()
        self.append_log(f"\n处理完成，本次用时：{elapsed_text}。\n")
        messagebox.showinfo("完成", f"音频处理完成。\n处理时间：{elapsed_text}")

    # 处理任务失败
    def on_process_failed(self, return_code: int):
        elapsed_text = self.finish_processing_timer()
        self.stop_current_indeterminate()
        self.set_running_state(False)
        self.status_var.set(f"处理失败（用时：{elapsed_text}）")
        self.set_player_ready("output", True)
        self.refresh_player_source("output", reset_position=True)
        self.append_log(f"\n处理失败，退出码：{return_code}，本次用时：{elapsed_text}。\n")
        messagebox.showerror(
            "处理失败",
            f"处理失败，退出码：{return_code}。\n处理时间：{elapsed_text}\n请点击“显示日志”查看详情。",
        )

    # 处理任务异常
    def on_process_exception(self, exc: Exception):
        elapsed_text = self.finish_processing_timer()
        self.stop_current_indeterminate()
        self.set_running_state(False)
        self.status_var.set(f"程序异常（用时：{elapsed_text}）")
        self.set_player_ready("output", True)
        self.refresh_player_source("output", reset_position=True)
        self.append_log(f"\n程序异常：{exc}，本次用时：{elapsed_text}。\n")
        messagebox.showerror("程序异常", f"{exc}\n处理时间：{elapsed_text}")

    # 关闭主界面
    def on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("确认退出", "任务仍在运行，确定要停止并退出吗？"):
                return

        self._closing = True
        self.synth_test_task_token += 1
        self.stop_all_audio()

        if self.process is not None and self.process.poll() is None:
            self.terminate_process_without_blocking(self.process)

        self.destroy()


# 运行界面程序
def main():
    app = ChiptuneUI()
    app.mainloop()


if __name__ == "__main__":
    main()

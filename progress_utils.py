from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


PROGRESS_PREFIX = "PROGRESS "

# 规范化进度数值
def clamp_progress(current: int | float, total: int | float) -> tuple[int, int, float]:
    total_int = int(total) if total is not None else 0
    current_int = int(current) if current is not None else 0

    if total_int <= 0:
        total_int = 0
        current_int = max(0, current_int)
        percent = 100.0
    else:
        current_int = max(0, min(current_int, total_int))
        percent = current_int / total_int * 100.0

    return current_int, total_int, round(percent, 2)

# 输出统一进度事件
def emit_progress(
    stage: str,
    current: int | float,
    total: int | float,
    message: str = "",
    *,
    unit: str = "step",
    extra: dict[str, Any] | None = None,
    stream=None,
) -> None:
    if stream is None:
        stream = sys.stdout

    current_int, total_int, percent = clamp_progress(current, total)

    # main.py 可以通过环境变量给子进程加唯一进度 stage。
    # 例如同一个 pitch_extraction_drums.py 被 kick/snare/hh 多次调用时，
    # UI 需要知道当前属于哪一个鼓件，才能把总进度映射到正确位置。
    stage_text = os.environ.get("PROGRESS_STAGE_OVERRIDE") or str(stage)

    payload: dict[str, Any] = {
        "type": "progress",
        "stage": stage_text,
        "current": current_int,
        "total": total_int,
        "percent": percent,
        "message": str(message),
        "unit": unit,
        "done": total_int == 0 or current_int >= total_int,
        "timestamp": round(time.time(), 3),
    }

    if extra:
        payload["extra"] = extra

    print(
        PROGRESS_PREFIX + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        file=stream,
        flush=True,
    )

# 进度跟踪器
class ProgressTracker:

    # 初始化进度跟踪器
    def __init__(
        self,
        stage: str,
        total: int,
        *,
        unit: str = "step",
    ):
        self.stage = stage
        self.total = max(0, int(total))
        self.unit = unit
        self.current = 0

    # 更新进度
    def update(
        self,
        current: int | None = None,
        message: str = "",
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if current is None:
            self.current += 1
        else:
            self.current = int(current)

        emit_progress(
            stage=self.stage,
            current=self.current,
            total=self.total,
            message=message,
            unit=self.unit,
            extra=extra,
        )

    # 推进进度
    def step(
        self,
        message: str = "",
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.update(
            current=None,
            message=message,
            extra=extra,
        )

    # 完成进度
    def finish(self, message: str = "完成") -> None:
        self.update(
            current=self.total,
            message=message,
        )

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from progress_utils import emit_progress


# usage
# python Separator.py --input test.mp3 --output_dir output


MODEL_SLUG = "roformer-model-bs-roformer-sw-by-jarredou"    #BS-RoFormer Model
CONFIG_NAME = "BS-Rofo-SW-Fixed.yaml"                       #Config File
CKPT_NAME = "BS-Rofo-SW-Fixed.ckpt"                         #Weight File

# 运行外部命令
def run(cmd, *, quiet: bool = False):
    if quiet:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    else:
        subprocess.run(cmd, check=True)


# 获取音频时长
def get_audio_duration_seconds(input_path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(input_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        return max(0.0, float(result.stdout.strip()))
    except Exception:
        return None


# 估算分离耗时
def estimate_separator_seconds(duration_sec: float | None) -> float:
    if duration_sec is None:
        return 30.0
    return max(10.0, duration_sec * 0.08)


# 运行带预估进度的命令
def run_with_estimated_progress(cmd, *, stage: str, message: str, estimated_seconds: float):
    started_at = time.time()
    emit_progress(stage, 0, 1000, message, unit="percent", extra={"eta_seconds": estimated_seconds})

    # 分离器自身可能输出 tqdm/文本进度条。这里不转发原始输出，避免污染 UI 日志，
    # 由本函数定时发出结构化 PROGRESS 事件给 UI。
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    try:
        while process.poll() is None:
            elapsed = time.time() - started_at
            ratio = min(0.95, elapsed / max(estimated_seconds, 1e-6))
            eta = max(0.0, estimated_seconds - elapsed)
            emit_progress(stage, int(ratio * 1000), 1000, message, unit="percent", extra={"eta_seconds": eta})
            time.sleep(0.5)

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
    finally:
        if process.poll() is None:
            process.terminate()

    emit_progress(stage, 1000, 1000, message + "完成", unit="percent")

# 转换并规范化输入音频
def normalize_audio(input_path, output_wav):
    run([
        "ffmpeg",               # using ffmpeg
        "-y",                   # overwrite existing output file
        "-i", str(input_path),  # input file path
        "-vn",                  # disable video stream
        "-acodec", "pcm_s16le", # use 16-bit PCM codec
        "-ar", "44100",         # sample rate
        "-ac", "2",             # channel
        str(output_wav),        # output
    ], quiet=True)

# 下载分离模型
def download_model(models_dir):
    run([
        "bs-roformer-download",         
        "--model", MODEL_SLUG,          
        "--output-dir", str(models_dir),
    ])

# 分离主音轨
def separate(input_audio, output_dir, models_dir):
    emit_progress("separator", 0, 4, "准备分离主音频")

    input_audio = Path(input_audio)
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)

    # create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # construct path
    model_dir = models_dir / MODEL_SLUG
    config_path = model_dir / CONFIG_NAME
    ckpt_path = model_dir / CKPT_NAME

    # check if model exists. if not, download model.
    if not (config_path.exists() and ckpt_path.exists()):
        emit_progress("separator", 1, 4, "正在下载主分离模型")
        download_model(models_dir)
    else:
        emit_progress("separator", 1, 4, "主分离模型已就绪")

    # create temporary directory 
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir = tmpdir / "inputs"
        input_dir.mkdir()

        wav_path = input_dir / f"{input_audio.stem}.wav"
        emit_progress("separator", 2, 4, "正在标准化输入音频")
        normalize_audio(input_audio, wav_path)

        # separate
        emit_progress("separator", 3, 4, "正在执行主音频分离")
        duration_sec = get_audio_duration_seconds(wav_path)
        estimated_seconds = estimate_separator_seconds(duration_sec)
        run_with_estimated_progress(
            [
                "bs-roformer-infer",
                "--config_path", str(config_path),
                "--model_path", str(ckpt_path),
                "--input_folder", str(input_dir),
                "--store_dir", str(output_dir),
            ],
            stage="separator:infer",
            message="正在执行主音频分离",
            estimated_seconds=estimated_seconds,
        )
        emit_progress("separator", 4, 4, "主音频分离完成")

# main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()                      # create a argument parser
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models_dir", default="./models")
    args = parser.parse_args()                              # parse arguments

    separate(args.input, args.output_dir, args.models_dir)  # separate


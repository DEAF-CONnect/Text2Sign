# audio_webm_encoder.py
"""
48kHz mono 16-bit PCM 바이너리를 WebM(Opus)로 인코딩하는 헬퍼.

- 입력: RNNoise로 노이즈 제거된 PCM (s16le, 48kHz, mono)
- 출력: WebM 컨테이너 안에 Opus로 인코딩된 바이트(webm_bytes)
"""

import subprocess

SAMPLE_RATE = 48000
CHANNELS = 1


def pcm_bytes_to_webm_bytes(pcm_bytes: bytes, bitrate: str = "32k") -> bytes:
    """
    48kHz mono 16-bit PCM bytes -> WebM(Opus) 바이너리로 변환.

    STT 서버에 HTTP POST로 바로 보낼 때 사용.

    :param pcm_bytes: s16le, mono, 48kHz PCM 데이터
    :param bitrate: Opus 비트레이트(예: "16k", "32k", "64k")
    :return: WebM 컨테이너로 감싼 Opus 바이트
    """
    if not pcm_bytes:
        return b""

    cmd = [
        "ffmpeg",
        "-f", "s16le",               # raw PCM, signed 16bit little-endian
        "-ar", str(SAMPLE_RATE),     # sample rate
        "-ac", str(CHANNELS),        # channels
        "-i", "-",                   # stdin에서 입력
        "-c:a", "libopus",           # audio codec = Opus
        "-b:a", bitrate,             # bitrate
        "-f", "webm",                # container = WebM
        "-"                          # stdout으로 출력
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate(input=pcm_bytes)

    if proc.returncode != 0:
        print("ffmpeg stderr:\n", stderr_data.decode("utf-8", "ignore"))
        raise RuntimeError(f"ffmpeg WebM_Opus 인코딩 실패 (코드 {proc.returncode})")

    return stdout_data

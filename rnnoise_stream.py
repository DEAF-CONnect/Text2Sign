# rnnoise_stream.py
from typing import List, Tuple
import numpy as np
from pyrnnoise import RNNoise

FRAME_SIZE = 480          # RNNoise가 사용하는 프레임 크기 (샘플 개수)
SAMPLE_RATE = 48000       # RNNoise 기본 샘플레이트
BYTES_PER_SAMPLE = 2      # int16 = 2 bytes

class RnnoiseStream:
    """
    Unity에서 넘어오는 16-bit PCM(48kHz, mono) 스트림에
    RNNoise를 프레임 단위로 실시간 적용하는 헬퍼 클래스.
    """
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"RNNoise는 기본적으로 {SAMPLE_RATE}Hz 기준이야. "
                             f"Unity도 {SAMPLE_RATE}Hz로 맞추는 게 편해.")
        self.denoiser = RNNoise(sample_rate=sample_rate)
        # 남은 샘플들(프레임이 안 될 만큼 애매하게 남은 것들)
        self._remnant = np.zeros((1, 0), dtype=np.int16)  # shape: (channels=1, samples)

    def process_bytes(self, pcm_bytes: bytes) -> Tuple[bytes, List[float]]:
        """
        Unity에서 온 PCM bytes를 넣으면:
        - RNNoise를 돌린 클린 PCM bytes를 반환
        - 각 프레임별 음성 확률 리스트도 같이 반환

        입력/출력 포맷:
          - 입력: 16-bit PCM, mono, 48kHz, little-endian
          - 출력: 같은 포맷(노이즈 제거된 버전)
        """
        if not pcm_bytes:
            return b"", []

        # bytes -> int16 numpy (mono)
        new_samples = np.frombuffer(pcm_bytes, dtype=np.int16).reshape(1, -1)  # (1, N)
        # 이전에 남아있던 remnant와 이어붙이기
        all_samples = np.concatenate([self._remnant, new_samples], axis=1)     # (1, total)

        total_samples = all_samples.shape[1]
        n_full_frames = total_samples // FRAME_SIZE
        used_samples = n_full_frames * FRAME_SIZE

        if n_full_frames == 0:
            # 아직 1프레임도 안 될 만큼만 들어온 상황 → remnant에 저장만 해둠
            self._remnant = all_samples
            return b"", []

        # RNNoise에 넘길 부분 (480 샘플 단위로 딱 떨어지는 부분)
        process_chunk = all_samples[:, :used_samples]        # (1, n_full_frames * 480)
        # 남는 나머지는 remnant로 저장
        self._remnant = all_samples[:, used_samples:]        # (1, remainder)

        out_frames = []
        vad_probs: List[float] = []

        # pyrnnoise의 denoise_chunk는 [channels, samples] 배열을 받고
        # 내부에서 480샘플씩 잘라서 처리하면서
        # (speech_probabilities, denoised_frame)을 yield 해줌 :contentReference[oaicite:2]{index=2}
        for speech_prob, den_frame in self.denoiser.denoise_chunk(process_chunk):
            # speech_prob: 채널별 확률 배열 (mono면 길이 1)
            vad_probs.append(float(speech_prob[0]))
            # den_frame: shape = (channels, FRAME_SIZE)
            out_frames.append(den_frame)

        if not out_frames:
            return b"", vad_probs

        # 프레임들을 다시 하나의 chunk로 이어붙이기
        out_chunk = np.concatenate(out_frames, axis=1)  # (1, n_full_frames * 480)
        out_chunk = out_chunk.astype(np.int16)

        # numpy -> bytes
        out_bytes = out_chunk.tobytes()
        return out_bytes, vad_probs

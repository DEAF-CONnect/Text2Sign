import librosa
import numpy as np
import json
from pathlib import Path

def extract_beats(audio_path: str, out_json: str, double_tempo=True):
    # 1) 오디오 로드
    y, sr = librosa.load(audio_path)

    # 2) 템포 + 비트 추출
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # tempo가 numpy 배열로 나오는 경우 처리
    try:
        tempo_val = float(tempo)
    except Exception:
        tempo_val = float(tempo[0])

    print(f"[INFO] librosa tempo: {tempo_val:.2f} BPM")

    # 3) 시간(초 단위) 변환
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"[INFO] original beats: {len(beat_times)}개")

    # 4) 2배 템포 보정
    if double_tempo and len(beat_times) > 1:
        beat_times = np.array(beat_times)

        # 인접 박자 중간값 계산
        mid_beats = (beat_times[:-1] + beat_times[1:]) / 2.0

        # 원래 박 + 새 박 합치기
        beat_times = np.sort(np.concatenate([beat_times, mid_beats]))

        print(f"[INFO] doubled beats: {len(beat_times)}개 (2배 템포 적용됨)")

    # 5) JSON 저장
    data = {
        "beats": beat_times.tolist()
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 저장 완료: {out_path}")

if __name__ == "__main__":
    extract_beats("QWER-고민중독.mp3", "song_beats.json", double_tempo=True)

# Text2Sign — Real-Time Korean Sign Language Generation

Text2Sign은 **음성(STT) 또는 텍스트 입력을 LLM으로 처리하여**,  
**한국어 문장을 실시간 Sign Language Animation(수어 동작)** 으로 변환하는 시스템입니다.

Unity와 WebSocket으로 연결되며,  
LLM → Gloss Plan → Template Retrieval → Keypoint Frames → Unity 전송  
의 파이프라인을 실시간으로 수행합니다.

---

# 🚀 주요 특징

### ✔ **1. 실시간 WebSocket 서버 (`realtime_sign_llm_server.py`)**
- Unity가 음성/텍스트를 보내면 즉시 수어 모션 프레임(JSONL) 스트리밍.
- 토큰 단위(gloss)로 나누어 **단어별로 바로 동작 재생** 가능.
- 프레임 t(ms), pose(25점), left/right hand(21점) 모두 MediaPipe 형태.

### ✔ **2. LLM 기반 Gloss Plan**
- `llm_preprocessor.py`  
  - 한국어 문장을 LLM(OpenRouter)으로 gloss sequence + seconds 로 변환
  - 조사/어미 제거, punctuation pause 처리, OOV 보정

### ✔ **3. Retrieval 기반 Keypoint 생성**
- `text2sign_retrieval_full.py`  
  - gloss → 템플릿(npz) → 리샘플링 → crossfade → JSONL 포맷으로 변환

### ✔ **4. RNNoise 기반 실시간 노이즈 제거**
- `rnnoise_stream.py`  
  - Unity의 PCM 오디오를 서버에서 실시간 노이즈 제거 후 STT로 전달 (선택적 기능)

### ✔ **5. 배치/도구 스크립트 지원**
- `batch_lyrics_llm.py`  
- `lrc_to_plan_llm.py`  
- `make_manifest_fast.py`  
- `shrink_manifest.py`

모두 최신 구조와 호환됨.

---

# 📁 디렉토리 구조

```
Text2Sign/
│
├── realtime_sign_llm_server.py      # ⭐ 프로젝트 메인 (Unity와 실시간 통신)
├── text2sign_retrieval_full.py      # 키프레임 조립 엔진
├── llm_preprocessor.py              # LLM gloss planning
├── rnnoise_stream.py                # 오디오 노이즈 제거
├── service.py                       # REST API 서버 (옵션)
│
├── batch_lyrics_llm.py              # 가사 전체 자동 변환
├── lrc_to_plan_llm.py               # LRC → Plan 변환기
├── make_manifest_fast.py            # templates 폴더 스캔하여 manifest 생성
├── shrink_manifest.py               # manifest 용량 축소
│
├── data/
│   └── templates_*/                 # 템플릿(pose) NPZ 데이터
│
└── out/
    └── ...                          # 생성된 JSON/JSONL 출력
```

---

# 🔧 설치 방법

## 1) Python 환경
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

필수 패키지:
- `numpy`
- `fastapi`
- `uvicorn`
- `openai` (OpenRouter용)
- `pyrnnoise` (옵션)
- `librosa` (박자 추출용)

---

# 🔑 환경 변수 설정

아래 값들은 민감정보이므로 Git에 올리면 안 됩니다.

```
export OPENROUTER_API_KEY="YOUR_API_KEY"
export TEMPLATES_ROOT="/mnt/text2sign/home/ubuntu/data/templates_xxx"
export MANIFEST_PATH="/mnt/text2sign/home/ubuntu/manifest_merged.json"
export API_KEY="OPTIONAL_SERVER_AUTH"
```

Windows PowerShell:

```
setx OPENROUTER_API_KEY "YOUR_API_KEY"
```

---

# ▶ 서버 실행

## ⭐ **실시간 서버 실행 (메인)**

```
python realtime_sign_llm_server.py
```

서버가 준비되면 Unity에서 아래 WebSocket 주소로 접속:

```
ws://YOUR_EC2_IP:8000/ws/unity
```

---

# 🧩 Unity 연동

### Unity → 서버

- 오디오 PCM(16bit, mono 48kHz) 또는 STT 텍스트 전송
- 웹소켓 메시지(JSON) 형태:

```json
{
  "text": "안녕하세요 만나서 반갑습니다"
}
```

또는 오디오 스트리밍.

---

### 서버 → Unity

토큰별 pose 프레임 스트리밍:

```json
{
  "t": 420,
  "pose": [...],
  "left_hand": [...],
  "right_hand": [...],
  "token": "안녕하세요"
}
```

Unity에서는 메시지를 받을 때마다 바로 해당 pose로 캐릭터 본을 업데이트하면 됨.

---

# 🎼 음악 기반 기능

### 비트 추출 (`extract_beats.py`)

```
python extract_beats.py input.mp3 out.json
```

출력 예:

```json
{
  "beats": [0.52, 1.04, 1.56, ...]
}
```

Unity에서 박자 진동(타이밍 트리거)에 사용 가능.

---

# 📦 Manifest 생성/관리

### 템플릿 전체 스캔 → manifest 생성

```
python make_manifest_fast.py \
  --templates_root data/templates_realsen_v3 \
  --out manifest_realsen_v3.json
```

### 용량 줄이기 (글로스당 샘플 2개만)

```
python shrink_manifest.py \
  --manifest manifest_realsen_v3.json \
  --out_manifest manifest_realsen_v3_small.json \
  --max_per_gloss 2
```

---

# ⚙ LLM 도구 스크립트

### 1) 문장 하나 plan 생성
```
python llm_preprocessor.py \
  --text "오늘 공연 와줘서 감사합니다" \
  --templates_root data/templates_crowd_v3 \
  --manifest_json manifest_crowd_v3.json \
  --out_plan plan.json
```

### 2) 가사 전체 변환
```
python batch_lyrics_llm.py
```

### 3) LRC 가사 자동 변환
```
python lrc_to_plan_llm.py lyrics.txt --out song_plan.json
```

---

# 🔥 전체 파이프라인 흐름

```
Unity (Audio/Text)
       ↓
Realtime WebSocket Server
       ↓
LLM Gloss Planning (llm_preprocessor)
       ↓
Template Retrieval (text2sign_retrieval_full)
       ↓
Frame Assembly (JSONL)
       ↓
Unity Avatar Animation
```

---

# 🛡 민감정보 관리

`.env`, `settings.json`, 절대경로, IP 주소 등은 **절대 Git에 올리지 마세요.**

필요한 값은 모두:

```
YOUR_EC2_IP
YOUR_LOCAL_PATH
YOUR_API_KEY
```

와 같은 placeholder 로 대체해서 공유하세요.

---

# 📞 문의 / 유지보수

이 프로젝트는 유지보수가 용이하도록 모듈화되어 있으며,  
Unity 쪽 최적화, 템플릿 개선, LLM 프롬프트 튜닝까지 확장 가능합니다.

---

# 🎉 끝!

복사해서 `README.md` 로 바로 사용하면 됩니다.  
더 “기업용/학회 제출용” 스타일로 꾸며달라고 하면 그 버전도 다시 만들어줄게.

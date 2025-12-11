# audio_server.py  (RNNoise + WebM(Opus) + END 텍스트 신호)

from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
import websockets

from rnnoise_stream import RnnoiseStream
from audio_webm_encoder import pcm_bytes_to_webm_bytes  # ★ WebM(Opus) 인코더

STT_WS_URL = "ws://YOUR/EC2IP"


async def send_to_stt_ws(webm_bytes: bytes) -> dict:
    """
    WebM(Opus) 바이너리를 자바 STT WebSocket 서버로 전송하고,
    자바 쪽에서 보내주는 텍스트 메시지들을 모아서 dict로 반환.
    """
    if not webm_bytes:
        raise ValueError("STT에 보낼 WebM 데이터가 비어 있습니다.")

    texts: List[str] = []

    async with websockets.connect(STT_WS_URL, max_size=None) as stt_ws:
        # 1) WebM 바이너리 전송
        await stt_ws.send(webm_bytes)
        print(f"[STT-WS] sent {len(webm_bytes)} bytes to {STT_WS_URL} (WebM/Opus)")

        # 2) END 텍스트로 "오디오 끝" 알리기
        await stt_ws.send("END")
        print("[STT-WS] sent END marker (text)")

        # 3) STT 서버에서 오는 텍스트를 최대 N번 / 최대 T초 동안 받기
        try:
            MAX_MESSAGES = 5
            RECV_TIMEOUT = 10.0  # STT가 처리할 시간 여유

            for _ in range(MAX_MESSAGES):
                try:
                    msg = await asyncio.wait_for(stt_ws.recv(), timeout=RECV_TIMEOUT)
                except asyncio.TimeoutError:
                    print("[STT-WS] recv timeout, stop listening further")
                    break

                if isinstance(msg, str):
                    texts.append(msg)
                    print(f"[STT-WS] recv text: {msg}")
                else:
                    print(f"[STT-WS] unexpected binary len={len(msg)}")
        except websockets.exceptions.ConnectionClosedOK:
            print("[STT-WS] connection closed normally")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[STT-WS] connection closed with error: {e}")

    final_text = texts[-1] if texts else ""
    return {
        "ok": True,
        "text": final_text,
        "partials": texts,
    }


app = FastAPI()

SAMPLE_RATE = 48000
FRAME_SIZE = 480
BYTES_PER_FRAME = FRAME_SIZE * 2
FRAME_SEC = 0.01

MIN_SEGMENT_SEC = 1.0
SILENCE_TAIL_SEC = 0.4
SILENCE_THRESHOLD = 0.3
MAX_SEGMENT_SEC = 5.0


def should_cut_segment(vads: List[float], pcm_len_bytes: int) -> bool:
    if pcm_len_bytes <= 0:
        return False

    total_frames = pcm_len_bytes // BYTES_PER_FRAME
    total_sec = total_frames * FRAME_SEC

    if total_sec < MIN_SEGMENT_SEC:
        return False

    if total_sec >= MAX_SEGMENT_SEC:
        return True

    if not vads:
        return False

    tail_frames = int(SILENCE_TAIL_SEC / FRAME_SEC)
    if tail_frames <= 0:
        return False

    if len(vads) <= tail_frames:
        tail = vads
    else:
        tail = vads[-tail_frames:]

    if not tail:
        return False

    avg_tail = sum(tail) / len(tail)
    return avg_tail < SILENCE_THRESHOLD


async def process_and_send_segment(ws: WebSocket, segment_pcm: bytearray, segment_vads: List[float]) -> None:
    """
    segment_pcm: RNNoise 통과 후 누적된 깨끗한 PCM (int16 mono, 48kHz)
    1) PCM → WebM(Opus) 인코딩
    2) WebM(Opus)을 자바 STT WebSocket 서버로 전송
    3) STT 결과를 Unity 쪽으로 JSON으로 전달
    """
    try:
        pcm_bytes = bytes(segment_pcm)
        if not pcm_bytes:
            print("[SEGMENT] 빈 PCM 세그먼트 → STT 스킵")
            return

        print(f"[SEGMENT] encoding PCM {len(pcm_bytes)} bytes to WebM(Opus)...")

        # ★ 여기서 WebM(Opus)로 인코딩
        webm_bytes = pcm_bytes_to_webm_bytes(pcm_bytes)
        print(f"[SEGMENT] encoded WebM size={len(webm_bytes)} bytes")

        # ★ WebM(Opus) 그대로 STT WebSocket으로 전송
        stt_result = await send_to_stt_ws(webm_bytes)
        print(f"[STT-WS] result: {stt_result}")

        # Unity로 결과 전송
        await ws.send_json({
            "type": "stt_result",
            "ok": True,
            "result": stt_result,
            "meta": {
                "pcm_bytes": len(pcm_bytes),
                "webm_bytes": len(webm_bytes),
                "vad_frames": len(segment_vads),
            },
        })
        print("[WS] STT result sent to Unity")

    except Exception as e:
        print("[ERR] process_and_send_segment 오류:", e)
        try:
            await ws.send_json({
                "type": "stt_error",
                "ok": False,
                "error": str(e),
            })
        except Exception:
            pass


@app.websocket("/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    print("[WS] audio client connected")

    denoiser = RnnoiseStream(sample_rate=SAMPLE_RATE)

    segment_pcm = bytearray()
    segment_vads: List[float] = []

    chunk_count = 0
    total_bytes = 0

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                print("[WS] WebSocketDisconnect 발생 → 루프 종료")
                break
            except RuntimeError as e:
                print(f"[WS] RuntimeError in receive(): {e} → 루프 종료")
                break

            print(
                f"[DEBUG] type={message.get('type')} "
                f"bytes_len={len(message.get('bytes', b'')) if message.get('bytes') is not None else None} "
                f"text={message.get('text')}"
            )

            mtype = message.get("type")
            if mtype == "websocket.disconnect":
                print("[WS] websocket.disconnect 메시지 수신 → 루프 종료")
                break

            if mtype == "websocket.receive":
                if message.get("bytes") is not None:
                    pcm_chunk: bytes = message["bytes"]

                    chunk_count += 1
                    total_bytes += len(pcm_chunk)
                    approx_sec = total_bytes / (SAMPLE_RATE * 2)
                    print(
                        f"[WS] chunk #{chunk_count}, size={len(pcm_chunk)} bytes, "
                        f"total={total_bytes} bytes (~{approx_sec:.2f}s)"
                    )

                    clean_bytes, vad_probs = denoiser.process_bytes(pcm_chunk)

                    if clean_bytes:
                        segment_pcm.extend(clean_bytes)
                    if vad_probs:
                        segment_vads.extend(vad_probs)

                    if should_cut_segment(segment_vads, len(segment_pcm)):
                        print(
                            f"[SEGMENT] end detected, "
                            f"len={len(segment_pcm)} bytes, vads={len(segment_vads)} frames"
                        )
                        await process_and_send_segment(ws, segment_pcm, segment_vads)
                        segment_pcm.clear()
                        segment_vads.clear()

                elif message.get("text") is not None:
                    txt = message["text"].strip()
                    print("[WS] text message:", txt)
                    if txt.upper() == "END":
                        print("[WS] END 수신 → 루프 종료")
                        break

            else:
                print(f"[WS] unknown message type={mtype}, raw={message}")

    finally:
        if len(segment_pcm) > 0:
            print(f"[SEGMENT] final leftover, len={len(segment_pcm)} bytes")
            try:
                await process_and_send_segment(ws, segment_pcm, segment_vads)
            except Exception as e:
                print("[ERR] final segment 처리 중 오류:", e)

        print("[WS] audio session done, 소켓 닫기 시도")
        try:
            await ws.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

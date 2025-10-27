# aihub2mediapipe.py
# ------------------------------------------------------------
# OpenPose JSON -> MediaPipe-like dict (BODY_25 + HAND_L/R_21 + FACE_70)
# - Input: one frame's OpenPose result (dict loaded from *_keypoints.json)
# - Output:
#   {
#     "pose":      [ {x,y,visibility}, ... ]  # len = 25
#     "left_hand": [ {x,y,visibility}, ... ]  # len = 21
#     "right_hand":[ {x,y,visibility}, ... ]  # len = 21
#     "face":      [ {x,y,visibility}, ... ]  # len = 70 (옵션)
#   }
# normalize=True 이면 width/height로 [0,1] 정규화.
# 누락/이상치는 x,y = NaN, visibility=0.0 으로 패딩.
# target_*_count=0 으로 주면 해당 파트는 빈 리스트 반환.
# ------------------------------------------------------------
from __future__ import annotations
import math
from typing import Dict, List, Any, Optional

# 포인트 개수(기본값)
COCO_BODY_18   = 18
BODY_25        = 25
MP_HAND_21     = 21
OP_FACE_70     = 70   # OpenPose face(70). 참고: MediaPipe FaceMesh는 468이라 직접 매핑 불가.

def _to_list(x) -> list:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return []

def _take_people(jd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    OpenPose JSON에서 첫 번째 사람을 꺼냄.
    Crowd/RealSen 모두: jd["people"] 가 list 또는 dict 일 수 있음.
    """
    ppl = jd.get("people", [])
    if isinstance(ppl, dict):
        return ppl
    if isinstance(ppl, list) and len(ppl) > 0:
        return ppl[0]
    return None

def _parse_xyc(flat: List[float], expected_count: int) -> List[Dict[str, float]]:
    """
    flat: [x0,y0,c0, x1,y1,c1, ...]
    expected_count 길이만큼 (x,y,visibility) 채워 dict 리스트로 반환.
    부족분은 NaN/0.0 패딩.
    """
    pts: List[Dict[str, float]] = []
    n = len(flat) // 3
    for i in range(expected_count):
        if i < n:
            x = float(flat[3*i + 0]); y = float(flat[3*i + 1]); c = float(flat[3*i + 2])
        else:
            x = float("nan"); y = float("nan"); c = 0.0
        pts.append({"x": x, "y": y, "visibility": c})
    return pts

def _guess_body_count(flat: List[float]) -> int:
    """pose_keypoints_2d 길이로 COCO(18) vs BODY_25 추정."""
    n = len(flat) // 3
    if n >= BODY_25:
        return BODY_25
    elif n >= COCO_BODY_18:
        return COCO_BODY_18
    return n

def _normalize_pts_inplace(pts: List[Dict[str, float]], width: int, height: int):
    """픽셀 좌표를 [0,1] 정규화. NaN은 그대로 둠."""
    w = max(1, int(width)); h = max(1, int(height))
    for p in pts:
        x = p.get("x"); y = p.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
            p["x"] = float(x) / w
            p["y"] = float(y) / h

def _pad_or_trim(pts: List[Dict[str, float]], target_len: int) -> List[Dict[str, float]]:
    """목표 길이에 맞춰 패딩/트림."""
    if target_len <= 0:
        return []
    if len(pts) == target_len:
        return pts
    if len(pts) > target_len:
        return pts[:target_len]
    # pad
    return pts + ([{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * (target_len - len(pts)))

def convert_frame_openpose_to_mp(
    jd: Dict[str, Any],
    width: int,
    height: int,
    normalize: bool = False,
    target_body_count: int = BODY_25,
    target_hand_count: int = MP_HAND_21,
    target_face_count: int = OP_FACE_70,
) -> Dict[str, List[Dict[str, float]]]:
    """
    OpenPose 1프레임 JSON → MediaPipe 스타일 딕셔너리로 변환.

    Args:
      - jd: OpenPose frame dict (json.load 결과)
      - width, height: 원본 해상도(정규화에 사용)
      - normalize: True면 [0,1] 정규화
      - target_body_count: 25 권장(BODY_25)
      - target_hand_count: 21 권장
      - target_face_count: 70 권장(0이면 face 비활성화)

    Returns:
      {
        "pose": [... len=target_body_count ...],
        "left_hand": [... len=target_hand_count ...],
        "right_hand":[... len=target_hand_count ...],
        "face": [... len=target_face_count ...]  # target_face_count==0이면 []
      }
    """
    person = _take_people(jd)
    if person is None:
        # 전부 NaN 패딩
        def _blank(n): return [{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * n
        return {
            "pose": _blank(max(0, target_body_count)),
            "left_hand": _blank(max(0, target_hand_count)),
            "right_hand": _blank(max(0, target_hand_count)),
            "face": _blank(max(0, target_face_count)),
        }

    # 1) BODY
    pose_flat = _to_list(person.get("pose_keypoints_2d", []))
    detected_body = _guess_body_count(pose_flat) if pose_flat else target_body_count
    pose_pts = _parse_xyc(pose_flat, detected_body)
    pose_pts = _pad_or_trim(pose_pts, max(0, target_body_count))

    # 2) HANDS
    lh_flat = _to_list(person.get("hand_left_keypoints_2d", []))
    rh_flat = _to_list(person.get("hand_right_keypoints_2d", []))
    left_pts  = _parse_xyc(lh_flat,  target_hand_count) if target_hand_count > 0 else []
    right_pts = _parse_xyc(rh_flat, target_hand_count) if target_hand_count > 0 else []
    if target_hand_count > 0:
        left_pts  = _pad_or_trim(left_pts,  target_hand_count)
        right_pts = _pad_or_trim(right_pts, target_hand_count)

    # 3) FACE (옵션)
    face_pts: List[Dict[str, float]] = []
    if target_face_count > 0:
        # OpenPose face 키 이름은 보통 'face_keypoints_2d'
        face_flat = _to_list(person.get("face_keypoints_2d", []))
        # (혹시 다른 키로 저장된 경우 대비)
        if not face_flat:
            # 일부 변종 키 이름 예비 처리
            face_flat = _to_list(person.get("face_keypoints", [])) or _to_list(person.get("face_keypoints2d", []))
        face_pts = _parse_xyc(face_flat, target_face_count)
        face_pts = _pad_or_trim(face_pts, target_face_count)

    # 4) normalize
    if normalize:
        _normalize_pts_inplace(pose_pts, width, height)
        if target_hand_count > 0:
            _normalize_pts_inplace(left_pts, width, height)
            _normalize_pts_inplace(right_pts, width, height)
        if target_face_count > 0:
            _normalize_pts_inplace(face_pts, width, height)

    return {
        "pose": pose_pts,
        "left_hand": left_pts,
        "right_hand": right_pts,
        "face": face_pts,
    }

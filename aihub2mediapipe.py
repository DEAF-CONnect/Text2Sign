# aihub2mediapipe.py
# ------------------------------------------------------------
# OpenPose JSON -> MediaPipe-like dict
# - Input: one frame's OpenPose result (python dict loaded from *_keypoints.json)
# - Output: {
#     "pose":      [{"x":..,"y":..,"visibility":..}, ...],  # len = 25 (BODY_25) padded if needed
#     "left_hand": [{"x":..,"y":..,"visibility":..}, ...],  # len = 21
#     "right_hand":[{"x":..,"y":..,"visibility":..}, ...],  # len = 21
#   }
# normalize=True이면 width/height로 [0,1] 정규화.
# 누락/이상치는 x,y = NaN, visibility=0.0 로 패딩(이후 파이프라인에서 mask=0으로 처리).
# ------------------------------------------------------------
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# OpenPose 포맷별 포인트 수
COCO_BODY_18 = 18
BODY_25 = 25
MP_HAND_21 = 21

def _to_list(x) -> list:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        return []

def _take_people(jd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    OpenPose JSON에서 첫 번째 사람을 꺼내 반환.
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
    flat: [x0,y0,c0,x1,y1,c1,...]
    expected_count 길이만큼 (x,y,visibility) 채워 dict 리스트로 반환.
    부족하면 NaN/0.0으로 패딩.
    """
    pts: List[Dict[str, float]] = []
    n_triplets = len(flat) // 3
    for i in range(expected_count):
        if i < n_triplets:
            x = float(flat[3*i + 0])
            y = float(flat[3*i + 1])
            c = float(flat[3*i + 2])
        else:
            x = float("nan")
            y = float("nan")
            c = 0.0
        pts.append({"x": x, "y": y, "visibility": c})
    return pts

def _guess_body_count(flat: List[float]) -> int:
    """pose_keypoints_2d 길이를 보고 COCO(18) vs BODY_25(25) 추정."""
    n_triplets = len(flat) // 3
    if n_triplets >= BODY_25:
        return BODY_25
    elif n_triplets >= COCO_BODY_18:
        return COCO_BODY_18
    return n_triplets  # 예외적으로 더 작을 수도 있음

def _normalize_pts_inplace(pts: List[Dict[str, float]], width: int, height: int):
    """픽셀 좌표를 [0,1]로 정규화. NaN은 그대로 둠."""
    w = max(1, int(width))
    h = max(1, int(height))
    for p in pts:
        x = p.get("x", float("nan"))
        y = p.get("y", float("nan"))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
            p["x"] = float(x) / w
            p["y"] = float(y) / h
        # visibility는 변경하지 않음

def _pad_or_trim(pts: List[Dict[str, float]], target_len: int) -> List[Dict[str, float]]:
    """목표 길이에 맞춰 패딩/트림."""
    if len(pts) == target_len:
        return pts
    if len(pts) > target_len:
        return pts[:target_len]
    # pad
    padded = pts + ([{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * (target_len - len(pts)))
    return padded

def convert_frame_openpose_to_mp(
    jd: Dict[str, Any],
    width: int,
    height: int,
    normalize: bool = False,
    target_body_count: int = BODY_25,
    target_hand_count: int = MP_HAND_21
) -> Dict[str, List[Dict[str, float]]]:
    """
    OpenPose 1프레임 JSON → MediaPipe 스타일 딕셔너리로 변환.

    Args
    - jd: OpenPose frame dict (already loaded via json.load)
    - width, height: 원본 프레임 해상도(정규화할 때 사용)
    - normalize: True면 (x/w, y/h)로 [0,1] 정규화
    - target_body_count: 반환할 상체 포인트 개수(기본 25)
    - target_hand_count: 반환할 손 포인트 개수(기본 21)

    Returns
    {
      "pose":      [ {x,y,visibility}, ... ]  # len = target_body_count
      "left_hand": [ {...} x target_hand_count ],
      "right_hand":[ {...} x target_hand_count ],
    }
    """
    person = _take_people(jd)
    if person is None:
        # 미검출: 전부 NaN/0.0
        return {
            "pose":      [{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * target_body_count,
            "left_hand": [{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * target_hand_count,
            "right_hand":[{"x": float("nan"), "y": float("nan"), "visibility": 0.0}] * target_hand_count,
        }

    # 1) BODY
    pose_flat = _to_list(person.get("pose_keypoints_2d", []))
    detected_body = _guess_body_count(pose_flat) if pose_flat else target_body_count
    pose_pts = _parse_xyc(pose_flat, detected_body)
    # 목표 길이에 맞춰 패딩/트림
    pose_pts = _pad_or_trim(pose_pts, target_body_count)

    # 2) HANDS
    lh_flat = _to_list(person.get("hand_left_keypoints_2d", []))
    rh_flat = _to_list(person.get("hand_right_keypoints_2d", []))
    left_pts  = _parse_xyc(lh_flat, target_hand_count) if lh_flat else _parse_xyc([], target_hand_count)
    right_pts = _parse_xyc(rh_flat, target_hand_count) if rh_flat else _parse_xyc([], target_hand_count)

    # 3) 정규화 (옵션)
    if normalize:
        _normalize_pts_inplace(pose_pts, width, height)
        _normalize_pts_inplace(left_pts, width, height)
        _normalize_pts_inplace(right_pts, width, height)

    return {
        "pose": pose_pts,
        "left_hand": left_pts,
        "right_hand": right_pts,
    }

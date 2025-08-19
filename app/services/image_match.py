from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple
from app.core.settings import IMAGES_DIR, MIN_MATCH_COUNT, GOOD_MATCH_RATIO

def _load_gray(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _detect_and_compute(gray):
    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des

def compare_images(query_path: Path, candidate_path: Path) -> Tuple[int, float]:
    q = _load_gray(query_path)
    c = _load_gray(candidate_path)
    if q is None or c is None:
        return (0, 0.0)

    qkp, qdes = _detect_and_compute(q)
    ckp, cdes = _detect_and_compute(c)
    if qdes is None or cdes is None:
        return (0, 0.0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(qdes, cdes, k=2)

    good = [m for m, n in matches if m.distance < GOOD_MATCH_RATIO * n.distance]

    return (len(good), len(good)/max(1, len(matches)))

def find_best_match(query_path: Path) -> Optional[Path]:
    best = (0, 0.0, None)
    for p in IMAGES_DIR.glob("*.*"):
        if p.is_file():
            count, ratio = compare_images(query_path, p)
            if count > best[0]:
                best = (count, ratio, p)

    if best[0] >= MIN_MATCH_COUNT:
        return best[2]
    return None

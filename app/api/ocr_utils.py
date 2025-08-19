from typing import Dict, List, Optional, Tuple
import re
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
from app.core.settings import TESSERACT_EXE
import os
from pathlib import Path

# point to tesseract.exe if needed
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# Languages to try independently to reduce cross-language interference
LANGS_TO_TRY: List[str] = ["ara+eng", "ara", "eng"]

# --------- regex patterns (line-level, multi-word safe) ----------
AR_PAT = {
    "full_name":   re.compile(r"الاسم\s*[:：]?\s*(.+)", re.IGNORECASE),
    "mother_name": re.compile(r"(?:اسم\s*الأم|اسم\s*الام)\s*[:：]?\s*(.+)", re.IGNORECASE),
    "dob":         re.compile(r"(?:تاريخ\s*الولادة|تاريخ\s*الميلاد)\s*[:：]?\s*(\d{2}[\/\-.]\d{2}[\/\-.]\d{4}|\d{4}[\/\-.]\d{2}[\/\-.]\d{2})", re.IGNORECASE),
    "birth_place": re.compile(r"(?:مكان\s*الولادة)\s*[:：]?\s*(.+)", re.IGNORECASE),
}

EN_PAT = {
    "full_name":   re.compile(r"(?:Full\s*Name|Name)\s*[:：]?\s*(.+)", re.IGNORECASE),
    "mother_name": re.compile(r"(?:Mother(?:'s)?\s*Name|Mother)\s*[:：]?\s*(.+)", re.IGNORECASE),
    "dob":         re.compile(r"(?:Date\s*of\s*Birth|DOB)\s*[:：]?\s*(\d{2}[\/\-.]\d{2}[\/\-.]\d{4}|\d{4}[\/\-.]\d{2}[\/\-.]\d{2})", re.IGNORECASE),
    "birth_place": re.compile(r"(?:Place\s*of\s*Birth)\s*[:：]?\s*(.+)", re.IGNORECASE),
}

DATE_ANY = re.compile(r"(\d{2}[\/\-.]\d{2}[\/\-.]\d{4}|\d{4}[\/\-.]\d{2}[\/\-.]\d{2})")

def _preprocess_variants(pil_img: Image.Image) -> List[Image.Image]:
    """Generate rotated + cleaned variants."""
    base = ImageOps.exif_transpose(pil_img.convert("RGB"))
    outs: List[Image.Image] = []
    for rot in (0, 90, 180, 270):
        img = base.rotate(rot, expand=True) if rot else base

        # plain
        outs.append(img)

        # gray + unsharp
        gray = img.convert("L").filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=3)).convert("RGB")
        outs.append(gray)

        # upscale + median
        w, h = img.size
        up = img.resize((int(w*1.6), int(h*1.6)), Image.Resampling.LANCZOS)
        up = up.filter(ImageFilter.MedianFilter(size=3))
        outs.append(up)

        # adaptive threshold (remove background texture)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        g = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
        thr = cv2.medianBlur(thr, 3)
        outs.append(Image.fromarray(cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)))

        # CLAHE contrast boost
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(g)
            outs.append(Image.fromarray(cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)))
        except Exception:
            pass

        # Bilateral filtering to preserve edges while denoising
        try:
            bil = cv2.bilateralFilter(cv_img, d=9, sigmaColor=75, sigmaSpace=75)
            outs.append(Image.fromarray(cv2.cvtColor(bil, cv2.COLOR_BGR2RGB)))
        except Exception:
            pass
    return outs

def _ocr_text(img: Image.Image, lang: str, psm: int) -> str:
    cfg = f"--oem 1 --psm {psm}"
    try:
        return pytesseract.image_to_string(img, lang=lang, config=cfg)
    except Exception:
        return ""

def _line_clean(s: str) -> str:
    s = _normalize_text(s.strip())
    s = re.sub(r"\s+", " ", s)
    # drop trailing junk punctuation
    s = re.sub(r"[\|؛;:,\.\-]+$", "", s).strip()
    return s

def _normalize_text(s: str) -> str:
    """Normalize punctuation and digits (Arabic-Indic to ASCII)."""
    # unify colon-like and dash characters
    s = s.replace("：", ":").replace("–", "-").replace("—", "-").replace("−", "-")
    # Arabic comma to ASCII comma
    s = s.replace("،", ",")
    # Arabic-Indic and Persian digits -> ASCII
    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
    persian_indic = "۰۱۲۳۴۵۶۷۸۹"
    for i, d in enumerate(arabic_indic):
        s = s.replace(d, str(i))
    for i, d in enumerate(persian_indic):
        s = s.replace(d, str(i))
    return s

def _extract_by_regex(text: str) -> Dict[str, str]:
    """Return only *_ar and *_en. Never merge/convert."""
    normalized_text = _normalize_text(text)
    lines = [l for l in (t.strip() for t in normalized_text.splitlines()) if l]

    out = {
        "full_name_ar": "", "full_name_en": "",
        "mother_name_ar": "", "mother_name_en": "",
        "dob_ar": "", "dob_en": "",
        "birth_place_ar": "", "birth_place_en": "",
    }

    # Arabic scan
    for ln in lines:
        for key, pat in AR_PAT.items():
            m = pat.search(ln)
            if m and not out[f"{key}_ar"]:
                out[f"{key}_ar"] = _line_clean(m.group(1))

    # English scan
    for ln in lines:
        for key, pat in EN_PAT.items():
            m = pat.search(ln)
            if m and not out[f"{key}_en"]:
                out[f"{key}_en"] = _line_clean(m.group(1))

    # Fallbacks for DOB if label not found: first date-looking token per script
    if not out["dob_ar"]:
        m = DATE_ANY.search(normalized_text)
        if m: out["dob_ar"] = _line_clean(m.group(1))
    if not out["dob_en"]:
        m = DATE_ANY.search(normalized_text)
        if m: out["dob_en"] = _line_clean(m.group(1))

    return out

def ocr_extract_fields(image_path: str) -> Dict[str, str]:
    """Try many variants, languages and PSMs; aggregate best per field."""
    try:
        original = Image.open(image_path)
    except Exception:
        return {
            "full_name_ar": "", "full_name_en": "",
            "mother_name_ar": "", "mother_name_en": "",
            "dob_ar": "", "dob_en": "",
            "birth_place_ar": "", "birth_place_en": "",
        }

    candidates: List[Dict[str, str]] = []
    psms: List[int] = [6, 11, 4, 7, 12, 13]

    for img in _preprocess_variants(original):
        for lang in LANGS_TO_TRY:
            for psm in psms:
                txt = _ocr_text(img, lang=lang, psm=psm)
                if txt.strip():
                    candidates.append(_extract_by_regex(txt))

    return _aggregate_best(candidates)

def _aggregate_best(candidates: List[Dict[str, str]]) -> Dict[str, str]:
    keys = [
        "full_name_ar", "full_name_en",
        "mother_name_ar", "mother_name_en",
        "dob_ar", "dob_en",
        "birth_place_ar", "birth_place_en",
    ]
    result = {k: "" for k in keys}
    if not candidates:
        return result

    for key in keys:
        vals = [c.get(key, "").strip() for c in candidates if c.get(key)]
        if not vals:
            continue
        # if a value occurs most frequently, prefer it; fallback to longest
        freq: Dict[str, int] = {}
        for v in vals:
            vv = v.lower()
            freq[vv] = freq.get(vv, 0) + 1
        best_by_freq = max(vals, key=lambda v: (freq[v.lower()], len(v)))
        result[key] = _standardize_field(key, best_by_freq)

    return result

def _standardize_field(key: str, value: str) -> str:
    if not value:
        return value
    if key in ("dob_ar", "dob_en"):
        v = _normalize_text(value)
        # unify separators
        v = re.sub(r"[\-.]", "/", v)
        return v
    return value

# app/api/ocr_utils.py

# def ocr_extract_fields(image_path: str) -> dict:
#     results = arabicocr.arabic_ocr(image_path, image_path)
#     fields = {
#         "full_name_ar": "",
#         "mother_name_ar": "",
#         "dob_ar": "",
#         "birth_place_ar": "",
#         "full_name_en": "",
#         "mother_name_en": "",
#         "dob_en": "",
#         "birth_place_en": ""
#     }

#     arabic_keywords = {
#         "الاسم": "full_name_ar",
#         "الإسم": "full_name_ar",
#         "اسم الأم": "mother_name_ar",
#         "الام": "mother_name_ar",
#         "تاريخ الولادة": "dob_ar",
#         "تاريخ": "dob_ar",
#         "مكان الولادة": "birth_place_ar",
#         "مكان": "birth_place_ar"
#     }

#     last_key = None
#     for _, text, _ in results:
#         cleaned_text = text.strip().replace(":", "")

#         # If this is a keyword line
#         for keyword, field_name in arabic_keywords.items():
#             if keyword in cleaned_text:
#                 last_key = field_name
#                 break
#         else:
#             # If previous line was a keyword, assign this as its value
#             if last_key and not fields[last_key]:
#                 fields[last_key] = cleaned_text
#                 last_key = None

#     # Try extracting English fallback if needed (optional)
#     for _, text, _ in results:
#         if "Name" in text and not fields["full_name_en"]:
#             fields["full_name_en"] = text.replace("Name:", "").strip()
#         elif "Mother" in text and not fields["mother_name_en"]:
#             fields["mother_name_en"] = text.replace("Mother:", "").strip()
#         elif "Date of Birth" in text and not fields["dob_en"]:
#             fields["dob_en"] = text.replace("Date of Birth:", "").strip()
#         elif "Place of Birth" in text and not fields["birth_place_en"]:
#             fields["birth_place_en"] = text.replace("Place of Birth:", "").strip()

#     return fields





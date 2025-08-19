import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.settings import IMAGES_DIR
from app.services.storage import add_user, list_users, find_user_by_image_path
from app.services.image_match import find_best_match
from app.api.ocr_utils import ocr_extract_fields

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, id_image: UploadFile = File(...)):
    temp_path = IMAGES_DIR / f"temp_{uuid.uuid4().hex}.jpg"
    content = await id_image.read()
    temp_path.write_bytes(content)

    match_path: Optional[Path] = find_best_match(temp_path)

    if not match_path:
        temp_path.unlink(missing_ok=True)
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "No match found."},
        )

    # Extract fresh OCR fields (now includes *_ar / *_en + main)
    fields: Dict[str, str] = ocr_extract_fields(str(temp_path))
    user = find_user_by_image_path(str(match_path))
    temp_path.unlink(missing_ok=True)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "success": "Login successful!",
            "user_fields": fields,
            "user_record": user,
        },
    )

@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@router.post("/signup", response_class=HTMLResponse)
async def signup_submit(
    request: Request,
    id_image: Optional[UploadFile] = File(None),
    confirm: Optional[str] = Form(None),

    # language-specific fields (editable or hidden from confirm step)
    full_name_ar: Optional[str] = Form(None),
    full_name_en: Optional[str] = Form(None),
    mother_name_ar: Optional[str] = Form(None),
    mother_name_en: Optional[str] = Form(None),
    dob_ar: Optional[str] = Form(None),
    dob_en: Optional[str] = Form(None),
    birth_place_ar: Optional[str] = Form(None),
    birth_place_en: Optional[str] = Form(None),
):
    if confirm is None:
        if id_image is None:
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error": "Please upload an ID image."},
                status_code=400,
            )

        # sanitize filename and always write to our controlled directory
        temp_path = IMAGES_DIR / f"temp_{uuid.uuid4().hex}.jpg"
        temp_path.write_bytes(await id_image.read())

        fields = ocr_extract_fields(str(temp_path))  # returns *_ar/*_en only now
        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "stage": "confirm",
                "temp_image": str(temp_path.name),
                "fields": fields,
            },
        )

    # Save step
    temp_image_name = request.query_params.get("tmp")
    # prevent path traversal by resolving only against IMAGES_DIR and using name
    temp_path = IMAGES_DIR / Path(temp_image_name).name if temp_image_name else None

    if temp_path and temp_path.exists():
        final_path = IMAGES_DIR / f"{uuid.uuid4().hex}.jpg"
        final_path.write_bytes(temp_path.read_bytes())
        temp_path.unlink(missing_ok=True)
    else:
        if id_image is None:
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "error": "Upload lost. Please re-upload the ID image."},
                status_code=400,
            )
        final_path = IMAGES_DIR / f"{uuid.uuid4().hex}.jpg"
        final_path.write_bytes(await id_image.read())

    record = {
        "id": uuid.uuid4().hex,
        "image_path": str(final_path),

        # store ONLY language-specific variants (no merged fields)
        "full_name_ar": (full_name_ar or "").strip(),
        "full_name_en": (full_name_en or "").strip(),
        "mother_name_ar": (mother_name_ar or "").strip(),
        "mother_name_en": (mother_name_en or "").strip(),
        "dob_ar": (dob_ar or "").strip(),
        "dob_en": (dob_en or "").strip(),
        "birth_place_ar": (birth_place_ar or "").strip(),
        "birth_place_en": (birth_place_en or "").strip(),

        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    add_user(record)

    return templates.TemplateResponse(
        "signup.html",
        {"request": request, "success": "Signup successful! You can now log in.", "saved": record},
    )


@router.get("/users", response_class=HTMLResponse)
def users_page(request: Request):
    return templates.TemplateResponse(
        "signup.html",
        {"request": request, "users": list_users()},
    )

import json
from typing import Dict, List, Optional
from app.core.settings import USERS_JSON

def _read_all() -> List[Dict]:
    try:
        with open(USERS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_all(users: List[Dict]) -> None:
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def add_user(user: Dict) -> Dict:
    users = _read_all()
    users.append(user)
    _write_all(users)
    return user

def list_users() -> List[Dict]:
    return _read_all()

def find_user_by_image_path(image_path: str) -> Optional[Dict]:
    for u in _read_all():
        if u.get("image_path") == image_path:
            return u
    return None

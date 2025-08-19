from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.routes_auth import router as auth_router

app = FastAPI(title="ID OCR Auth (Prototype)")

# serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# include routes
app.include_router(auth_router)

@app.get("/health")
def health():
    return {"status": "ok"}

import secrets

from fastapi import FastAPI, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from . import analysis, config
from .auth import check_credentials, current_user

app = FastAPI(title="AI Text Detector")
app.add_middleware(
    SessionMiddleware,
    secret_key=config.SECRET_KEY or secrets.token_hex(32),
    same_site="strict",
    max_age=60 * 60 * 12,
)
app.mount("/static", StaticFiles(directory=config.ROOT / "app" / "static"), name="static")
templates = Jinja2Templates(directory=config.ROOT / "app" / "templates")


def page(request: Request, name: str, **ctx):
    return templates.TemplateResponse(request, name, {"user": current_user(request), **ctx})


@app.get("/login")
def login_form(request: Request):
    return page(request, "login.html", error=None)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if not check_credentials(username, password):
        return templates.TemplateResponse(
            request, "login.html", {"user": None, "error": "Wrong username or password."}, status_code=401
        )
    request.session["user"] = username
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def index(request: Request):
    if not current_user(request):
        return RedirectResponse("/login", status_code=303)
    return page(request, "index.html")


@app.get("/about")
def about(request: Request):
    if not current_user(request):
        return RedirectResponse("/login", status_code=303)
    return page(request, "about.html")


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/api/analyze")
async def analyze(request: Request, body: AnalyzeRequest):
    if not current_user(request):
        return JSONResponse({"detail": "Not logged in."}, status_code=401)
    if len(body.text) > 50_000:
        return JSONResponse({"detail": "Text is too long (max 50,000 characters)."}, status_code=400)
    try:
        return await run_in_threadpool(analysis.analyze, body.text)
    except analysis.TextTooShort as e:
        return JSONResponse({"detail": str(e)}, status_code=400)

import base64
import json
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI


APP_DIR = Path(__file__).parent
MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

DEFAULT_PROMPT = """你是一位谨慎的遥感图像分析专家。请判断用户上传的图片中是否存在疑似露天垃圾堆放或非法倾倒区域。
重点观察：道路或道路尽头附近的不规则堆积物；远离正常居民区、学校、医院等区域的异常堆放；与施工、料场、自然裸地明显不同的纹理和分布。
仅依据图像中可见信息判断，不要虚构地点或背景。单张图片不能构成执法结论；信息不足时应降低置信度并说明局限。"""

app = FastAPI(title="遥感图像垃圾倾倒研判", version="1.0.0")


def validate_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Base URL 必须是有效的 http(s) 地址")
    return value


def parse_result(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="模型未返回有效 JSON，请换用支持结构化输出的模型") from exc

    required = {"is_dumping", "confidence", "analysis", "limitations"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise HTTPException(status_code=502, detail="模型返回内容缺少必要字段")
    result["is_dumping"] = bool(result["is_dumping"])
    try:
        result["confidence"] = max(0, min(100, int(result["confidence"])))
    except (TypeError, ValueError):
        result["confidence"] = 0
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return (APP_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    base_url: str = Form(...),
    model: str = Form(...),
    prompt: str = Form(...),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPG、PNG 或 WebP 图片")
    image_bytes = await file.read(MAX_IMAGE_SIZE + 1)
    if not image_bytes or len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="图片不能为空且不能超过 10MB")
    if not api_key.strip() or not model.strip():
        raise HTTPException(status_code=400, detail="API Key 和模型名称不能为空")
    if not prompt.strip() or len(prompt) > 4000:
        raise HTTPException(status_code=400, detail="分析提示词不能为空且不能超过 4000 个字符")

    client = AsyncOpenAI(api_key=api_key.strip(), base_url=validate_base_url(base_url), timeout=90.0)
    data_url = f"data:{file.content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    prompt = """分析这张图片，并只返回 JSON：
{"is_dumping": boolean, "confidence": 0到100的整数, "analysis": "可见依据和判断过程", "limitations": "局限与人工复核建议"}"""
    try:
        completion = await client.chat.completions.create(
            model=model.strip(),
            messages=[
                {"role": "system", "content": prompt.strip()},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
    except Exception as exc:
        message = str(exc)
        if len(message) > 300:
            message = message[:300] + "..."
        raise HTTPException(status_code=502, detail=f"模型请求失败：{message}") from exc

    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=502, detail="模型返回了空内容")
    return parse_result(content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8111)

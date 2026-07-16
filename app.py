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

DEFAULT_PROMPT = """
你是一位专业的卫星图像分析专家，专门识别垃圾倾倒后的地点。
垃圾倾倒后的地方的一些典型特征: 
1、垃圾堆位于公路附近或者道路的尽头(车辆开过去倾倒，倾倒后车辆离开)。
2、不和居住区/医院/工厂/学校等建筑在一起，因为如果有附近居民会举报
3、偷偷进行，倾倒位置不会出现新的类似道路施工/建筑施工的痕迹
只从上面3点来判断是否为垃圾倾倒地点。不要考虑其他因素。
"""

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

    required = {"is_dumping", "analysis_logic"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise HTTPException(status_code=502, detail="模型返回内容缺少必要字段")
    return {
        "is_dumping": bool(result["is_dumping"]),
        "analysis": str(result["analysis_logic"]),
        "confidence": None,
        "limitations": "模型结果仅用于辅助筛查，请结合原始影像、时相变化和现场核查复核。",
    }


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

    normalized_base_url = validate_base_url(base_url)
    client = AsyncOpenAI(api_key=api_key.strip(), base_url=normalized_base_url, timeout=90.0)
    data_url = f"data:{file.content_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    request_kwargs = {
        "model": model.strip(),
        "messages": [
            {"role": "system", "content": prompt.strip()},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "分析图片中是否可能为垃圾倾倒地点？"},
                ],
            },
        ],
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "garbage_dumping_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_dumping": {
                            "type": "boolean",
                            "description": "是否为垃圾倾倒地点",
                        },
                        "analysis_logic": {
                            "type": "string",
                            "description": "分析逻辑和判断依据",
                        },
                    },
                    "required": ["is_dumping", "analysis_logic"],
                    "additionalProperties": False,
                },
            },
        },
    }
    if "dashscope.aliyuncs.com" in normalized_base_url:
        request_kwargs["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": 1000,
        }
    try:
        completion = await client.chat.completions.create(**request_kwargs)
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

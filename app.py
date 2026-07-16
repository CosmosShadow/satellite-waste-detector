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
你是一位专业的卫星图像分析专家，需要判断图像中是否存在垃圾倾倒地点。

请围绕最可疑的候选区域进行判断，不要因为图像其他位置存在一栋建筑就直接排除，也不要因为有道路、道路尽头或裸地就直接判定为垃圾。

判断步骤：
1. 检查候选区域是否有道路或土路可达，是否位于道路附近或尽头。
2. 检查候选区域周边是否存在成片居住区、医院、学校、正在使用的工厂等容易被发现的建筑。零散且距离候选区域较远的建筑不能单独作为排除依据。
3. 检查候选区域是否存在区别于周围环境的不规则扰动，例如边界不规则、颜色或纹理杂乱、零散堆积、局部异常裸露。遥感分辨率不足时，不要求看清单个垃圾物体。
4. 排除正常农田、自然裸地、海岸岩石、规则平整的土方，以及具有规则边界、连续开挖、机械车辙或明显工程布局的施工区域。

最终规则：
- 道路可达 + 周边没有成片敏感建筑 + 存在不规则异常扰动，倾向判定为垃圾倾倒。
- 只有道路或普通裸地、没有异常扰动，判定为非垃圾。
- 有明确正常施工、农田、自然地貌或成片建筑，判定为非垃圾。
只依据图像可见信息作出判断。
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

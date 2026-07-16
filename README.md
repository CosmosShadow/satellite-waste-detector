# 遥感图像垃圾倾倒研判

一个可本地运行的开源 Web 工具：用户上传遥感或航拍图片，并使用自己的多模态模型 API 对疑似露天垃圾堆放、非法倾倒区域进行辅助筛查。

## 功能

- 本地上传 JPG、PNG、WebP 图片，支持预览和拖拽
- 预设阿里云百炼、OpenAI，也可填写任意 OpenAI-compatible Base URL
- API Key 由用户在页面填写，仅随本次分析请求发送，不保存到磁盘
- 分析提示词直接在页面编辑，刷新后恢复项目默认值
- 展示风险结论、模型置信度、判断依据和人工复核建议
- 不包含地图抓取、Google Maps、坐标解析或截图功能

## 快速开始

需要 Python 3.9 或更高版本。

```bash
git clone <你的仓库地址>
cd satellite-waste-detector
# 推荐使用隔离环境；已有依赖时可跳过下面两行
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器打开 <http://127.0.0.1:8111>，选择服务商，填写 API Key，上传图片后点击“开始研判”。Windows 激活虚拟环境可使用 `.venv\Scripts\activate`。

`.venv` 只是本地依赖隔离目录，不是项目运行的硬性要求，也不会被 Git 提交。若本机 Python 已安装所需依赖，可直接执行 `python3 app.py`。

## 示例数据

仓库中的 `images/全部数据/` 包含整理后的垃圾识别图片集（共 24 张，约 106MB）：

- `images/全部数据/垃圾/`：12 张垃圾倾倒正样本
- `images/全部数据/非垃圾/`：12 张非垃圾负样本

这些图片可用于界面试用、提示词评估和模型效果对照。图片可能来源于地图截图或业务采集，公开发布前请逐项确认版权、隐私和再分发授权；项目的 MIT 软件协议不会自动覆盖第三方图片素材。

## 模型配置

| 服务商 | Base URL | 默认模型 |
| --- | --- | --- |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3.6-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4.1-mini` |
| 自定义 | 用户填写 | 用户填写 |

自定义接口需要兼容 OpenAI Chat Completions、多模态 `image_url` 输入及 JSON Object 输出。不同服务的模型名称和能力可能变化，请以服务商文档为准。

原项目使用的是 `qwen3-vl-plus`。本项目默认选择更新的 `qwen3.6-plus`，模型输入框仍可自由修改，便于对比其他视觉模型。

## 使用效果

完成分析后，页面会给出：

- 是否发现疑似垃圾倾倒风险
- 0-100 的模型置信度
- 基于图像可见内容的判断依据
- 图像和模型的局限，以及现场复核建议

模型输出仅用于辅助筛查。遥感分辨率、拍摄时间、遮挡、施工料场和自然裸地等因素都可能造成误判，结果不能替代现场核查或作为执法结论。

## 安全说明

- 项目不需要 `.env`，也不要把真实 API Key 写入代码或提交到 Git。
- 页面输入的 API Key 会经过本机 FastAPI 服务转发给所选模型服务，不会由本项目保存。
- 建议只在可信电脑上运行，并使用有额度限制、可随时撤销的 API Key。
- 公开部署前应增加身份认证、请求限流，并限制可访问的 Base URL，避免成为开放代理。

## 开源协议

[MIT](LICENSE)

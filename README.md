# Qwen API Proxy

将 Qwen API 转换为 OpenAI / Anthropic 标准格式的代理服务。

## 快速开始

### 本地运行

```bash
# 确保已安装 Deno
# Windows (PowerShell): irm https://deno.land/install.ps1 | iex
# macOS/Linux: curl -fsSL https://deno.land/install.sh | sh

# 运行服务
deno run --allow-net --allow-env --allow-read main.ts

# 或使用任务
deno task start
```

### 测试服务

```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试对话（需要在 .env 中配置 QWEN_TOKEN）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-plus","messages":[{"role":"user","content":"你好"}]}'

# 测试 Anthropic 格式
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_QWEN_TOKEN" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":100,"messages":[{"role":"user","content":"你好"}]}'
```

## 支持的模型

- `qwen3.6-plus` - 标准对话
- `qwen3.6-plus-search` - 搜索增强
- `qwen3.6-plus-thinking` - 思维链模式
- `qwen3.6-plus-image` - 图片生成
- `qwen3.6-plus-video` - 视频生成
- `qwen3.6-plus-research` - 深度研究模式

## API 端点

### OpenAI 格式
- `POST /v1/chat/completions` — 标准 OpenAI Chat Completions API
- `GET /v1/models` — 模型列表

### Anthropic 格式
- `POST /v1/messages` — Anthropic Messages API（兼容 Claude Code 等工具）

Anthropic 格式使用 `x-api-key` 头传递 token，支持完整的 tool use、流式响应等功能。

## 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `QWEN_TOKEN` | ✅ | Qwen 的 Bearer Token |
| `USE_DENO_ENV` | ⚪ | 使用环境变量中的 token（true/false） |
| `DEBUG` | ⚪ | 开启调试日志（true/false） |
| `QWEN_SESSION_TEMP` | ⚪ | 使用临时会话（true/false） |

## Deno Deploy 部署

1. 将代码推送到 GitHub 公开仓库
2. 访问 https://deno.com/deploy
3. 创建新项目，连接 GitHub 仓库
4. 设置入口文件为 `main.ts`
5. 配置环境变量

## 注意事项

- Token 有效期约 30 天，过期需重新获取
- 不要将 `.env` 文件提交到公开仓库
- Deno Deploy 免费版仅支持公开仓库

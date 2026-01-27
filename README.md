# Databricks Claude Load Balancer

一个用于 Claude Code 的智能负载均衡代理，将 Claude API 请求分发到多个 Databricks workspace 端点。

## 为什么需要这个项目？

- **突破单一 workspace 限制**：通过多个 Databricks workspace 分散请求，提高整体吞吐量
- **高可用性**：内置熔断器机制，自动检测故障端点并切换到健康端点
- **成本优化**：利用多个 workspace 的配额，避免单点瓶颈

## 功能特性

- **负载均衡** - 支持最少请求数 (least_requests)、轮询 (round_robin)、随机 (random) 策略
- **多端点支持** - 可配置多个 Databricks workspace 端点
- **熔断器** - 自动检测故障端点并临时禁用，超时后自动恢复
- **API Key 认证** - 支持自定义 API Key 验证
- **流式响应** - 完整支持 SSE 流式输出
- **Extended Thinking** - 支持 Claude Opus 的思考模式
- **自动重试** - 请求失败时自动切换端点重试

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/databricks-claude-lb.git
cd databricks-claude-lb
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置端点

复制示例配置文件：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，填入你的 Databricks workspace 信息：

```yaml
load_balancer:
  strategy: least_requests        # 负载均衡策略
  circuit_breaker_threshold: 5    # 熔断器错误阈值
  circuit_breaker_timeout: 60     # 熔断器恢复超时（秒）

auth:
  api_key: your-secret-api-key    # 自定义的 API Key

endpoints:
  - name: workspace-1
    api_base: https://adb-xxx.azuredatabricks.net/serving-endpoints
    token: dapi_xxx               # Databricks Personal Access Token
    weight: 1

  - name: workspace-2
    api_base: https://adb-yyy.azuredatabricks.net/serving-endpoints
    token: ${DATABRICKS_TOKEN_2}  # 支持环境变量
    weight: 1
```

### 4. 启动服务

```bash
python main.py
```

或使用 uvicorn 带热重载：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务将在 `http://localhost:8000` 启动。

### 5. 配置 Claude Code

设置环境变量后启动 Claude Code：

```bash
export ANTHROPIC_BASE_URL='http://localhost:8000'
export ANTHROPIC_API_KEY='your-secret-api-key'  # 与 config.yaml 中的 api_key 一致

claude
```

## Docker 部署

### 构建镜像

```bash
docker build -t claude-lb .
```

### 运行容器

```bash
docker run -d \
  --name claude-lb \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  claude-lb
```

### Docker Compose（可选）

创建 `docker-compose.yaml`：

```yaml
version: '3.8'
services:
  claude-lb:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml
    restart: unless-stopped
```

运行：

```bash
docker-compose up -d
```

## API 端点

| 端点 | 方法 | 认证 | 描述 |
|------|------|------|------|
| `/v1/messages` | POST | 需要 | 主要的消息 API（兼容 Claude API） |
| `/v1/messages/count_tokens` | POST | 不需要 | Token 计数估算 |
| `/health` | GET | 不需要 | 健康检查 |
| `/stats` | GET | 不需要 | 查看各端点统计信息 |
| `/reset` | POST | 不需要 | 重置所有熔断器状态 |

### 查看统计信息

```bash
curl http://localhost:8000/stats
```

返回示例：

```json
{
  "endpoints": [
    {
      "name": "workspace-1",
      "active_requests": 1,
      "total_requests": 100,
      "total_errors": 2,
      "circuit_open": false
    },
    {
      "name": "workspace-2",
      "active_requests": 0,
      "total_requests": 98,
      "total_errors": 1,
      "circuit_open": false
    }
  ]
}
```

## 架构说明

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│             │     │                      │     │ Databricks          │
│ Claude Code │────►│  Load Balancer Proxy │────►│ Workspace 1         │
│             │     │  (localhost:8000)    │     │ /anthropic/v1/...   │
└─────────────┘     │                      │     └─────────────────────┘
                    │  - 负载均衡          │     ┌─────────────────────┐
                    │  - 熔断器            │────►│ Databricks          │
                    │  - 自动重试          │     │ Workspace 2         │
                    │  - 请求透传          │     └─────────────────────┘
                    └──────────────────────┘     ┌─────────────────────┐
                                           ────►│ Databricks          │
                                                 │ Workspace 3         │
                                                 └─────────────────────┘
```

代理使用 Databricks 原生的 Anthropic 端点 (`/anthropic/v1/messages`)，直接透传请求和响应，无需格式转换。

## 支持的模型

| Claude 模型 | Databricks 模型 |
|------------|-----------------|
| claude-*-sonnet-* | databricks-claude-sonnet-4-5 |
| claude-*-opus-* | databricks-claude-opus-4-5 |
| claude-*-haiku-* | databricks-claude-sonnet-4-5 (Databricks 无 haiku) |

## 负载均衡策略

| 策略 | 说明 |
|------|------|
| `least_requests` | 选择当前活跃请求数最少的端点（默认，推荐） |
| `round_robin` | 轮询方式分配请求 |
| `random` | 随机选择端点 |

## 熔断器机制

- 当某个端点连续发生 `circuit_breaker_threshold` 次服务端错误时，熔断器开启
- 熔断器开启后，该端点在 `circuit_breaker_timeout` 秒内不会收到新请求
- 超时后自动恢复，错误计数重置
- 客户端错误（4xx，除 429）不触发熔断器
- 可通过 `/reset` 端点手动重置所有熔断器

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `CONFIG_PATH` | 配置文件路径 | `config.yaml` |

配置文件中的 token 支持环境变量语法：`${ENV_VAR_NAME}`

## 前置条件

- Python 3.10+
- 至少一个启用了 Claude 模型的 Databricks workspace
- Databricks Personal Access Token

## 获取 Databricks Token

1. 登录 Databricks workspace
2. 点击右上角用户图标 → User Settings
3. 选择 Developer → Access tokens
4. 点击 Generate new token

## 常见问题

**Q: 为什么三个端点的请求数不均衡？**

A: `least_requests` 策略基于当前活跃请求数分配，而非总请求数。处理速度快的端点会接收更多请求，这是合理的负载均衡行为。

**Q: 熔断器触发后如何恢复？**

A: 等待 `circuit_breaker_timeout` 秒后自动恢复，或调用 `POST /reset` 手动重置。

**Q: 支持哪些 Claude 功能？**

A: 支持所有 Databricks Claude 端点支持的功能，包括流式输出、Extended Thinking 等。

## License

MIT

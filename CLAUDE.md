# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Databricks Claude Load Balancer - 一个用于 Claude Code 的智能负载均衡代理，将 Claude API 请求分发到多个 Databricks workspace 端点。使用 Databricks 原生 Anthropic 端点 (`/anthropic/v1/messages`) 直接透传请求。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 本地开发运行
python main.py
# 或带热重载
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker 构建和运行
docker build -t claude-lb .
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml claude-lb
```

## 架构概览

整个项目是单文件架构 (`main.py`)，包含以下核心模块：

### 模型映射 (第 25-56 行)
- `get_databricks_model()`: 将 Claude 模型名映射到 Databricks 模型
- `claude-*-sonnet-*` → `databricks-claude-sonnet-4-5`
- `claude-*-opus-*` → `databricks-claude-opus-4-5`
- `claude-*-haiku-*` → `databricks-claude-sonnet-4-5` (Databricks 无 haiku)

### 负载均衡 (第 60-143 行)
- `WorkspaceEndpoint`: 端点数据类，包含配置和运行时状态
- `LoadBalancer`: 支持三种策略
  - `least_requests`: 最少活跃请求数
  - `round_robin`: 轮询
  - `random`: 随机
- 熔断器机制: 错误达阈值自动禁用端点，超时后自动恢复

### 代理核心 (第 148-272 行)
- `ClaudeProxy`: 请求代理主类
- `proxy_request()`: 代理请求入口，最多 3 次重试
- `_stream_request()`: SSE 流式响应处理
- `_normal_request()`: 普通 JSON 响应处理
- 请求路径: `{endpoint.api_base}/anthropic/v1/messages`

### 配置管理 (第 277-308 行)
- `load_config()`: 加载 YAML 配置
- `expand_env_vars()`: 支持 `${VAR_NAME}` 环境变量语法

## API 端点

| 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|
| `/v1/messages` | POST | 需要 | 主消息 API |
| `/v1/messages/count_tokens` | POST | 不需要 | Token 估算 |
| `/health` | GET | 不需要 | 健康检查 |
| `/stats` | GET | 不需要 | 端点统计 |
| `/reset` | POST | 不需要 | 重置熔断器 |

## 配置文件

`config.yaml` 结构：
```yaml
load_balancer:
  strategy: least_requests        # 负载均衡策略
  circuit_breaker_threshold: 5    # 熔断器错误阈值
  circuit_breaker_timeout: 60     # 熔断器恢复超时（秒）

auth:
  api_key: your-key               # 客户端认证密钥

endpoints:
  - name: workspace-1
    api_base: https://adb-xxx.azuredatabricks.net/serving-endpoints
    token: dapi_xxx               # Databricks 访问令牌，支持 ${ENV_VAR}
    weight: 1
```

## 环境变量

- `CONFIG_PATH`: 配置文件路径（默认 `config.yaml`）
- `ANTHROPIC_BASE_URL`: Claude Code 需设为 `http://localhost:8000`
- `ANTHROPIC_API_KEY`: Claude Code 需设为与 `config.yaml` 中 `api_key` 一致的值

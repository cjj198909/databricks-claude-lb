"""
Databricks Claude Load Balancer Proxy for Claude Code
使用 Databricks 原生 Anthropic 端点 (/anthropic/v1/messages)
"""

import os
import re
import asyncio
import json
import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import yaml
import httpx
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 模型名称映射 ====================

DATABRICKS_MODELS = {
    "sonnet": "databricks-claude-sonnet-4-5",
    "opus": "databricks-claude-opus-4-6",  # 默认使用 4-6
    "opus-4-5": "databricks-claude-opus-4-5",
    "opus-4-6": "databricks-claude-opus-4-6",
    "haiku": "databricks-claude-haiku-4-5",
}

DEFAULT_MODEL = "databricks-claude-sonnet-4-5"


def get_databricks_model(model: str) -> str:
    """将 Claude 模型名称映射到 Databricks 模型名称"""
    model_lower = model.lower()

    # 已经是 Databricks 模型名称，直接返回
    if model_lower.startswith("databricks-"):
        return model

    # 检查是否指定了具体版本 (如 claude-opus-4-5, opus-4-5)
    if "opus" in model_lower:
        if "4-5" in model_lower or "4.5" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-5"]
        elif "4-6" in model_lower or "4.6" in model_lower:
            mapped = DATABRICKS_MODELS["opus-4-6"]
        else:
            mapped = DATABRICKS_MODELS["opus"]  # 默认 4-6
    elif "sonnet" in model_lower:
        mapped = DATABRICKS_MODELS["sonnet"]
    elif "haiku" in model_lower:
        mapped = DATABRICKS_MODELS["haiku"]
    else:
        logger.warning(f"Unknown model '{model}', using default: {DEFAULT_MODEL}")
        mapped = DEFAULT_MODEL

    if mapped != model:
        logger.info(f"Model mapping: {model} -> {mapped}")

    return mapped


# ==================== Load Balancer ====================

@dataclass
class WorkspaceEndpoint:
    name: str
    api_base: str
    token: str
    weight: int = 1
    
    active_requests: int = field(default=0, repr=False)
    total_requests: int = field(default=0, repr=False)
    total_errors: int = field(default=0, repr=False)
    last_error_time: Optional[float] = field(default=None, repr=False)
    circuit_open: bool = field(default=False, repr=False)


class LoadBalancer:
    def __init__(
        self, 
        endpoints: list[WorkspaceEndpoint],
        strategy: str = "least_requests",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60,
    ):
        self.endpoints = endpoints
        self.strategy = strategy
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
    
    def get_available_endpoints(self) -> list[WorkspaceEndpoint]:
        now = time.time()
        available = []
        
        for ep in self.endpoints:
            if ep.circuit_open:
                if ep.last_error_time and now - ep.last_error_time > self.circuit_breaker_timeout:
                    ep.circuit_open = False
                    ep.total_errors = 0
                    logger.info(f"Circuit breaker reset for {ep.name}")
                else:
                    continue
            available.append(ep)
        
        return available
    
    def select_endpoint(self) -> Optional[WorkspaceEndpoint]:
        available = self.get_available_endpoints()
        if not available:
            logger.error("No available endpoints!")
            return None
        
        if self.strategy == "least_requests":
            return min(available, key=lambda ep: ep.active_requests)
        elif self.strategy == "round_robin":
            return available[0]
        else:  # random
            return available[int(time.time() * 1000) % len(available)]
    
    async def on_request_start(self, endpoint: WorkspaceEndpoint):
        endpoint.active_requests += 1
        endpoint.total_requests += 1
    
    async def on_request_end(self, endpoint: WorkspaceEndpoint, success: bool, is_client_error: bool = False):
        endpoint.active_requests = max(0, endpoint.active_requests - 1)

        if not success and not is_client_error:
            # 只有服务端错误才计入错误数，客户端错误（4xx）不触发熔断器
            endpoint.total_errors += 1
            endpoint.last_error_time = time.time()

            if endpoint.total_errors >= self.circuit_breaker_threshold:
                endpoint.circuit_open = True
                logger.warning(f"Circuit breaker opened for {endpoint.name}")
    
    def get_stats(self) -> dict:
        return {
            "endpoints": [
                {
                    "name": ep.name,
                    "active_requests": ep.active_requests,
                    "total_requests": ep.total_requests,
                    "total_errors": ep.total_errors,
                    "circuit_open": ep.circuit_open,
                }
                for ep in self.endpoints
            ]
        }


# ==================== Claude Proxy (使用原生 Anthropic 端点) ====================

class ClaudeProxy:
    def __init__(self, load_balancer: LoadBalancer, api_key: str):
        self.load_balancer = load_balancer
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    
    async def close(self):
        await self.client.aclose()
    
    def verify_api_key(self, key: str) -> bool:
        return key == self.api_key
    
    async def proxy_request(self, body: dict, stream: bool = False):
        """代理请求到 Databricks 原生 Anthropic 端点"""
        
        # 转换模型名称
        if "model" in body:
            original_model = body["model"]
            body["model"] = get_databricks_model(original_model)
        
        # 移除 Databricks 不支持的字段（如新版 Claude Code 发送的 context_management 等）
        unsupported_fields = ["context_management"]
        for field in unsupported_fields:
            if field in body:
                logger.info(f"Removing unsupported field: {field}")
                del body[field]

        # 处理 thinking 参数兼容性
        # Opus 4.6: 支持 adaptive（推荐），enabled + budget_tokens 已废弃
        # 旧模型 (Sonnet 4.5, Opus 4.5 等): 仅支持 enabled + budget_tokens
        if "thinking" in body and isinstance(body["thinking"], dict):
            thinking_type = body["thinking"].get("type")
            model = body.get("model", "")
            is_opus_4_6 = "opus-4-6" in model

            if is_opus_4_6:
                # Opus 4.6: 使用 adaptive，移除多余的 budget_tokens
                if thinking_type == "adaptive" and "budget_tokens" in body["thinking"]:
                    del body["thinking"]["budget_tokens"]
                    logger.info("Removed budget_tokens for adaptive thinking (Opus 4.6)")
            else:
                # 旧模型: 不支持 adaptive，需转换为 enabled + budget_tokens
                if thinking_type == "adaptive":
                    body["thinking"]["type"] = "enabled"
                    max_tokens = body.get("max_tokens", 16000)
                    budget = body["thinking"].pop("budget_tokens", None) or max(1024, int(max_tokens * 0.8))
                    if max_tokens <= budget:
                        body["max_tokens"] = budget + 1
                    body["thinking"]["budget_tokens"] = budget
                    logger.info(f"Converted adaptive -> enabled with budget_tokens={budget} for {model}")
                elif thinking_type == "enabled" and "budget_tokens" not in body["thinking"]:
                    max_tokens = body.get("max_tokens", 16000)
                    budget = max(1024, int(max_tokens * 0.8))
                    if max_tokens <= budget:
                        body["max_tokens"] = budget + 1
                    body["thinking"]["budget_tokens"] = budget
                    logger.info(f"Added missing budget_tokens: {budget}")
        
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            endpoint = self.load_balancer.select_endpoint()
            if not endpoint:
                raise HTTPException(status_code=503, detail={"error": {"message": "No available endpoints"}})
            
            await self.load_balancer.on_request_start(endpoint)
            
            # 使用原生 Anthropic 端点
            url = f"{endpoint.api_base}/anthropic/v1/messages"
            
            logger.info(f"[{body.get('model')}] -> {endpoint.name} (attempt {attempt + 1})")
            
            try:
                headers = {
                    "Authorization": f"Bearer {endpoint.token}",
                    "Content-Type": "application/json",
                }
                
                if stream:
                    return await self._stream_request(endpoint, url, body, headers)
                else:
                    return await self._normal_request(endpoint, url, body, headers)
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                # 429 rate limit 也应该触发熔断，因为表示服务端过载
                is_client_error = 400 <= e.response.status_code < 500 and e.response.status_code != 429
                await self.load_balancer.on_request_end(endpoint, success=False, is_client_error=is_client_error)

                if e.response.status_code in (429, 500, 502, 503, 504):
                    logger.warning(f"{endpoint.name} returned {e.response.status_code}, retrying...")
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                else:
                    try:
                        error_body = e.response.json()
                    except:
                        error_body = {"error": {"message": e.response.text}}

                    logger.error(f"Request failed with {e.response.status_code}: {json.dumps(error_body, ensure_ascii=False)}")
                    raise HTTPException(status_code=e.response.status_code, detail=error_body)
                    
            except Exception as e:
                last_error = e
                logger.error(f"{endpoint.name} failed: {e}")
                await self.load_balancer.on_request_end(endpoint, success=False)
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
        
        raise HTTPException(status_code=503, detail={"error": {"message": f"All retries failed: {last_error}"}})
    
    async def _normal_request(self, endpoint, url, body, headers) -> JSONResponse:
        """非流式请求 - 直接透传"""
        response = await self.client.post(url, json=body, headers=headers)
        response.raise_for_status()
        await self.load_balancer.on_request_end(endpoint, success=True)
        
        return JSONResponse(content=response.json(), status_code=response.status_code)
    
    async def _stream_request(self, endpoint, url, body, headers, max_retries: int = 3) -> StreamingResponse:
        """流式请求 - 直接透传 Databricks 的 Anthropic 格式响应，支持重试"""
        
        async def stream_generator():
            current_endpoint = endpoint
            current_url = url
            current_headers = headers
            
            for attempt in range(max_retries):
                success = False
                is_client_error = False
                should_retry = False
                
                try:
                    req = self.client.build_request("POST", current_url, json=body, headers=current_headers)
                    response = await self.client.send(req, stream=True)

                    if response.status_code >= 400:
                        error_body = await response.aread()
                        # 429 rate limit 也触发熔断
                        is_client_error = 400 <= response.status_code < 500 and response.status_code != 429

                        try:
                            error_json = json.loads(error_body)
                            error_msg = error_json.get('message', 'Request failed')
                            logger.error(f"Stream request failed ({response.status_code}): {error_json}")
                        except:
                            error_msg = error_body.decode('utf-8') if isinstance(error_body, bytes) else str(error_body)
                            logger.error(f"Stream request failed ({response.status_code}): {error_msg}")

                        await self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=is_client_error)
                        
                        # 可重试的状态码
                        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                            logger.warning(f"{current_endpoint.name} returned {response.status_code}, retrying stream...")
                            await asyncio.sleep(min(2 ** attempt, 8))
                            # 选择新的 endpoint 重试
                            new_endpoint = self.load_balancer.select_endpoint()
                            if new_endpoint:
                                current_endpoint = new_endpoint
                                current_url = f"{current_endpoint.api_base}/anthropic/v1/messages"
                                current_headers = {
                                    "Authorization": f"Bearer {current_endpoint.token}",
                                    "Content-Type": "application/json",
                                }
                                await self.load_balancer.on_request_start(current_endpoint)
                                logger.info(f"[{body.get('model')}] -> {current_endpoint.name} (stream attempt {attempt + 2})")
                                continue
                        
                        yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_msg}})}\n\n".encode()
                        return

                    # 直接透传响应，不做任何转换
                    async for chunk in response.aiter_bytes():
                        yield chunk

                    success = True
                    await self.load_balancer.on_request_end(current_endpoint, success=True)
                    return
                    
                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as e:
                    # 网络超时/连接错误 - 应该触发熔断并重试
                    error_detail = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                    logger.error(f"Stream network error on {current_endpoint.name}: {error_detail}")
                    await self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=False)
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        new_endpoint = self.load_balancer.select_endpoint()
                        if new_endpoint:
                            current_endpoint = new_endpoint
                            current_url = f"{current_endpoint.api_base}/anthropic/v1/messages"
                            current_headers = {
                                "Authorization": f"Bearer {current_endpoint.token}",
                                "Content-Type": "application/json",
                            }
                            await self.load_balancer.on_request_start(current_endpoint)
                            logger.info(f"[{body.get('model')}] -> {current_endpoint.name} (stream retry {attempt + 2})")
                            continue
                    
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_detail}})}\n\n".encode()
                    return
                    
                except Exception as e:
                    import traceback
                    error_detail = f"{type(e).__name__}: {str(e) or 'Unknown error'}"
                    logger.error(f"Stream error: {error_detail}\n{traceback.format_exc()}")
                    await self.load_balancer.on_request_end(current_endpoint, success=False, is_client_error=False)
                    yield f"data: {json.dumps({'type': 'error', 'error': {'message': error_detail}})}\n\n".encode()
                    return
        
        return StreamingResponse(
            stream_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    def get_stats(self) -> dict:
        return self.load_balancer.get_stats()


# ==================== Config ====================

def expand_env_vars(value: str) -> str:
    pattern = r'\$\{([^}]+)\}'
    def replace(match):
        return os.getenv(match.group(1), "")
    return re.sub(pattern, replace, value)


def load_config(config_path: str = "config.yaml") -> ClaudeProxy:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    lb_config = config.get("load_balancer", {})
    api_key = expand_env_vars(config.get("auth", {}).get("api_key", ""))
    
    endpoints = []
    for ep in config.get("endpoints", []):
        endpoints.append(WorkspaceEndpoint(
            name=ep["name"],
            api_base=ep["api_base"],
            token=expand_env_vars(ep["token"]),
            weight=ep.get("weight", 1),
        ))
        logger.info(f"Loaded endpoint: {ep['name']}")
    
    load_balancer = LoadBalancer(
        endpoints=endpoints,
        strategy=lb_config.get("strategy", "least_requests"),
        circuit_breaker_threshold=lb_config.get("circuit_breaker_threshold", 5),
        circuit_breaker_timeout=lb_config.get("circuit_breaker_timeout", 60),
    )
    
    return ClaudeProxy(load_balancer, api_key)


# ==================== FastAPI App ====================

proxy: Optional[ClaudeProxy] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    proxy = load_config(config_path)
    logger.info(f"Proxy started with {len(proxy.load_balancer.endpoints)} endpoints (using native Anthropic endpoint)")
    yield
    if proxy:
        await proxy.close()


app = FastAPI(title="Databricks Claude Proxy (Native Anthropic)", lifespan=lifespan)


MAX_REQUEST_SIZE = 4 * 1024 * 1024  # 4MB Databricks limit


@app.post("/v1/messages")
async def messages(request: Request, x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    auth_header = request.headers.get("authorization", "")
    actual_key = x_api_key or (auth_header[7:] if auth_header.startswith("Bearer ") else "")

    if not proxy.verify_api_key(actual_key):
        raise HTTPException(status_code=401, detail={"error": {"message": "Invalid API key"}})

    # 获取原始请求体并检查大小
    body_bytes = await request.body()
    body_size = len(body_bytes)

    if body_size > MAX_REQUEST_SIZE:
        size_mb = body_size / (1024 * 1024)
        logger.warning(f"Request too large: {size_mb:.2f}MB (limit: 4MB)")
        raise HTTPException(
            status_code=413,
            detail={
                "error": {
                    "type": "request_too_large",
                    "message": f"Request size ({size_mb:.2f}MB) exceeds Databricks 4MB limit. Please use /clear to start a new conversation or remove large content from context."
                }
            }
        )

    body = json.loads(body_bytes)
    stream = body.get("stream", False)

    logger.info(f"Request: model={body.get('model')}, stream={stream}, thinking={body.get('thinking')}, size={body_size/1024:.1f}KB")

    return await proxy.proxy_request(body, stream=stream)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    body = await request.json()
    content = json.dumps(body.get("messages", []))
    estimated_tokens = len(content) // 4
    return {"input_tokens": estimated_tokens}


@app.post("/api/event_logging/batch")
async def event_logging():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/stats")
async def stats():
    return proxy.get_stats()


@app.post("/reset")
async def reset():
    for ep in proxy.load_balancer.endpoints:
        ep.circuit_open = False
        ep.total_errors = 0
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)

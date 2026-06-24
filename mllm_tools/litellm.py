import json
import re
import time
import datetime
import asyncio
import warnings
from typing import List, Dict, Any, Union, Optional
import io
import os
import base64
from PIL import Image
import mimetypes
import litellm
from litellm import completion, completion_cost, acompletion
from dotenv import load_dotenv

# Suppress Pydantic serialization warnings from litellm version mismatches
warnings.filterwarnings("ignore", message="Expected.*fields but got.*serialized value may not be as expected")

load_dotenv(os.environ.get("DOTENV_PATH") or None, override=True)  # 强制覆盖;DOTENV_PATH 可指定备用 .env(多模型并行隔离,见 generate_explanation.py)

class LiteLLMWrapper:
    """Wrapper for LiteLLM to support multiple models and logging"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-vision-preview",
        temperature: float = 0.7,
        print_cost: bool = False,
        verbose: bool = False,
        use_langfuse: bool = False,
    ):
        """
        Initialize the LiteLLM wrapper
        
        Args:
            model_name: Name of the model to use (e.g. "azure/gpt-4", "vertex_ai/gemini-pro")
            temperature: Temperature for completion
            print_cost: Whether to print the cost of the completion
            verbose: Whether to print verbose output
            use_langfuse: Whether to enable Langfuse logging
        """
        self.model_name = model_name
        self.temperature = temperature
        self.print_cost = print_cost
        self.verbose = verbose
        self.accumulated_cost = 0
        self._trace_tokens: dict = {}  # per-trace_id token accumulator
        self.custom_api_base = os.getenv("CUSTOM_API_BASE",None)
        self.custom_api_key = os.getenv("CUSTOM_API_KEY", "sk-none")
        print(f"[DEBUG] LiteLLMWrapper init in PID {os.getpid()}")
        print(f"[DEBUG] .env file location: {os.path.abspath('.env')}")
        print(f"[DEBUG] CUSTOM_API_BASE: {self.custom_api_base}")
        print(f"[DEBUG] CUSTOM_API_KEY: {self.custom_api_key[:30]}..." if self.custom_api_key else "[DEBUG] CUSTOM_API_KEY: None")
        print(f"Using custom endpoint: {self.custom_api_base}")
        
        # --- 新增代码开始：注册自定义模型 ---
        # 提取不带前缀的模型名 (例如从 "openai/Qwen3..." 提取 "Qwen3...")
        base_model_name = model_name.split("/")[-1] if "/" in model_name else model_name

        # 手动注册模型到 LiteLLM，设置成本为 0，防止 mapped error
        model_config = {
            "max_tokens": 32768,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "litellm_provider": "openai",
            "mode": "chat"
        }

        try:
            litellm.register_model({base_model_name: model_config})
            # 同时注册带日期后缀的版本（API 可能返回这个名称）
            litellm.register_model({f"{base_model_name}-20250929": model_config})
            if self.verbose:
                print(f"Registered custom model '{base_model_name}' and variant to LiteLLM.")
        except Exception as e:
            print(f"Warning: Failed to register custom model: {e}")
        # --- 新增代码结束 ---
        if self.verbose:
            os.environ['LITELLM_LOG'] = 'DEBUG'
        
        # Set langfuse callback only if enabled
        if use_langfuse:
            litellm.success_callback = ["langfuse"]
            litellm.failure_callback = ["langfuse"]

    def _encode_file(self, file_path: Union[str, Image.Image]) -> str:
        """
        Encode local file or PIL Image to base64 string
        
        Args:
            file_path: Path to local file or PIL Image object
            
        Returns:
            Base64 encoded file string
        """
        if isinstance(file_path, Image.Image):
            buffered = io.BytesIO()
            file_path.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")

    def _get_mime_type(self, file_path: str) -> str:
        """
        Get the MIME type of a file based on its extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            MIME type as a string (e.g., "image/jpeg", "audio/mp3")
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        return mime_type

    async def __call__(self, messages: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process messages and return completion (async version)

        Args:
            messages: List of message dictionaries with 'type' and 'content' keys
            metadata: Optional metadata to pass to litellm completion, e.g. for Langfuse tracking

        Returns:
            Generated text response
        """
        if metadata is None:
            print("No metadata provided, using empty metadata")
            metadata = {}
        metadata["trace_name"] = f"litellm-completion-{self.model_name}"
        # Convert messages to LiteLLM format
        formatted_messages = []
        for msg in messages:
            if msg["type"] == "text":
                formatted_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif msg["type"] in ["image", "audio", "explanation"]:
                # Check if content is a local file path or PIL Image
                if isinstance(msg["content"], Image.Image) or os.path.isfile(msg["content"]):
                    try:
                        if isinstance(msg["content"], Image.Image):
                            mime_type = "image/png"
                        else:
                            mime_type = self._get_mime_type(msg["content"])
                        base64_data = self._encode_file(msg["content"])
                        data_url = f"data:{mime_type};base64,{base64_data}"
                    except ValueError as e:
                        print(f"Error processing file {msg['content']}: {e}")
                        continue
                else:
                    data_url = msg["content"]

                # Append the formatted message based on the model
                if "gemini" in self.model_name.lower():
                    formatted_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": data_url
                            }
                        ]
                    })
                elif "gpt" in self.model_name.lower():
                    # GPT models expect a different format
                    if msg["type"] == "image":
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url,
                                        "detail": "high"
                                    }
                                }
                            ]
                        })
                    else:
                        raise ValueError("For GPT, only text and image inferencing are supported")
                else:
                    # Generic OpenAI-compatible format for other models (Kimi, Claude, etc.)
                    if msg["type"] == "image":
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                }
                            ]
                        })
                    elif msg["type"] == "explanation":
                        # Some models may support explanation URLs
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "explanation_url",
                                    "explanation_url": {
                                        "url": data_url
                                    }
                                }
                            ]
                        })
                    elif msg["type"] == "audio":
                        # Some models may support audio
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_url",
                                    "audio_url": {
                                        "url": data_url
                                    }
                                }
                            ]
                        })
                    else:
                        raise ValueError(f"Unsupported media type: {msg['type']}")

        try:
            # 准备 completion 参数
            completion_kwargs = {
                "model": self.model_name,
                "messages": formatted_messages,
                "metadata": metadata,
                # disable litellm internal silent retry; we loop manually below so
                # every individual HTTP attempt is timed and logged
                "max_retries": 0,
                # raise the per-request timeout well above litellm's 600s default:
                # long M3 reasoning scenes can run many minutes and were being cut
                # off. In principle generation should not be capped as long as the
                # model keeps producing tokens; 3600s is just a hang safety net.
                "timeout": float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "3600")),
            }

            # 如果有自定义端点，添加到参数中
            if self.custom_api_base:
                completion_kwargs["api_base"] = self.custom_api_base
                completion_kwargs["api_key"] = self.custom_api_key

            # O 系列模型特殊处理
            if (re.match(r"^o\d+.*$", self.model_name) or re.match(r"^openai/o.*$", self.model_name)):
                completion_kwargs["temperature"] = None
                completion_kwargs["reasoning_effort"] = "medium"
            else:
                completion_kwargs["temperature"] = self.temperature

            # 使用异步 API —— 流式 + 空闲超时(idle timeout)。
            # 原则：只要模型还在持续吐 token 就不打断;连续 STREAM_IDLE_TIMEOUT 秒
            # 收不到任何新 token 才判定真卡死并快速重试。这取代了"整请求总超时",
            # 既不会误杀超长但仍在产出的推理场景,又能看到实时进度(心跳写入计时日志)。
            _timing_log = os.environ.get("API_TIMING_LOG", "output/_api_timing.log")
            _trace = (metadata or {}).get("trace_id", "?")
            _idle_timeout = float(os.environ.get("STREAM_IDLE_TIMEOUT", "300"))
            _heartbeat = float(os.environ.get("STREAM_HEARTBEAT", "60"))
            _use_stream = os.environ.get("LITELLM_DISABLE_STREAM", "") == ""
            _max_attempts = 99

            def _log(line):
                try:
                    with open(_timing_log, "a") as _f:
                        _f.write(line)
                except Exception:
                    pass

            content = None
            usage_obj = None
            for _attempt in range(1, _max_attempts + 1):
                _t0 = time.time()
                stream = None
                try:
                    if _use_stream:
                        skw = dict(completion_kwargs)
                        skw["stream"] = True
                        skw["stream_options"] = {"include_usage": True}
                        stream = await acompletion(**skw)
                        parts = []
                        n_content = n_reason = 0
                        last_hb = time.time()
                        _it = stream.__aiter__()
                        while True:
                            try:
                                chunk = await asyncio.wait_for(_it.__anext__(), timeout=_idle_timeout)
                            except StopAsyncIteration:
                                break
                            cu = getattr(chunk, "usage", None)
                            if cu:
                                usage_obj = cu
                            choices = getattr(chunk, "choices", None) or []
                            if choices:
                                delta = choices[0].delta
                                piece = getattr(delta, "content", None)
                                if piece:
                                    parts.append(piece); n_content += len(piece)
                                rpiece = getattr(delta, "reasoning_content", None)
                                if rpiece:
                                    n_reason += len(rpiece)
                            now = time.time()
                            if now - last_hb >= _heartbeat:
                                _log(f"{datetime.datetime.now().isoformat()} trace={_trace} "
                                     f"model={self.model_name} attempt={_attempt} STREAM "
                                     f"elapsed={now-_t0:.0f}s content_chars={n_content} "
                                     f"reasoning_chars={n_reason}\n")
                                last_hb = now
                        content = "".join(parts)
                        _dt = time.time() - _t0
                        _ct = (getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj else 0
                        _tput = (_ct / _dt) if (_dt and _ct) else 0.0
                        _log(f"{datetime.datetime.now().isoformat()} trace={_trace} "
                             f"model={self.model_name} attempt={_attempt} OK "
                             f"elapsed={_dt:.1f}s completion_tokens={_ct} tput={_tput:.1f}tok/s "
                             f"content_chars={n_content} reasoning_chars={n_reason} stream=1\n")
                    else:
                        response = await acompletion(**completion_kwargs)
                        _dt = time.time() - _t0
                        usage_obj = getattr(response, "usage", None)
                        _ct = (getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj else 0
                        content = response.choices[0].message.content
                        _log(f"{datetime.datetime.now().isoformat()} trace={_trace} "
                             f"model={self.model_name} attempt={_attempt} OK "
                             f"elapsed={_dt:.1f}s completion_tokens={_ct} "
                             f"tput={(_ct/_dt if _dt else 0):.1f}tok/s stream=0\n")
                    break
                except Exception as _e:
                    _dt = time.time() - _t0
                    _kind = "IDLE_TIMEOUT" if isinstance(_e, asyncio.TimeoutError) else type(_e).__name__
                    _log(f"{datetime.datetime.now().isoformat()} trace={_trace} "
                         f"model={self.model_name} attempt={_attempt} ERR "
                         f"elapsed={_dt:.1f}s {_kind}: {str(_e)[:300]}\n")
                    if _attempt >= _max_attempts:
                        raise
                    await asyncio.sleep(min(2 ** _attempt, 30))
                finally:
                    if stream is not None:
                        try:
                            await stream.close()
                        except Exception:
                            try:
                                stream.close()
                            except Exception:
                                pass

            # Track token usage (per-trace_id, isolated across concurrent scenes/topics)
            if usage_obj:
                trace_id = metadata.get("trace_id") if metadata else None
                if trace_id:
                    if trace_id not in self._trace_tokens:
                        self._trace_tokens[trace_id] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    self._trace_tokens[trace_id]["input_tokens"] += getattr(usage_obj, 'prompt_tokens', 0) or 0
                    self._trace_tokens[trace_id]["output_tokens"] += getattr(usage_obj, 'completion_tokens', 0) or 0
                    self._trace_tokens[trace_id]["total_tokens"] += getattr(usage_obj, 'total_tokens', 0) or 0

            if not content:
                print(f"Got null/empty response from model (trace={_trace}).")
            return content

        except Exception as e:
            print(f"Error in model completion: {e}")
            return str(e)

    def get_token_usage(self) -> Dict[str, int]:
        """Get total accumulated token usage across all trace IDs."""
        total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for v in self._trace_tokens.values():
            for k in total:
                total[k] += v.get(k, 0)
        return total

    def reset_token_usage(self) -> None:
        """Reset all accumulated token usage."""
        self._trace_tokens.clear()
        self.accumulated_cost = 0

if __name__ == "__main__":
    import asyncio

    async def test():
        # 测试自定义模型
        wrapper = LiteLLMWrapper(
            model_name="Kimi-K25",#gemini-3-pro-preview
            verbose=True
        )

        imgae_path = "/inspire/hdd/project/ai4education/bishuzhen-CZXS24220022/edubench/TheoremExplainAgent/output/kimi2/problem_0_geometry/media/images/problem_0_geometry_scene3_v0/Scene3_ManimCE_v0.18.1.png"
        result = await wrapper(messages=[
            {"type": "text", "content": "请详细描述图片的内容，包括场景、动画效果、以及视觉元素的变化"},
            {"type": "image", "content": imgae_path}
        ])
        print(f"Response: {result}")

    asyncio.run(test())
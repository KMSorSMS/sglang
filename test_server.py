#!/usr/bin/env python3
"""SGLang Server 测试脚本"""

import argparse
import json
import requests

SERVER_URL = "http://127.0.0.1:30000"


def generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7, stream: bool = False):
    """发送生成请求到服务器"""
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        },
    }

    if stream:
        payload["stream"] = True
        response = requests.post(f"{SERVER_URL}/generate", json=payload, stream=True)
        print("Response (streaming):")
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                print(data.get("text", ""), end="", flush=True)
        print()
    else:
        response = requests.post(f"{SERVER_URL}/generate", json=payload)
        result = response.json()
        print("Response:")
        print(result.get("text", result))

    return response


def chat(messages: list, max_tokens: int = 256, temperature: float = 0.7):
    """发送聊天请求（OpenAI 兼容格式）"""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
    result = response.json()

    if "choices" in result:
        print("Response:")
        print(result["choices"][0]["message"]["content"])
    else:
        print("Result:", result)

    return response


def main():
    parser = argparse.ArgumentParser(description="测试 SGLang Server")
    parser.add_argument("--prompt", "-p", type=str, help="直接输入 prompt")
    parser.add_argument("--file", "-f", type=str, help="从文件读取 prompt")
    parser.add_argument("--max-tokens", "-m", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="温度参数")
    parser.add_argument("--stream", "-s", action="store_true", help="使用流式输出")
    parser.add_argument("--chat", "-c", action="store_true", help="使用聊天模式")
    args = parser.parse_args()

    # 获取 prompt
    if args.file:
        with open(args.file, "r") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        # 默认测试 prompt
        prompt = "Hello, how are you today?"

    print(f"Prompt: {prompt}")
    print("-" * 50)

    if args.chat:
        messages = [{"role": "user", "content": prompt}]
        chat(messages, args.max_tokens, args.temperature)
    else:
        generate(prompt, args.max_tokens, args.temperature, args.stream)


if __name__ == "__main__":
    main()

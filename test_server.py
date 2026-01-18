#!/usr/bin/env python3
"""SGLang Server 测试脚本"""

import os
import sys

# 禁用代理，避免本地请求走代理
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

# 检查是否需要使用 HuggingFace 镜像 (需要在 import datasets 之前设置)
if "--longbench" in sys.argv and "--no-mirror" not in sys.argv:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import requests
import time
from pathlib import Path

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


def generate_long_prompt(target_tokens: int, base_text: str = None) -> str:
    """
    生成指定 token 长度的 prompt

    注意: 这是粗略估计，实际 token 数取决于 tokenizer
    平均每个英文单词约 1.3 tokens，每个中文字约 1-2 tokens
    """
    if base_text is None:
        # 使用重复的文本来填充
        base_text = """The following is a long document for testing purposes.
This text will be repeated multiple times to reach the target length.
It contains various sentences and paragraphs to simulate real content.
Please read through this document and answer questions about it.
"""

    # 粗略估计: 平均每个字符约 0.25 tokens (英文)
    # 为了安全，我们用 0.3 来估计
    chars_needed = int(target_tokens / 0.3)

    # 重复文本直到达到目标长度
    repeated = base_text * (chars_needed // len(base_text) + 1)
    prompt = repeated[:chars_needed]

    # 添加问题
    prompt += "\n\nBased on the above content, please provide a brief summary."

    return prompt


def get_tokenizer(model_path: str = None):
    """加载 tokenizer"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("需要安装 transformers: pip install transformers")
        return None

    if model_path is None:
        # 默认使用 DeepSeek-V3.2 tokenizer
        model_path = "deepseek-ai/DeepSeek-V3.2"

    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer


LONGBENCH_CACHE_FILE = "longbench_v2_cache.json"


def load_longbench_v2(length_bins: list = None, use_mirror: bool = True, tokenizer=None, force_recompute: bool = False):
    """
    加载 LongBench V2 数据集，按 token 数分组

    Args:
        length_bins: 长度区间列表 (token 数)，如 [8000, 16000, 32000, 64000, 128000]
                    会按这些区间对样本进行分组
        use_mirror: 是否使用国内镜像 (hf-mirror.com)
        tokenizer: 用于计算 token 数的 tokenizer
        force_recompute: 强制重新计算，忽略缓存

    Returns:
        dict: {length_bin: [(item, token_count), ...]}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets: pip install datasets")
        return None

    # 默认长度区间 (token 数)
    if length_bins is None:
        length_bins = [8000, 16000, 32000, 64000, 128000]

    # 尝试从缓存加载
    cache_path = Path(LONGBENCH_CACHE_FILE)
    cached_token_counts = None

    if cache_path.exists() and not force_recompute:
        print(f"从缓存加载 token 计数: {cache_path}")
        try:
            with open(cache_path, "r") as f:
                cached_token_counts = json.load(f)
            print(f"  缓存中有 {len(cached_token_counts)} 条记录")
        except Exception as e:
            print(f"  缓存加载失败: {e}，将重新计算")
            cached_token_counts = None

    if use_mirror:
        print("使用 Hugging Face 镜像: hf-mirror.com")

    print("加载 LongBench V2 数据集...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    print(f"数据集大小: {len(dataset)} 条")

    # 如果缓存存在且数量匹配，直接使用
    if cached_token_counts is not None and len(cached_token_counts) == len(dataset):
        print("使用缓存的 token 计数")
        token_counts = cached_token_counts
    else:
        # 重新计算
        print("\n计算每个样本的 token 数...")
        token_counts = []

        for idx, item in enumerate(dataset):
            # 格式化完整 prompt，然后计算 token 数
            prompt = format_longbench_prompt(item)

            if tokenizer is not None:
                token_count = len(tokenizer.encode(prompt))
            else:
                # 没有 tokenizer 时用粗略估计 (英文约 4 字符/token)
                token_count = len(prompt) // 4

            token_counts.append(token_count)

            # 进度提示
            if (idx + 1) % 100 == 0:
                print(f"  已处理 {idx + 1}/{len(dataset)} 条")

        # 保存缓存
        print(f"\n保存 token 计数缓存到: {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(token_counts, f)

    # 按长度分组 (使用 float('inf') 表示超出最大区间的样本)
    grouped = {b: [] for b in length_bins}
    grouped[float('inf')] = []  # 超出最大区间的样本

    for idx, item in enumerate(dataset):
        token_count = token_counts[idx]

        # 找到合适的区间
        for i, bin_max in enumerate(length_bins):
            bin_min = length_bins[i-1] if i > 0 else 0
            if bin_min <= token_count < bin_max:
                grouped[bin_max].append((item, token_count))
                break
        else:
            # 超过最大区间，放到 overflow bin
            if token_count >= length_bins[-1]:
                grouped[float('inf')].append((item, token_count))

    # 打印统计
    print("\n按 token 数分组统计:")
    for bin_max in length_bins:
        bin_min = length_bins[length_bins.index(bin_max) - 1] if length_bins.index(bin_max) > 0 else 0
        count = len(grouped[bin_max])
        if count > 0:
            tokens = [t for _, t in grouped[bin_max]]
            avg_tokens = sum(tokens) // len(tokens)
            print(f"  [{bin_min:>6}, {bin_max:>6}) tokens: {count:>3} 条 (平均 {avg_tokens} tokens)")
        else:
            print(f"  [{bin_min:>6}, {bin_max:>6}) tokens: {count:>3} 条")

    # 打印超出范围的样本
    overflow_count = len(grouped[float('inf')])
    if overflow_count > 0:
        tokens = [t for _, t in grouped[float('inf')]]
        avg_tokens = sum(tokens) // len(tokens)
        print(f"  >= {length_bins[-1]:>6} tokens: {overflow_count:>3} 条 (平均 {avg_tokens} tokens) [不参与测试]")

    return grouped


def format_longbench_prompt(item: dict) -> str:
    """格式化 LongBench V2 的单个样本为 prompt"""
    context = item.get('context', '')
    question = item.get('question', '')

    # 构建选项
    choices = []
    for c in ['A', 'B', 'C', 'D']:
        choice_text = item.get(f'choice_{c}', '')
        if choice_text:
            choices.append(f"{c}. {choice_text}")

    prompt = f"""{context}

Question: {question}

{chr(10).join(choices)}

Please answer with the letter of the correct choice (A, B, C, or D)."""

    return prompt


def test_longbench_v2(
    length_bins: list = None,
    samples_per_bin: int = 1,
    max_tokens: int = 128,
    temperature: float = 0.7,
    use_mirror: bool = True,
    model_path: str = None,
    force_recompute: bool = False,
):
    """
    使用 LongBench V2 数据集测试不同 context length

    复现 ESS 论文的实验设置
    """
    print("=" * 60)
    print("LongBench V2 测试模式")
    print("=" * 60)

    # 加载 tokenizer
    tokenizer = get_tokenizer(model_path)
    if tokenizer is None:
        print("警告: 无法加载 tokenizer，将使用粗略估计")

    # 加载数据集
    grouped = load_longbench_v2(length_bins, use_mirror=use_mirror, tokenizer=tokenizer, force_recompute=force_recompute)
    if grouped is None:
        return

    print(f"\n每个长度区间测试 {samples_per_bin} 个样本")
    print(f"每个样本生成 {max_tokens} tokens")
    print()

    results = []

    # 只遍历有效的 bins，跳过 overflow (float('inf'))
    valid_bins = [k for k in sorted(grouped.keys()) if k != float('inf')]
    for bin_max in valid_bins:
        samples = grouped[bin_max]
        if not samples:
            bin_min = valid_bins[valid_bins.index(bin_max) - 1] if valid_bins.index(bin_max) > 0 else 0
            print(f"\n[跳过] 长度 [{bin_min}, {bin_max}) tokens: 无样本")
            continue

        # 选择指定数量的样本
        test_samples = samples[:samples_per_bin]

        bin_min = valid_bins[valid_bins.index(bin_max) - 1] if valid_bins.index(bin_max) > 0 else 0
        for idx, (item, token_count) in enumerate(test_samples):
            print(f"\n[{bin_min}-{bin_max}] 样本 {idx+1}/{len(test_samples)}, 长度 = {token_count} tokens")
            print("-" * 40)

            # 格式化 prompt
            prompt = format_longbench_prompt(item)
            print(f"Prompt 长度: {len(prompt)} 字符, {token_count} tokens")

            # 发送请求
            start_time = time.time()
            try:
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }
                response = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=600)
                result = response.json()
                elapsed = time.time() - start_time

                output_text = result.get("text", "")
                print(f"生成完成，耗时 {elapsed:.1f}s")
                print(f"输出: {output_text[:200]}...")

                # 检查答案
                answer = item.get('answer', '')
                print(f"正确答案: {answer}")

                results.append({
                    "length_bin": bin_max,
                    "token_count": token_count,
                    "elapsed": elapsed,
                    "success": True,
                    "answer": answer,
                    "output": output_text[:100],
                })
            except Exception as e:
                print(f"错误: {e}")
                results.append({
                    "length_bin": bin_max,
                    "token_count": token_count,
                    "success": False,
                    "error": str(e),
                })

            # 等待 collector 保存
            time.sleep(2)

    # 打印总结
    print("\n" + "=" * 60)
    print("LongBench V2 测试完成")
    print("=" * 60)
    for r in results:
        if r["success"]:
            print(f"  {r['token_count']:>6} tokens: ✓ 成功 ({r['elapsed']:.1f}s)")
        else:
            print(f"  {r['token_count']:>6} tokens: ✗ 失败")

    print("\n数据已收集到 topk_locality_data/topk_raw_data.pt")
    print("每个样本对应一个 session_id，可用 analyze_topk.py --plot 分析")


def test_multiple_lengths(
    context_lengths: list,
    max_tokens: int = 128,
    temperature: float = 0.7,
    base_file: str = None,
):
    """
    测试多个不同的 context length

    用于收集不同长度下的 intra-layer similarity 数据
    复现 ESS 论文的实验
    """
    print("=" * 60)
    print("多长度测试模式 (Intra-Layer Similarity 数据收集)")
    print("=" * 60)
    print(f"测试长度: {context_lengths}")
    print(f"每个长度生成 {max_tokens} tokens")
    print()

    # 如果提供了基础文件，读取内容
    base_text = None
    if base_file:
        with open(base_file, "r") as f:
            base_text = f.read()
        print(f"使用基础文件: {base_file}")

    results = []

    for i, target_len in enumerate(context_lengths):
        print(f"\n[{i+1}/{len(context_lengths)}] 测试 Context Length ≈ {target_len}")
        print("-" * 40)

        # 生成指定长度的 prompt
        prompt = generate_long_prompt(target_len, base_text)
        print(f"Prompt 长度: {len(prompt)} 字符 (目标 ~{target_len} tokens)")

        # 发送请求
        start_time = time.time()
        try:
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                },
            }
            response = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=300)
            result = response.json()
            elapsed = time.time() - start_time

            output_text = result.get("text", "")
            print(f"生成完成，耗时 {elapsed:.1f}s")
            print(f"输出: {output_text[:100]}...")

            results.append({
                "context_length": target_len,
                "prompt_chars": len(prompt),
                "output_tokens": len(output_text.split()),  # 粗略估计
                "elapsed": elapsed,
                "success": True,
            })
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                "context_length": target_len,
                "success": False,
                "error": str(e),
            })

        # 等待一下，让 collector 保存数据
        time.sleep(2)

    # 打印总结
    print("\n" + "=" * 60)
    print("测试完成总结")
    print("=" * 60)
    for r in results:
        if r["success"]:
            print(f"  {r['context_length']:>6} tokens: ✓ 成功 ({r['elapsed']:.1f}s)")
        else:
            print(f"  {r['context_length']:>6} tokens: ✗ 失败 ({r.get('error', 'unknown')})")

    print("\n提示: 数据已收集到 topk_locality_data/topk_raw_data.pt")
    print("      每个长度对应一个 session_id，可用 analyze_topk.py 分析")


def main():
    parser = argparse.ArgumentParser(description="测试 SGLang Server")
    parser.add_argument("--prompt", "-p", type=str, help="直接输入 prompt")
    parser.add_argument("--file", "-f", type=str, help="从文件读取 prompt")
    parser.add_argument("--max-tokens", "-m", type=int, default=1, help="最大生成 token 数")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="温度参数")
    parser.add_argument("--stream", "-s", action="store_true", help="使用流式输出")
    parser.add_argument("--chat", "-c", action="store_true", help="使用聊天模式")

    # 新增: 多长度测试
    parser.add_argument("--test-lengths", action="store_true",
                        help="测试多个不同的 context length (用于收集 intra-layer similarity 数据)")
    parser.add_argument("--lengths", type=str, default="4096,8192,16384,32768",
                        help="要测试的 context lengths，逗号分隔 (默认: 4096,8192,16384,32768)")

    # 新增: LongBench V2 测试
    parser.add_argument("--longbench", action="store_true",
                        help="使用 LongBench V2 数据集测试 (复现 ESS 论文)")
    parser.add_argument("--longbench-bins", type=str, default="8000,16000,32000,64000,128000",
                        help="LongBench 长度区间 (token 数)，逗号分隔 (默认: 8000,16000,32000,64000,128000)")
    parser.add_argument("--samples-per-bin", type=int, default=1,
                        help="每个长度区间测试的样本数 (默认: 1)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Tokenizer 模型路径 (默认: deepseek-ai/DeepSeek-V3.2)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="不使用 Hugging Face 国内镜像 (默认使用 hf-mirror.com)")
    parser.add_argument("--recompute", action="store_true",
                        help="强制重新计算 token 数，忽略缓存")

    args = parser.parse_args()

    # LongBench V2 测试模式
    if args.longbench:
        bins = [int(x.strip()) for x in args.longbench_bins.split(",")]
        test_longbench_v2(
            length_bins=bins,
            samples_per_bin=args.samples_per_bin,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_mirror=not args.no_mirror,
            model_path=args.model_path,
            force_recompute=args.recompute,
        )
        return

    # 多长度测试模式
    if args.test_lengths:
        lengths = [int(x.strip()) for x in args.lengths.split(",")]
        test_multiple_lengths(
            context_lengths=lengths,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            base_file=args.file,
        )
        return

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

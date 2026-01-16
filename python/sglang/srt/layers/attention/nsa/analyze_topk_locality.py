#!/usr/bin/env python3
"""
TopK Indices Data Loader and Plotting Utilities

加载收集的原始索引数据，提供绘图示例。

使用方法:
    # 作为模块导入
    from analyze_topk_locality import load_data, plot_layer_heatmap

    data = load_data("topk_raw_xxx.pt")
    plot_layer_heatmap(data, request_id, layer_id)

    # 或直接运行生成示例图
    python analyze_topk_locality.py /path/to/topk_raw.pt --plot-dir ./plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np


def load_data(path: str) -> Dict[str, Any]:
    """
    加载收集的原始数据

    返回数据结构:
    {
        "requests": {
            request_id: {
                "layers": {
                    layer_id: {
                        "topk_indices": Tensor (num_tokens, topk),
                        "seq_lens": List[int],
                        "num_tokens": int,
                        "topk": int,
                        ...
                    }
                }
            }
        },
        "raw_records": [...],  # 按时间顺序的所有记录
        "metadata": {...}
    }
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def get_indices_matrix(data: Dict, request_id: str) -> Tuple[torch.Tensor, List[int]]:
    """
    获取指定 request 所有层的索引矩阵

    返回:
        indices: Tensor of shape (num_layers, num_tokens, topk)
        layer_ids: 排序后的 layer id 列表
    """
    if request_id not in data["requests"]:
        raise ValueError(f"Request {request_id} not found")

    layers_data = data["requests"][request_id]["layers"]
    layer_ids = sorted(layers_data.keys())

    indices_list = []
    for layer_id in layer_ids:
        indices_list.append(layers_data[layer_id]["topk_indices"])

    # Stack: (num_layers, num_tokens, topk)
    indices = torch.stack(indices_list, dim=0)
    return indices, layer_ids


def get_layer_indices(data: Dict, request_id: str, layer_id: int) -> torch.Tensor:
    """获取指定 request 和 layer 的索引 Tensor"""
    return data["requests"][request_id]["layers"][layer_id]["topk_indices"]


def compute_layer_overlap_matrix(indices: torch.Tensor) -> np.ndarray:
    """
    计算层间重叠矩阵

    Args:
        indices: (num_layers, num_tokens, topk)

    Returns:
        overlap_matrix: (num_layers, num_layers) 每对层之间的平均重叠率
    """
    num_layers = indices.shape[0]
    overlap_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue

            # 计算每个 token 的重叠
            overlaps = []
            for t in range(indices.shape[1]):
                set_i = set(indices[i, t].tolist()) - {-1}
                set_j = set(indices[j, t].tolist()) - {-1}
                if len(set_i) > 0 and len(set_j) > 0:
                    overlap = len(set_i & set_j) / max(len(set_i), len(set_j))
                    overlaps.append(overlap)

            overlap_matrix[i, j] = np.mean(overlaps) if overlaps else 0.0

    return overlap_matrix


def compute_token_overlap_curve(indices: torch.Tensor) -> np.ndarray:
    """
    计算 token 间重叠曲线 (相邻 token 的重叠率随 token 位置的变化)

    Args:
        indices: (num_tokens, topk)

    Returns:
        overlap_curve: (num_tokens - 1,) 每对相邻 token 的重叠率
    """
    num_tokens = indices.shape[0]
    overlaps = []

    for t in range(num_tokens - 1):
        set_curr = set(indices[t].tolist()) - {-1}
        set_next = set(indices[t + 1].tolist()) - {-1}
        if len(set_curr) > 0 and len(set_next) > 0:
            overlap = len(set_curr & set_next) / max(len(set_curr), len(set_next))
            overlaps.append(overlap)
        else:
            overlaps.append(0.0)

    return np.array(overlaps)


def export_to_numpy(data: Dict, output_dir: str):
    """
    导出所有数据为 numpy 格式，方便用其他工具分析

    输出文件:
    - {request_id}_indices.npy: (num_layers, num_tokens, topk)
    - {request_id}_layer_ids.npy: layer id 列表
    - raw_records.npz: 所有原始记录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 request 导出
    for request_id, req_data in data["requests"].items():
        try:
            indices, layer_ids = get_indices_matrix(data, request_id)
            np.save(output_dir / f"{request_id}_indices.npy", indices.numpy())
            np.save(output_dir / f"{request_id}_layer_ids.npy", np.array(layer_ids))
            print(f"Exported {request_id}: {indices.shape}")
        except Exception as e:
            print(f"Failed to export {request_id}: {e}")

    # 导出原始记录 (简化版)
    raw_data = {
        "request_ids": [],
        "layer_ids": [],
        "steps": [],
        "num_tokens": [],
    }
    for record in data["raw_records"]:
        raw_data["request_ids"].append(record["request_id"])
        raw_data["layer_ids"].append(record["layer_id"])
        raw_data["steps"].append(record["step"])
        raw_data["num_tokens"].append(record["topk_indices"].shape[0])

    np.savez(output_dir / "raw_records.npz", **raw_data)
    print(f"Exported raw records index to {output_dir}/raw_records.npz")


# ============== 绘图函数 ==============

def plot_layer_heatmap(
    data: Dict,
    request_id: str,
    layer_id: int,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    绘制单层的索引热力图

    X 轴: token 位置
    Y 轴: topk 排名
    颜色: 索引值
    """
    import matplotlib.pyplot as plt

    indices = get_layer_indices(data, request_id, layer_id)
    # indices: (num_tokens, topk)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(indices.T.numpy(), aspect="auto", cmap="viridis")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("TopK Rank")
    ax.set_title(title or f"Layer {layer_id} TopK Indices")
    plt.colorbar(im, ax=ax, label="Index Value")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_layer_overlap_heatmap(
    data: Dict,
    request_id: str,
    output_path: Optional[str] = None,
):
    """
    绘制层间重叠热力图

    X/Y 轴: 层 ID
    颜色: 重叠率
    """
    import matplotlib.pyplot as plt

    indices, layer_ids = get_indices_matrix(data, request_id)
    overlap_matrix = compute_layer_overlap_matrix(indices)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(overlap_matrix, cmap="RdYlGn", vmin=0, vmax=1)

    # 设置刻度
    tick_step = max(1, len(layer_ids) // 10)
    tick_positions = list(range(0, len(layer_ids), tick_step))
    tick_labels = [str(layer_ids[i]) for i in tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Layer ID")
    ax.set_title(f"Inter-Layer Index Overlap (Request: {request_id[:8]}...)")
    plt.colorbar(im, ax=ax, label="Overlap Ratio")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_token_overlap_curve(
    data: Dict,
    request_id: str,
    layer_id: int,
    output_path: Optional[str] = None,
):
    """
    绘制 token 间重叠曲线
    """
    import matplotlib.pyplot as plt

    indices = get_layer_indices(data, request_id, layer_id)
    overlap_curve = compute_token_overlap_curve(indices)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(overlap_curve, linewidth=0.5)
    ax.axhline(y=np.mean(overlap_curve), color="r", linestyle="--",
               label=f"Mean: {np.mean(overlap_curve):.2%}")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Overlap with Next Token")
    ax.set_title(f"Layer {layer_id} Inter-Token Overlap")
    ax.legend()
    ax.set_ylim(0, 1)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_index_distribution(
    data: Dict,
    request_id: str,
    layer_id: int,
    output_path: Optional[str] = None,
):
    """
    绘制索引分布直方图
    """
    import matplotlib.pyplot as plt

    indices = get_layer_indices(data, request_id, layer_id)
    flat_indices = indices.flatten().numpy()
    valid_indices = flat_indices[flat_indices >= 0]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(valid_indices, bins=100, edgecolor="none", alpha=0.7)

    ax.set_xlabel("Index Value (KV Cache Position)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Layer {layer_id} Index Distribution")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_all_layers_comparison(
    data: Dict,
    request_id: str,
    token_idx: int = 0,
    output_path: Optional[str] = None,
):
    """
    比较同一 token 在所有层的索引选择

    X 轴: 层 ID
    Y 轴: topk 排名
    颜色: 索引值
    """
    import matplotlib.pyplot as plt

    indices, layer_ids = get_indices_matrix(data, request_id)
    # indices: (num_layers, num_tokens, topk)

    # 取指定 token 的数据
    token_indices = indices[:, token_idx, :]  # (num_layers, topk)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(token_indices.T.numpy(), aspect="auto", cmap="viridis")

    ax.set_xlabel("Layer ID")
    ax.set_ylabel("TopK Rank")
    ax.set_title(f"Token {token_idx}: Index Selection Across Layers")
    plt.colorbar(im, ax=ax, label="Index Value")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def generate_all_plots(data: Dict, output_dir: str):
    """生成所有示例图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取第一个 request 作为示例
    request_id = next(iter(data["requests"].keys()))
    layers = sorted(data["requests"][request_id]["layers"].keys())

    print(f"Generating plots for request {request_id[:16]}...")
    print(f"  Layers: {layers[:5]}...{layers[-5:]} (total {len(layers)})")

    # 1. 层间重叠热力图
    plot_layer_overlap_heatmap(
        data, request_id,
        output_path=str(output_dir / "layer_overlap_heatmap.png")
    )

    # 2. 几个代表层的索引热力图
    sample_layers = [layers[0], layers[len(layers)//2], layers[-1]]
    for layer_id in sample_layers:
        plot_layer_heatmap(
            data, request_id, layer_id,
            output_path=str(output_dir / f"layer_{layer_id}_heatmap.png")
        )

    # 3. Token 间重叠曲线
    plot_token_overlap_curve(
        data, request_id, layers[len(layers)//2],
        output_path=str(output_dir / "token_overlap_curve.png")
    )

    # 4. 索引分布
    plot_index_distribution(
        data, request_id, layers[len(layers)//2],
        output_path=str(output_dir / "index_distribution.png")
    )

    # 5. 跨层比较
    plot_all_layers_comparison(
        data, request_id, token_idx=0,
        output_path=str(output_dir / "cross_layer_token0.png")
    )

    print(f"All plots saved to {output_dir}")


def print_data_summary(data: Dict):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)

    metadata = data.get("metadata", {})
    print(f"Total records: {metadata.get('total_records', 'N/A')}")
    print(f"Num requests: {metadata.get('num_requests', 'N/A')}")
    print(f"Duration: {metadata.get('duration', 0):.2f}s")

    print("\nRequests:")
    for req_id, req_data in data["requests"].items():
        layers = sorted(req_data["layers"].keys())
        sample = req_data["layers"][layers[0]]
        print(f"  {req_id[:24]}...")
        print(f"    Layers: {len(layers)} ({layers[0]} to {layers[-1]})")
        print(f"    Tokens: {sample['num_tokens']}, TopK: {sample['topk']}")
        print(f"    Forward mode: {req_data['metadata'].get('forward_mode', 'N/A')}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Load and analyze TopK indices data")
    parser.add_argument("input", help="Path to topk_raw_xxx.pt file")
    parser.add_argument("--plot-dir", help="Generate plots to this directory")
    parser.add_argument("--export-numpy", help="Export data as numpy to this directory")
    parser.add_argument("--summary", action="store_true", help="Print data summary only")

    args = parser.parse_args()

    print(f"Loading {args.input}...")
    data = load_data(args.input)

    print_data_summary(data)

    if args.summary:
        return

    if args.export_numpy:
        export_to_numpy(data, args.export_numpy)

    if args.plot_dir:
        generate_all_plots(data, args.plot_dir)


if __name__ == "__main__":
    main()

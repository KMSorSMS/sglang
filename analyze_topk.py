#!/usr/bin/env python3
"""
TopK Locality 数据分析脚本

用法:
    python analyze_topk.py                    # 分析默认路径的数据
    python analyze_topk.py path/to/data.pt   # 分析指定文件
    python analyze_topk.py --plot             # 生成可视化图表
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch


def load_data(file_path: str) -> Dict:
    """加载保存的数据"""
    data = torch.load(file_path, weights_only=False)
    return data


def get_layer_ids(data: Dict) -> List[int]:
    """获取数据中所有的层 ID"""
    records = data.get("records", [])
    layer_ids = set()
    for record in records:
        layer_ids.add(record.get("layer_id", -1))
    return sorted(layer_ids)


def print_summary(data: Dict):
    """打印数据摘要"""
    metadata = data.get("metadata", {})
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)

    print("=" * 60)
    print("TopK Locality 数据摘要")
    print("=" * 60)
    print(f"总记录数: {metadata.get('total_records', len(records))}")
    print(f"层数: {len(layer_ids)} (layers: {min(layer_ids)}-{max(layer_ids)})")
    print(f"采集时长: {metadata.get('duration', 0):.2f} 秒")
    print()

    # 按 forward_mode 统计
    mode_counts = {}
    for record in records:
        mode = record.get("forward_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    print("按模式统计:")
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count} 条记录")
    print()

    # 按层统计（简化显示）
    layer_counts = {}
    for record in records:
        layer_id = record.get("layer_id", -1)
        layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1

    print(f"每层记录数: {layer_counts.get(layer_ids[0], 0)} (以 Layer {layer_ids[0]} 为例)")
    print()


def print_records(data: Dict, limit: int = 10):
    """打印原始记录详情"""
    records = data.get("records", [])

    print("=" * 60)
    print(f"原始记录 (前 {limit} 条)")
    print("=" * 60)

    for i, record in enumerate(records[:limit]):
        print(f"\n[Record {i}]")
        print(f"  layer_id: {record.get('layer_id', 'N/A')}")
        print(f"  forward_mode: {record.get('forward_mode', 'N/A')}")
        print(f"  seq_len: {record.get('seq_len', 0)} (KV cache 总长度)")
        print(f"  num_tokens: {record.get('num_tokens', 'N/A')}")
        print(f"  topk: {record.get('topk', 'N/A')}")

        positions = record.get("positions")
        if positions is not None:
            pos_list = positions.tolist()
            if len(pos_list) <= 5:
                print(f"  positions: {pos_list}")
            else:
                print(f"  positions: [{pos_list[0]}, {pos_list[1]}, ..., {pos_list[-1]}] (共 {len(pos_list)} 个)")

        indices = record.get("topk_indices")
        if indices is not None:
            print(f"  topk_indices shape: {indices.shape}")
            print(f"  topk_indices (前 3 个 query):")
            for j in range(min(3, indices.shape[0])):
                pos = positions[j].item() if positions is not None else j
                idx_list = indices[j].tolist()[:10]
                print(f"    query@{pos}: {idx_list}{'...' if len(indices.shape) > 1 and indices.shape[1] > 10 else ''}")


def analyze_distance(data: Dict, layer_id: Optional[int] = None):
    """分析 Query-Key 距离分布（核心局部性指标）"""
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)

    if layer_id is None:
        layer_id = layer_ids[0]  # 默认第一层

    layer_records = [r for r in records if r.get("layer_id") == layer_id]

    if not layer_records:
        print(f"没有找到 Layer {layer_id} 的记录")
        return

    print("=" * 60)
    print(f"Query-Key 距离分析 (Layer {layer_id})")
    print("=" * 60)

    all_distances = []
    all_recency_ratios = []  # 最近 10% 位置的选择比例

    for record in layer_records:
        positions = record.get("positions")
        indices = record.get("topk_indices")
        seq_len = record.get("seq_len", 0)

        if positions is None or indices is None:
            continue

        for i in range(positions.shape[0]):
            query_pos = positions[i].item()
            selected_keys = indices[i].tolist()

            # 计算距离: query_pos - key_pos (正数表示 key 在 query 之前)
            distances = [query_pos - k for k in selected_keys]
            all_distances.extend(distances)

            # 计算 recency: 选择的 key 中有多少在最近 10% 的位置
            if query_pos > 0:
                recent_threshold = max(1, int(query_pos * 0.9))  # 最近 10%
                recent_count = sum(1 for k in selected_keys if k >= recent_threshold)
                all_recency_ratios.append(recent_count / len(selected_keys))

    if all_distances:
        distances_tensor = torch.tensor(all_distances, dtype=torch.float)
        print(f"距离统计 (query_pos - key_pos):")
        print(f"  样本数: {len(all_distances)}")
        print(f"  均值: {distances_tensor.mean().item():.1f}")
        print(f"  标准差: {distances_tensor.std().item():.1f}")
        print(f"  最小值: {distances_tensor.min().item():.0f}")
        print(f"  最大值: {distances_tensor.max().item():.0f}")
        print(f"  中位数: {distances_tensor.median().item():.0f}")

        # 距离分布
        print(f"\n距离分布:")
        for threshold in [10, 50, 100, 500]:
            close_ratio = (distances_tensor.abs() <= threshold).float().mean().item()
            print(f"  距离 <= {threshold}: {close_ratio:.1%}")

    if all_recency_ratios:
        recency_tensor = torch.tensor(all_recency_ratios)
        print(f"\nRecency 偏好 (选择最近 10% key 的比例):")
        print(f"  平均: {recency_tensor.mean().item():.1%}")
        print(f"  标准差: {recency_tensor.std().item():.1%}")


def analyze_cross_layer(data: Dict):
    """分析跨层一致性"""
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)

    if len(layer_ids) < 2:
        print("需要至少 2 层数据才能分析跨层一致性")
        return

    print("=" * 60)
    print("跨层一致性分析")
    print("=" * 60)

    # 按 (forward_mode, positions) 分组，比较不同层的选择
    # 对于 decode，每个 position 对应一次 forward
    decode_records = [r for r in records if r.get("forward_mode") == "DECODE"]

    if not decode_records:
        print("没有 DECODE 模式的记录")
        return

    # 按 position 分组
    pos_to_layers = defaultdict(dict)  # pos -> {layer_id: indices}
    for record in decode_records:
        positions = record.get("positions")
        indices = record.get("topk_indices")
        layer_id = record.get("layer_id")

        if positions is not None and indices is not None and positions.shape[0] == 1:
            pos = positions[0].item()
            pos_to_layers[pos][layer_id] = set(indices[0].tolist())

    # 计算相邻层之间的 Jaccard 相似度
    layer_pairs_overlap = defaultdict(list)
    for pos, layer_dict in pos_to_layers.items():
        for i in range(len(layer_ids) - 1):
            l1, l2 = layer_ids[i], layer_ids[i + 1]
            if l1 in layer_dict and l2 in layer_dict:
                set1, set2 = layer_dict[l1], layer_dict[l2]
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    layer_pairs_overlap[(l1, l2)].append(jaccard)

    print("相邻层 Jaccard 相似度:")
    for (l1, l2), overlaps in sorted(layer_pairs_overlap.items()):
        if overlaps:
            avg = sum(overlaps) / len(overlaps)
            print(f"  Layer {l1} vs {l2}: {avg:.1%} (n={len(overlaps)})")


def analyze_decode_stability(data: Dict, layer_id: Optional[int] = None):
    """分析 decode 过程中选择的稳定性"""
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)

    if layer_id is None:
        layer_id = layer_ids[0]

    decode_records = [r for r in records
                      if r.get("forward_mode") == "DECODE" and r.get("layer_id") == layer_id]

    if len(decode_records) < 2:
        print(f"Layer {layer_id} 的 DECODE 记录不足")
        return

    print("=" * 60)
    print(f"Decode 稳定性分析 (Layer {layer_id})")
    print("=" * 60)

    # 按 position 排序
    decode_records = sorted(decode_records, key=lambda r: r.get("positions", torch.tensor([0]))[0].item())

    # 计算连续 decode step 之间的重叠
    overlaps = []
    prev_indices = None
    for record in decode_records:
        indices = record.get("topk_indices")
        if indices is not None and indices.shape[0] == 1:
            curr_indices = set(indices[0].tolist())
            if prev_indices is not None:
                if prev_indices or curr_indices:
                    jaccard = len(prev_indices & curr_indices) / len(prev_indices | curr_indices)
                    overlaps.append(jaccard)
            prev_indices = curr_indices

    if overlaps:
        overlaps_tensor = torch.tensor(overlaps)
        print(f"连续 decode step 的 Jaccard 重叠度:")
        print(f"  平均: {overlaps_tensor.mean().item():.1%}")
        print(f"  标准差: {overlaps_tensor.std().item():.1%}")
        print(f"  最小: {overlaps_tensor.min().item():.1%}")
        print(f"  最大: {overlaps_tensor.max().item():.1%}")


def plot_layer_overlap(records: list, layer_ids: list, output_dir, plt, np):
    """分析并可视化跨层重叠（热点 KV cache 分析）"""
    from collections import Counter

    # 按 position 分组，收集每层的选择
    # 对于 EXTEND (prefill)，一个 forward 有多个 position
    # 对于 DECODE，一个 forward 只有一个 position

    # 收集数据：position -> layer -> set of selected indices
    pos_layer_indices = defaultdict(lambda: defaultdict(set))

    for record in records:
        layer_id = record.get("layer_id")
        positions = record.get("positions")
        indices = record.get("topk_indices")

        if positions is None or indices is None:
            continue

        for i in range(positions.shape[0]):
            pos = positions[i].item()
            selected = set(indices[i].tolist())
            pos_layer_indices[pos][layer_id] = selected

    if not pos_layer_indices:
        print("  没有足够的数据进行跨层重叠分析")
        return

    # 只分析所有层都有数据的 position
    valid_positions = [pos for pos, layer_dict in pos_layer_indices.items()
                       if len(layer_dict) == len(layer_ids)]
    valid_positions = sorted(valid_positions)

    if not valid_positions:
        print("  没有所有层都有数据的 position")
        return

    print(f"  有效 position 数: {len(valid_positions)}")

    # ========== 分析 1: 每个 position 的跨层重叠度 ==========
    # 计算：intersection(所有层选择) / union(所有层选择)
    overlap_ratios = []  # 全层交集占比
    hot_counts = []  # 被多层选中的 KV 数量

    for pos in valid_positions:
        layer_dict = pos_layer_indices[pos]
        all_sets = [layer_dict[lid] for lid in layer_ids]

        # 全层交集
        intersection = set.intersection(*all_sets)
        union = set.union(*all_sets)

        overlap_ratio = len(intersection) / len(union) if union else 0
        overlap_ratios.append(overlap_ratio)

        # 统计每个 KV index 被多少层选中
        kv_counter = Counter()
        for s in all_sets:
            kv_counter.update(s)

        # "热点" = 被超过一半层选中的 KV
        hot_threshold = len(layer_ids) // 2 + 1
        hot_count = sum(1 for count in kv_counter.values() if count >= hot_threshold)
        hot_counts.append(hot_count)

    # ========== 分析 2: 全局热点 KV 统计 ==========
    # 统计每个 KV index 在所有 position 的所有层中被选中的总次数
    global_kv_counter = Counter()
    for pos in valid_positions:
        layer_dict = pos_layer_indices[pos]
        for lid in layer_ids:
            global_kv_counter.update(layer_dict[lid])

    # 找出最热门的 KV positions
    top_hot_kvs = global_kv_counter.most_common(20)

    print(f"  全层交集占比 (Jaccard): 平均 {np.mean(overlap_ratios):.1%}, 最大 {max(overlap_ratios):.1%}")
    print(f"  热点 KV 数量 (被 >50% 层选中): 平均 {np.mean(hot_counts):.1f}")
    print(f"  Top 10 最热 KV positions: {[kv for kv, _ in top_hot_kvs[:10]]}")

    # ========== 绘图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 跨层重叠度随 position 变化
    ax1 = axes[0, 0]
    ax1.plot(valid_positions, overlap_ratios, marker='.', markersize=2, alpha=0.7)
    ax1.axhline(y=np.mean(overlap_ratios), color='r', linestyle='--',
                label=f'mean: {np.mean(overlap_ratios):.1%}')
    ax1.set_xlabel("Query Position")
    ax1.set_ylabel("All-Layer Intersection / Union")
    ax1.set_title("Cross-Layer Overlap Ratio per Position")
    ax1.legend()
    ax1.set_ylim(0, 1)

    # 图2: 热点 KV 数量随 position 变化
    ax2 = axes[0, 1]
    ax2.plot(valid_positions, hot_counts, marker='.', markersize=2, alpha=0.7)
    ax2.axhline(y=np.mean(hot_counts), color='r', linestyle='--',
                label=f'mean: {np.mean(hot_counts):.1f}')
    ax2.set_xlabel("Query Position")
    ax2.set_ylabel("Hot KV Count (selected by >50% layers)")
    ax2.set_title("Hot KV Cache Count per Position")
    ax2.legend()

    # 图3: 全局 KV 热度分布（直方图）
    ax3 = axes[1, 0]
    kv_frequencies = list(global_kv_counter.values())
    ax3.hist(kv_frequencies, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel("Selection Frequency (across all positions & layers)")
    ax3.set_ylabel("Number of KV Positions")
    ax3.set_title("KV Cache Hotness Distribution")
    ax3.axvline(x=np.mean(kv_frequencies), color='r', linestyle='--',
                label=f'mean: {np.mean(kv_frequencies):.1f}')
    ax3.legend()

    # 图4: Top 20 最热 KV 的频率
    ax4 = axes[1, 1]
    if top_hot_kvs:
        kv_positions = [str(kv) for kv, _ in top_hot_kvs]
        kv_freqs = [freq for _, freq in top_hot_kvs]
        ax4.barh(range(len(kv_positions)), kv_freqs)
        ax4.set_yticks(range(len(kv_positions)))
        ax4.set_yticklabels(kv_positions)
        ax4.set_xlabel("Selection Frequency")
        ax4.set_ylabel("KV Position")
        ax4.set_title("Top 20 Hottest KV Positions")
        ax4.invert_yaxis()  # 最热的在上面

    plt.suptitle(f"Cross-Layer KV Cache Overlap Analysis ({len(layer_ids)} layers, {len(valid_positions)} positions)")
    plt.tight_layout()
    output_file = output_dir / "topk_layer_overlap.png"
    plt.savefig(output_file, dpi=150)
    print(f"  跨层重叠分析图已保存到: {output_file}")
    plt.close()

    # ========== 额外分析：热点 KV 的位置特征 ==========
    # 热点 KV 是靠近开头还是靠近当前位置？
    if top_hot_kvs and valid_positions:
        hot_kv_positions = [kv for kv, _ in top_hot_kvs[:10]]
        avg_query_pos = np.mean(valid_positions)
        hot_kv_avg = np.mean(hot_kv_positions)
        print(f"  热点 KV 位置特征:")
        print(f"    平均 query position: {avg_query_pos:.0f}")
        print(f"    Top10 热点 KV 平均位置: {hot_kv_avg:.0f}")
        print(f"    热点倾向: {'靠近开头 (可能是重要 context)' if hot_kv_avg < avg_query_pos * 0.5 else '靠近当前位置 (recency bias)' if hot_kv_avg > avg_query_pos * 0.8 else '分布较均匀'}")


def plot_all(data: Dict, output_dir: str = "."):
    """生成所有可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    print("输出文件:")
    print("  1. topk_locality_analysis.png - 基础统计 (2x2)")
    print("     - 左上: Query-Key 距离分布")
    print("     - 右上: Recency 偏好")
    print("     - 左下: 相邻层一致性")
    print("     - 右下: Decode 趋势")
    print("  2. topk_layer_overlap.png - 跨层重叠分析 (2x2) ★重要")
    print("     - 左上: 跨层重叠度随 position 变化")
    print("     - 右上: 热点 KV 数量随 position 变化")
    print("     - 左下: KV 热度分布直方图")
    print("     - 右下: Top 20 最热 KV positions")
    print("  3. topk_prefill_heatmap.png - 多层 Prefill 热力图")
    print("  4. topk_decode_heatmap.png - 多层 Decode 热力图")
    print()

    output_dir = Path(output_dir)
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)
    print(f"[DEBUG] Total records: {len(records)}, Layers: {layer_ids}")

    # ========== 图1: Query-Key 距离分布 (多层对比) ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    layer_distances = {}
    for layer_id in layer_ids:
        layer_records = [r for r in records if r.get("layer_id") == layer_id]
        distances = []
        for record in layer_records:
            positions = record.get("positions")
            indices = record.get("topk_indices")
            if positions is None or indices is None:
                continue
            for i in range(positions.shape[0]):
                query_pos = positions[i].item()
                selected_keys = indices[i].tolist()
                distances.extend([query_pos - k for k in selected_keys])
        if distances:
            layer_distances[layer_id] = distances

    # 绘制每层的距离分布箱线图
    print(f"[DEBUG] Distance data: {len(layer_distances)} layers with data")
    if layer_distances:
        box_data = [layer_distances[lid] for lid in sorted(layer_distances.keys())]
        box_labels = [f"L{lid}" for lid in sorted(layer_distances.keys())]
        ax1.boxplot(box_data, tick_labels=box_labels, showfliers=False)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Query-Key Distance")
        ax1.set_title("Query-Key Distance Distribution per Layer")
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    else:
        ax1.set_title("No Distance Data")

    # ========== 图2: Recency 偏好 per Layer ==========
    ax2 = axes[0, 1]
    layer_recency = {}
    for layer_id in layer_ids:
        layer_records = [r for r in records if r.get("layer_id") == layer_id]
        recency_ratios = []
        for record in layer_records:
            positions = record.get("positions")
            indices = record.get("topk_indices")
            if positions is None or indices is None:
                continue
            for i in range(positions.shape[0]):
                query_pos = positions[i].item()
                if query_pos > 10:  # 只统计有足够历史的 query
                    selected_keys = indices[i].tolist()
                    recent_threshold = int(query_pos * 0.9)
                    recent_count = sum(1 for k in selected_keys if k >= recent_threshold)
                    recency_ratios.append(recent_count / len(selected_keys))
        if recency_ratios:
            layer_recency[layer_id] = np.mean(recency_ratios)

    print(f"[DEBUG] Recency data: {len(layer_recency)} layers with data")
    if layer_recency:
        ax2.bar(layer_recency.keys(), layer_recency.values())
        ax2.set_xlabel("Layer ID")
        ax2.set_ylabel("Recency Ratio")
        ax2.set_title("Recency Preference per Layer\n(% of keys in last 10% positions)")
        ax2.set_ylim(0, 1)
    else:
        ax2.set_title("No Recency Data")

    # ========== 图3: 跨层一致性 ==========
    ax3 = axes[1, 0]
    decode_records = [r for r in records if r.get("forward_mode") == "DECODE"]

    pos_to_layers = defaultdict(dict)
    for record in decode_records:
        positions = record.get("positions")
        indices = record.get("topk_indices")
        layer_id = record.get("layer_id")
        if positions is not None and indices is not None and positions.shape[0] == 1:
            pos = positions[0].item()
            pos_to_layers[pos][layer_id] = set(indices[0].tolist())

    layer_pairs_overlap = {}
    for i in range(len(layer_ids) - 1):
        l1, l2 = layer_ids[i], layer_ids[i + 1]
        overlaps = []
        for pos, layer_dict in pos_to_layers.items():
            if l1 in layer_dict and l2 in layer_dict:
                set1, set2 = layer_dict[l1], layer_dict[l2]
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    overlaps.append(jaccard)
        if overlaps:
            layer_pairs_overlap[f"L{l1}-L{l2}"] = np.mean(overlaps)

    print(f"[DEBUG] Cross-layer data: {len(layer_pairs_overlap)} layer pairs")
    if layer_pairs_overlap:
        ax3.bar(range(len(layer_pairs_overlap)), list(layer_pairs_overlap.values()))
        ax3.set_xticks(range(len(layer_pairs_overlap)))
        ax3.set_xticklabels(list(layer_pairs_overlap.keys()), rotation=45)
        ax3.set_xlabel("Layer Pair")
        ax3.set_ylabel("Jaccard Similarity")
        ax3.set_title("Cross-Layer Consistency\n(Adjacent Layer Similarity)")
        ax3.set_ylim(0, 1)
    else:
        ax3.set_title("No Cross-Layer Data")

    # ========== 图4: Decode 过程中 mean index 变化 ==========
    ax4 = axes[1, 1]
    first_layer = layer_ids[0]
    layer_decode_records = [r for r in decode_records if r.get("layer_id") == first_layer]
    print(f"[DEBUG] Decode records for layer {first_layer}: {len(layer_decode_records)}")

    if layer_decode_records:
        layer_decode_records = sorted(layer_decode_records,
                                       key=lambda r: r.get("positions", torch.tensor([0]))[0].item())
        query_positions = []
        mean_indices = []
        for r in layer_decode_records[:200]:  # 最多显示 200 个点
            positions = r.get("positions")
            indices = r.get("topk_indices")
            if positions is not None and indices is not None:
                query_positions.append(positions[0].item())
                mean_indices.append(indices.float().mean().item())

        print(f"[DEBUG] Decode trend data points: {len(mean_indices)}")
        if mean_indices:
            ax4.plot(query_positions, mean_indices, marker=".", markersize=3, alpha=0.7, label='mean index')
            # 添加 y=x 参考线（表示选择的 key 平均位置等于 query 位置）
            ax4.plot(query_positions, query_positions, 'r--', alpha=0.5, label='y=x')
            ax4.set_xlabel("Query Position")
            ax4.set_ylabel("Mean Selected Key Index")
            ax4.set_title(f"Index Trend During Decoding (Layer {first_layer})")
            ax4.legend()
        else:
            ax4.set_title(f"No Decode Data (Layer {first_layer})")
    else:
        ax4.set_title("No Decode Records")

    plt.tight_layout()
    output_file = output_dir / "topk_locality_analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"分析图已保存到: {output_file}")
    plt.close()

    # ========== 图5: 跨层重叠分析 ==========
    print("\n[分析跨层重叠...]")
    plot_layer_overlap(records, layer_ids, output_dir, plt, np)

    # ========== 图6: 多层热力图对比 ==========
    prefill_records = [r for r in records if r.get("forward_mode") == "EXTEND"]
    print(f"[DEBUG] Total prefill records: {len(prefill_records)}")

    if prefill_records:
        # 收集每层的数据
        layer_data = {}
        for layer_id in layer_ids:
            layer_prefill = [r for r in prefill_records if r.get("layer_id") == layer_id]
            if layer_prefill:
                all_indices = []
                all_positions = []
                for record in layer_prefill:
                    indices = record.get("topk_indices")
                    positions = record.get("positions")
                    if indices is not None and positions is not None:
                        all_indices.append(indices)
                        all_positions.append(positions)
                if all_indices:
                    layer_data[layer_id] = {
                        "indices": torch.cat(all_indices, dim=0),
                        "positions": torch.cat(all_positions, dim=0),
                    }

        print(f"[DEBUG] Layers with prefill data: {list(layer_data.keys())}")

        if layer_data:
            n_layers = len(layer_data)

            # 找到所有层共有的 token 范围
            min_tokens = min(d["indices"].shape[0] for d in layer_data.values())
            n_tokens = min(50, min_tokens)  # 取前 50 个 token 对比
            topk = layer_data[layer_ids[0]]["indices"].shape[1]
            topk_to_show = min(20, topk)  # 只显示前 20 个 topk

            # 使用 gridspec 来更好地控制布局
            fig = plt.figure(figsize=(4 * n_layers + 1, 5))
            gs = fig.add_gridspec(1, n_layers + 1, width_ratios=[1] * n_layers + [0.05])

            axes = []
            for idx, layer_id in enumerate(sorted(layer_data.keys())):
                ax = fig.add_subplot(gs[0, idx])
                axes.append(ax)

                data = layer_data[layer_id]
                indices = data["indices"][:n_tokens, :topk_to_show]
                positions = data["positions"][:n_tokens]

                im = ax.imshow(indices.numpy(), aspect="auto", cmap="viridis")
                ax.set_title(f"Layer {layer_id}", fontsize=10)
                ax.set_xlabel("TopK Rank", fontsize=8)
                if idx == 0:
                    ax.set_ylabel("Query Position", fontsize=8)

                # Y 轴: 均匀选取 6 个刻度
                n_yticks = min(6, n_tokens)
                ytick_indices = np.linspace(0, n_tokens - 1, n_yticks, dtype=int)
                ax.set_yticks(ytick_indices)
                ax.set_yticklabels([f"{positions[i].item()}" for i in ytick_indices], fontsize=7)
                ax.tick_params(axis='x', labelsize=7)

            # 添加 colorbar
            cax = fig.add_subplot(gs[0, -1])
            fig.colorbar(im, cax=cax, label="Selected KV Index")

            plt.suptitle(f"TopK Indices Heatmap - Prefill\n(first {n_tokens} tokens, top {topk_to_show} ranks)", fontsize=11)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            output_file = output_dir / "topk_prefill_heatmap.png"
            plt.savefig(output_file, dpi=150)
            print(f"多层热力图已保存到: {output_file}")
            plt.close()

            # ========== 图6: 多层 Decode 热力图（如果有足够数据）==========
            if decode_records:
                # 收集每层的 decode 数据
                layer_decode_data = {}
                for layer_id in layer_ids:
                    layer_dec = [r for r in decode_records if r.get("layer_id") == layer_id]
                    layer_dec = sorted(layer_dec, key=lambda r: r.get("positions", torch.tensor([0]))[0].item())
                    if layer_dec:
                        all_indices = []
                        all_positions = []
                        for record in layer_dec:
                            indices = record.get("topk_indices")
                            positions = record.get("positions")
                            if indices is not None and positions is not None:
                                all_indices.append(indices)
                                all_positions.append(positions)
                        if all_indices:
                            layer_decode_data[layer_id] = {
                                "indices": torch.cat(all_indices, dim=0),
                                "positions": torch.cat(all_positions, dim=0),
                            }

                if layer_decode_data:
                    n_layers = len(layer_decode_data)

                    min_steps = min(d["indices"].shape[0] for d in layer_decode_data.values())
                    n_steps = min(50, min_steps)
                    topk = layer_decode_data[layer_ids[0]]["indices"].shape[1]
                    topk_to_show = min(20, topk)

                    # 使用 gridspec 来更好地控制布局
                    fig = plt.figure(figsize=(4 * n_layers + 1, 5))
                    gs = fig.add_gridspec(1, n_layers + 1, width_ratios=[1] * n_layers + [0.05])

                    for idx, layer_id in enumerate(sorted(layer_decode_data.keys())):
                        ax = fig.add_subplot(gs[0, idx])

                        data = layer_decode_data[layer_id]
                        indices = data["indices"][:n_steps, :topk_to_show]
                        positions = data["positions"][:n_steps]

                        im = ax.imshow(indices.numpy(), aspect="auto", cmap="viridis")
                        ax.set_title(f"Layer {layer_id}", fontsize=10)
                        ax.set_xlabel("TopK Rank", fontsize=8)
                        if idx == 0:
                            ax.set_ylabel("Query Position", fontsize=8)

                        # Y 轴: 均匀选取 6 个刻度
                        n_yticks = min(6, n_steps)
                        ytick_indices = np.linspace(0, n_steps - 1, n_yticks, dtype=int)
                        ax.set_yticks(ytick_indices)
                        ax.set_yticklabels([f"{positions[i].item()}" for i in ytick_indices], fontsize=7)
                        ax.tick_params(axis='x', labelsize=7)

                    # 添加 colorbar
                    cax = fig.add_subplot(gs[0, -1])
                    fig.colorbar(im, cax=cax, label="Selected KV Index")

                    plt.suptitle(f"TopK Indices Heatmap - Decode\n(first {n_steps} steps, top {topk_to_show} ranks)", fontsize=11)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    output_file = output_dir / "topk_decode_heatmap.png"
                    plt.savefig(output_file, dpi=150)
                    print(f"Decode 热力图已保存到: {output_file}")
                    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析 TopK Locality 数据")
    parser.add_argument("file", nargs="?", default="topk_locality_data/topk_raw_data.pt",
                        help="数据文件路径")
    parser.add_argument("--records", "-r", type=int, default=0,
                        help="显示的记录数量 (0=不显示)")
    parser.add_argument("--layer", "-l", type=int, default=None,
                        help="分析的层 ID (默认=第一层)")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--output", "-o", type=str, default=".",
                        help="图表输出目录")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        print("\n可用的数据文件:")
        data_dir = Path("topk_locality_data")
        if data_dir.exists():
            for f in data_dir.glob("*.pt"):
                print(f"  {f}")
        return

    print(f"加载数据: {file_path}")
    data = load_data(str(file_path))

    # 打印摘要
    print_summary(data)

    # 打印原始记录
    if args.records > 0:
        print_records(data, args.records)

    # 距离分析
    analyze_distance(data, args.layer)

    # 跨层一致性
    analyze_cross_layer(data)

    # Decode 稳定性
    analyze_decode_stability(data, args.layer)

    # 生成图表
    if args.plot:
        plot_all(data, args.output)


if __name__ == "__main__":
    main()

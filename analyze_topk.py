#!/usr/bin/env python3
"""
TopK Locality 数据分析脚本

用法:
    python analyze_topk.py                    # 分析默认路径的数据
    python analyze_topk.py path/to/data.pt   # 分析指定文件
    python analyze_topk.py --plot             # 生成可视化图表
"""

import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch


# For multiprocessing workers (fork-friendly): parent fills this mapping and workers read it.
_MP_LAYER_ENTRIES = None
_MP_INCLUDE_GAPS = False
_MP_PER_SESSION = False
_MP_RECORDS = None
_MP_COLLECT_PER_SESSION_STATS = False


def _mp_worker_compute_layer_overlap_from_record_idxs(packed_args):
    """Worker (fork path): compute one layer from record indices.

    packed_args: (layer_id, record_indices)
    Uses global _MP_RECORDS inherited via fork.
    """
    layer_id, record_indices = packed_args
    records = _MP_RECORDS
    include_gaps = _MP_INCLUDE_GAPS
    per_session = _MP_PER_SESSION
    collect_stats = _MP_COLLECT_PER_SESSION_STATS

    if records is None or not record_indices:
        return {"layer_id": layer_id, "steps": 0, "has_data": False, "per_session": []}

    # Build: session -> {pos: indices_tensor_row}
    session_pos_to_row: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
    for ridx in record_indices:
        r = records[ridx]
        positions = r.get("positions")
        indices = r.get("topk_indices")
        if positions is None or indices is None:
            continue
        sid = int(r.get("session_id", 0))

        for i in range(int(positions.shape[0])):
            pos = int(positions[i].item())
            # If the same pos appears multiple times, keep the last one.
            session_pos_to_row[sid][pos] = indices[i]

    layer_overlaps: List[float] = []
    per_session_stats = []

    for sid, pos_to_row in session_pos_to_row.items():
        if len(pos_to_row) < 2:
            continue

        sorted_pos = sorted(pos_to_row.keys())
        overlaps: List[float] = []

        prev_pos = sorted_pos[0]
        prev_set = set(int(x) for x in pos_to_row[prev_pos].tolist())

        for curr_pos in sorted_pos[1:]:
            curr_set = set(int(x) for x in pos_to_row[curr_pos].tolist())
            delta = curr_pos - prev_pos
            if delta <= 0:
                prev_pos, prev_set = curr_pos, curr_set
                continue
            if (not include_gaps) and delta != 1:
                prev_pos, prev_set = curr_pos, curr_set
                continue

            denom = len(prev_set | curr_set)
            jaccard = (len(prev_set & curr_set) / denom) if denom else 1.0
            overlaps.append(jaccard)
            layer_overlaps.append(jaccard)
            prev_pos, prev_set = curr_pos, curr_set

        if (per_session or collect_stats) and overlaps:
            t = torch.tensor(overlaps, dtype=torch.float)
            per_session_stats.append(
                {
                    "session_id": sid,
                    "steps": len(overlaps),
                    "mean": float(t.mean().item()),
                    "median": float(t.median().item()),
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                }
            )

    if not layer_overlaps:
        return {"layer_id": layer_id, "steps": 0, "has_data": False, "per_session": per_session_stats}

    t_layer = torch.tensor(layer_overlaps, dtype=torch.float)
    return {
        "layer_id": layer_id,
        "steps": len(layer_overlaps),
        "has_data": True,
        "mean": float(t_layer.mean().item()),
        "median": float(t_layer.median().item()),
        "min": float(t_layer.min().item()),
        "max": float(t_layer.max().item()),
        "per_session": per_session_stats,
    }


def _mp_worker_compute_layer_overlap(layer_id: int):
    """Worker: compute adjacent-step overlap stats for one layer.

    Reads shared globals populated in the parent process.
    Returns a dict with per-layer summary and optional per-session summaries.
    """
    layer_entries = _MP_LAYER_ENTRIES.get(layer_id, []) if _MP_LAYER_ENTRIES is not None else []
    include_gaps = _MP_INCLUDE_GAPS
    per_session = _MP_PER_SESSION
    collect_stats = _MP_COLLECT_PER_SESSION_STATS

    # entries: list[(session_id, position, tuple[int,...])]
    if len(layer_entries) < 2:
        return {"layer_id": layer_id, "steps": 0, "has_data": False, "per_session": []}

    layer_entries = sorted(layer_entries, key=lambda x: (x[0], x[1]))

    layer_overlaps: List[float] = []
    per_session_stats = []

    prev_sid = None
    prev_pos = None
    prev_tuple = None
    current_overlaps: List[float] = []

    def _flush_session(sid: int, overlaps: List[float]):
        if not overlaps:
            return
        t = torch.tensor(overlaps, dtype=torch.float)
        per_session_stats.append(
            {
                "session_id": sid,
                "steps": len(overlaps),
                "mean": float(t.mean().item()),
                "median": float(t.median().item()),
                "min": float(t.min().item()),
                "max": float(t.max().item()),
            }
        )

    for sid, pos, idx_tuple in layer_entries:
        if prev_sid is None:
            prev_sid, prev_pos, prev_tuple = sid, pos, idx_tuple
            continue

        if sid != prev_sid:
            if per_session:
                _flush_session(prev_sid, current_overlaps)
            current_overlaps = []
            prev_sid, prev_pos, prev_tuple = sid, pos, idx_tuple
            continue

        delta = pos - prev_pos
        if delta <= 0:
            prev_pos, prev_tuple = pos, idx_tuple
            continue
        if (not include_gaps) and delta != 1:
            prev_pos, prev_tuple = pos, idx_tuple
            continue

        prev_set = set(prev_tuple)
        curr_set = set(idx_tuple)
        denom = len(prev_set | curr_set)
        jaccard = (len(prev_set & curr_set) / denom) if denom else 1.0
        current_overlaps.append(jaccard)
        layer_overlaps.append(jaccard)

        prev_pos, prev_tuple = pos, idx_tuple

    if per_session or collect_stats:
        _flush_session(prev_sid, current_overlaps)

    if not layer_overlaps:
        return {"layer_id": layer_id, "steps": 0, "has_data": False, "per_session": per_session_stats}

    t_layer = torch.tensor(layer_overlaps, dtype=torch.float)
    return {
        "layer_id": layer_id,
        "steps": len(layer_overlaps),
        "has_data": True,
        "mean": float(t_layer.mean().item()),
        "median": float(t_layer.median().item()),
        "min": float(t_layer.min().item()),
        "max": float(t_layer.max().item()),
        "per_session": per_session_stats,
    }


def _spawn_worker_compute_layer_overlap(packed_args):
    """Picklable wrapper for non-fork start methods (spawn/forkserver)."""
    lid, entries, include_gaps_local, per_session_local = packed_args
    global _MP_LAYER_ENTRIES, _MP_INCLUDE_GAPS, _MP_PER_SESSION, _MP_COLLECT_PER_SESSION_STATS
    _MP_LAYER_ENTRIES = {lid: entries}
    _MP_INCLUDE_GAPS = include_gaps_local
    _MP_PER_SESSION = per_session_local
    # For spawn path we always compute per-session stats when caller wants them.
    _MP_COLLECT_PER_SESSION_STATS = True
    return _mp_worker_compute_layer_overlap(lid)


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


def get_session_ids(data: Dict) -> List[int]:
    """获取数据中所有的 session ID"""
    records = data.get("records", [])
    session_ids = set()
    for record in records:
        session_ids.add(record.get("session_id", 0))
    return sorted(session_ids)


def filter_by_session(data: Dict, session_id: int) -> Dict:
    """过滤指定 session 的数据"""
    records = data.get("records", [])
    filtered = [r for r in records if r.get("session_id", 0) == session_id]
    return {
        "records": filtered,
        "metadata": {
            **data.get("metadata", {}),
            "total_records": len(filtered),
            "filtered_session": session_id,
        }
    }


def print_summary(data: Dict):
    """打印数据摘要"""
    metadata = data.get("metadata", {})
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)
    session_ids = get_session_ids(data)

    print("=" * 60)
    print("TopK Locality 数据摘要")
    print("=" * 60)
    print(f"总记录数: {metadata.get('total_records', len(records))}")
    print(f"总 Session 数: {len(session_ids)} (IDs: {session_ids})")
    print(f"层数: {len(layer_ids)} (layers: {min(layer_ids)}-{max(layer_ids)})")
    print(f"采集时长: {metadata.get('duration', 0):.2f} 秒")
    print()

    # 按 session 统计 (只看 layer 0 的记录来统计 token 数)
    session_stats = defaultdict(lambda: {"EXTEND": 0, "DECODE": 0, "total": 0, "prefill_tokens": 0, "decode_tokens": 0})
    for record in records:
        sid = record.get("session_id", 0)
        mode = record.get("forward_mode", "unknown")
        layer_id = record.get("layer_id", -1)
        num_tokens = record.get("num_tokens", 0)

        session_stats[sid][mode] = session_stats[sid].get(mode, 0) + 1
        session_stats[sid]["total"] += 1

        # 只用 layer 0 来统计 token 数量（避免重复计算）
        if layer_id == layer_ids[0]:
            if mode == "EXTEND":
                session_stats[sid]["prefill_tokens"] += num_tokens
            elif mode == "DECODE":
                session_stats[sid]["decode_tokens"] += num_tokens

    print("按 Session 统计:")
    for sid in sorted(session_stats.keys()):
        stats = session_stats[sid]
        extend = stats.get("EXTEND", 0)
        decode = stats.get("DECODE", 0)
        prefill_tokens = stats.get("prefill_tokens", 0)
        decode_tokens = stats.get("decode_tokens", 0)
        print(f"  Session {sid}: {stats['total']} 条记录 (EXTEND: {extend}, DECODE: {decode})")
        print(f"             Prefill 长度: {prefill_tokens} tokens, Decode 长度: {decode_tokens} tokens")
    print()

    # DEBUG: 检查 EXTEND 和 DECODE 的 position 范围
    print("[DEBUG] Position 范围分析 (以 Layer 0 为例):")
    for sid in sorted(session_stats.keys()):
        sid_records = [r for r in records if r.get("session_id", 0) == sid and r.get("layer_id") == layer_ids[0]]

        # EXTEND records
        extend_records = [r for r in sid_records if r.get("forward_mode") == "EXTEND"]
        extend_positions = []
        for r in extend_records:
            pos = r.get("positions")
            if pos is not None:
                extend_positions.extend(pos.tolist())

        # DECODE records
        decode_records_sid = [r for r in sid_records if r.get("forward_mode") == "DECODE"]
        decode_positions = []
        for r in decode_records_sid:
            pos = r.get("positions")
            if pos is not None:
                decode_positions.extend(pos.tolist())

        print(f"  Session {sid}:")
        if extend_positions:
            print(f"    EXTEND positions: min={min(extend_positions)}, max={max(extend_positions)}, count={len(extend_positions)}")
            # 检查是否有 gap
            sorted_ext = sorted(set(extend_positions))
            if len(sorted_ext) > 1:
                gaps = []
                for i in range(1, len(sorted_ext)):
                    gap = sorted_ext[i] - sorted_ext[i-1]
                    if gap > 1:
                        gaps.append((sorted_ext[i-1], sorted_ext[i], gap))
                if gaps:
                    print(f"    EXTEND gaps (大于1): {gaps[:5]}{'...' if len(gaps) > 5 else ''}")
        else:
            print(f"    EXTEND positions: (无数据)")

        if decode_positions:
            print(f"    DECODE positions: min={min(decode_positions)}, max={max(decode_positions)}, count={len(decode_positions)}")
        else:
            print(f"    DECODE positions: (无数据)")

        # 检查 EXTEND 和 DECODE 之间的 gap
        if extend_positions and decode_positions:
            extend_max = max(extend_positions)
            decode_min = min(decode_positions)
            gap = decode_min - extend_max
            print(f"    EXTEND->DECODE gap: {extend_max} -> {decode_min} = {gap} tokens")
            if gap > 1:
                print(f"    [警告] Gap 过大! 可能有数据丢失或 chunked prefill")
    print()

    # 按 forward_mode 统计
    mode_counts = {}
    for record in records:
        mode = record.get("forward_mode", "unknown")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    print("按模式统计 (全部):")
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
    # 多个 recency 阈值: 最近 10%, 30%, 50%, 70%, 90%
    recency_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_recency_ratios = {thresh: [] for thresh in recency_thresholds}

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

            # 调试: 打印负距离的情况
            negative_keys = []
            negative_dists = []
            for k, d in zip(selected_keys, distances):
                if d < 0:
                    negative_keys.append(k)
                    negative_dists.append(d)
            if negative_keys:
                print(f"[DEBUG] 负距离! query_pos={query_pos}")
                print(f"        负的 selected_keys: {negative_keys[:20]}{'...' if len(negative_keys) > 20 else ''}")
                print(f"        对应的 distances:   {negative_dists[:20]}{'...' if len(negative_dists) > 20 else ''}")

            all_distances.extend(distances)

            # 计算多个 recency 阈值
            if query_pos > 0:
                for thresh in recency_thresholds:
                    # thresh=0.1 表示最近 10%，即 key >= query_pos * 0.9
                    recent_threshold = max(1, int(query_pos * (1 - thresh)))
                    recent_count = sum(1 for k in selected_keys if k >= recent_threshold)
                    all_recency_ratios[thresh].append(recent_count / len(selected_keys))
            else:
                # 应该不会存在这种情况
                raise ValueError(f"Invalid query_pos: {query_pos}. Query position must be greater than 0.")

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

    if all_recency_ratios[recency_thresholds[0]]:
        print(f"\nRecency 偏好 (选择最近 X% key 的比例):")
        for thresh in recency_thresholds:
            ratios = all_recency_ratios[thresh]
            recency_tensor = torch.tensor(ratios)
            pct = int(thresh * 100)
            print(f"  最近 {pct:2d}%: 平均 {recency_tensor.mean().item():.1%}, 标准差 {recency_tensor.std().item():.1%}")


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
    step_pairs = []  # 记录 (prev_pos, curr_pos) 对
    prev_indices = None
    prev_pos = None
    for record in decode_records:
        indices = record.get("topk_indices")
        positions = record.get("positions")
        if indices is not None and indices.shape[0] == 1:
            curr_indices = set(indices[0].tolist())
            curr_pos = positions[0].item() if positions is not None else None
            if prev_indices is not None:
                if prev_indices or curr_indices:
                    jaccard = len(prev_indices & curr_indices) / len(prev_indices | curr_indices)
                    overlaps.append(jaccard)
                    step_pairs.append((prev_pos, curr_pos))
            prev_indices = curr_indices
            prev_pos = curr_pos

    if overlaps:
        overlaps_tensor = torch.tensor(overlaps)
        print(f"连续 decode step 的 Jaccard 重叠度:")
        print(f"  平均: {overlaps_tensor.mean().item():.1%}")
        print(f"  标准差: {overlaps_tensor.std().item():.1%}")
        print(f"  最小: {overlaps_tensor.min().item():.1%}")
        print(f"  最大: {overlaps_tensor.max().item():.1%}")

        # 打印逐步重叠详情 (前5个 + 后5个)
        print(f"\n  逐步重叠详情 (共 {len(overlaps)} 步):")
        show_n = 5
        for i in range(min(show_n, len(overlaps))):
            prev_p, curr_p = step_pairs[i]
            print(f"    step {prev_p} -> {curr_p}: {overlaps[i]:.1%}")
        if len(overlaps) > show_n * 2:
            print(f"    ... (省略 {len(overlaps) - show_n * 2} 步) ...")
        if len(overlaps) > show_n:
            for i in range(max(show_n, len(overlaps) - show_n), len(overlaps)):
                prev_p, curr_p = step_pairs[i]
                print(f"    step {prev_p} -> {curr_p}: {overlaps[i]:.1%}")


def analyze_adjacent_step_overlap(
    data: Dict,
    layer_id: Optional[int] = None,
    session_id: Optional[int] = None,
    mode: str = "decode",
    include_gaps: bool = False,
    per_session: bool = False,
    collect_results: bool = False,
):
    """分析同一层相邻 step 的重叠率（Jaccard）。

    - 相邻 step: 默认只统计 position 差值为 1 的连续步（更贴近“逐 token decode step”）。
    - include_gaps=True: 也把 delta>1 的步纳入统计（用于数据不连续/分块时的粗略观察）。
    - mode: decode / extend / all
    """
    records = data.get("records", [])
    layer_ids = get_layer_ids(data)
    if not layer_ids:
        print("没有找到任何 layer 记录")
        return

    layers_to_analyze = [layer_id] if layer_id is not None else layer_ids

    mode_upper = mode.upper()
    if mode_upper not in ("DECODE", "EXTEND", "ALL"):
        raise ValueError(f"Invalid mode: {mode}")

    session_ids = get_session_ids(data)
    if session_id is not None:
        if session_id not in session_ids:
            print(f"错误: session_id={session_id} 不存在，可用的 session IDs: {session_ids}")
            return
        target_sessions = [session_id]
    else:
        target_sessions = session_ids

    print("=" * 60)
    if layer_id is None:
        print("同层相邻 Step 重叠率 (All Layers)")
    else:
        print(f"同层相邻 Step 重叠率 (Layer {layer_id})")
    print("=" * 60)
    print(f"mode={mode_upper}, include_gaps={include_gaps}")

    # Step 1) Build lightweight bucket: layer -> record indices (fast, single pass)
    # Also compute session context length (max position) using the first layer to avoid duplicates.
    layers_set = set(layers_to_analyze)
    sessions_set = set(target_sessions)
    layer_record_idxs: Dict[int, List[int]] = {lid: [] for lid in layers_to_analyze}

    first_layer_id = layer_ids[0]
    session_context_len: Dict[int, int] = {}

    print("[MP] Building layer->record buckets ...", flush=True)
    for ridx, r in enumerate(records):
        lid = r.get("layer_id")
        if lid not in layers_set:
            continue
        sid = r.get("session_id", 0)
        if sid not in sessions_set:
            continue

        # context length (use first layer only)
        if lid == first_layer_id:
            positions = r.get("positions")
            if positions is not None and positions.numel() > 0:
                try:
                    max_pos = int(positions.max().item())
                    prev = session_context_len.get(int(sid), -1)
                    if max_pos > prev:
                        session_context_len[int(sid)] = max_pos
                except Exception:
                    pass

        fm = str(r.get("forward_mode", "unknown")).upper()
        if mode_upper != "ALL" and fm != mode_upper:
            continue
        # Keep record idx; worker will expand positions/indices.
        layer_record_idxs[lid].append(ridx)

    # Step 2) Parallel per-layer computation
    results = []
    start_method = mp.get_start_method()
    use_fork = start_method == "fork"

    # User asked: one process per layer.
    max_workers = max(1, len(layers_to_analyze))
    print(f"[MP] start_method={start_method}, workers={max_workers}, layers={len(layers_to_analyze)}", flush=True)

    if use_fork:
        global _MP_RECORDS, _MP_INCLUDE_GAPS, _MP_PER_SESSION, _MP_COLLECT_PER_SESSION_STATS
        _MP_RECORDS = records
        _MP_INCLUDE_GAPS = include_gaps
        _MP_PER_SESSION = per_session
        _MP_COLLECT_PER_SESSION_STATS = bool(collect_results)

        if max_workers == 1:
            results = [
                _mp_worker_compute_layer_overlap_from_record_idxs((lid, layer_record_idxs.get(lid, [])))
                for lid in layers_to_analyze
            ]
        else:
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                futures = {
                    ex.submit(
                        _mp_worker_compute_layer_overlap_from_record_idxs,
                        (lid, layer_record_idxs.get(lid, [])),
                    ): lid
                    for lid in layers_to_analyze
                }
                total = len(futures)
                done = 0
                report_every = max(1, total // 10)
                for fut in as_completed(futures):
                    results.append(fut.result())
                    done += 1
                    if done % report_every == 0 or done == total:
                        print(f"[MP] done {done}/{total} layers", flush=True)
    else:
        # spawn/forkserver: fallback to the previous "pre-extract layer entries" approach.
        print("[MP] Non-fork start method detected; pre-extracting entries for spawn ...", flush=True)
        layer_entries: Dict[int, List] = {lid: [] for lid in layers_to_analyze}
        for r in records:
            lid = r.get("layer_id")
            if lid not in layers_set:
                continue
            sid = r.get("session_id", 0)
            if sid not in sessions_set:
                continue
            fm = str(r.get("forward_mode", "unknown")).upper()
            if mode_upper != "ALL" and fm != mode_upper:
                continue
            positions = r.get("positions")
            indices = r.get("topk_indices")
            if positions is None or indices is None:
                continue
            for i in range(int(positions.shape[0])):
                pos = int(positions[i].item())
                idx_tuple = tuple(int(x) for x in indices[i].tolist())
                layer_entries[lid].append((sid, pos, idx_tuple))

        ctx = mp.get_context(start_method)
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            packed = [(lid, layer_entries[lid], include_gaps, per_session) for lid in layers_to_analyze]
            total = len(packed)
            done = 0
            report_every = max(1, total // 10)
            for res in ex.map(_spawn_worker_compute_layer_overlap, packed):
                results.append(res)
                done += 1
                if done % report_every == 0 or done == total:
                    print(f"[MP] done {done}/{total} layers", flush=True)

    # Step 3) Merge + print in stable order
    results_by_layer = {r["layer_id"]: r for r in results}
    any_layer_has_data = False

    collected = {
        "mode": mode_upper,
        "include_gaps": include_gaps,
        "layers": {},  # lid -> {"summary": {...}, "sessions": {sid: {...}}}
        "session_context_len": session_context_len,
    }

    for lid in sorted(layers_to_analyze):
        r = results_by_layer.get(lid)
        if not r or not r.get("has_data"):
            continue
        any_layer_has_data = True

        # Store results for plotting (and potential downstream uses)
        sessions_map = {int(s["session_id"]): s for s in r.get("per_session", [])}
        collected["layers"][int(lid)] = {
            "summary": {
                "steps": int(r["steps"]),
                "mean": float(r["mean"]),
                "median": float(r["median"]),
                "min": float(r["min"]),
                "max": float(r["max"]),
            },
            "sessions": sessions_map,
        }

        if per_session:
            for s in sorted(r.get("per_session", []), key=lambda x: x["session_id"]):
                print(
                    f"Layer {lid} | Session {s['session_id']}: steps={s['steps']} | mean={s['mean']:.1%} | "
                    f"median={s['median']:.1%} | min={s['min']:.1%} | max={s['max']:.1%}",
                    flush=True,
                )

        print(
            f"Layer {lid}: steps={r['steps']} | mean={r['mean']:.1%} | "
            f"median={r['median']:.1%} | min={r['min']:.1%} | max={r['max']:.1%}",
            flush=True,
        )

    if not any_layer_has_data:
        print("没有足够的相邻 step 数据。可以尝试:")
        print("  - 用 --mode all 覆盖 EXTEND+DECODE")
        print("  - 加 --include-gaps 放宽连续性限制")
        return collected if collect_results else None

    return collected if collect_results else None


def plot_adjacent_step_overlap_heatmaps(result: Dict, output_dir: str = "."):
    """从 analyze_adjacent_step_overlap 的结果画三张热力图：mean / median / min。

    X: layer_id
    Y: session 的 max context length (max position)
    Cell: 该 session 在该 layer 上相邻 step overlap 的统计值
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib 和 numpy: pip install matplotlib numpy")
        return

    if not result or not result.get("layers"):
        print("没有可绘图的数据（result 为空或没有 layer 结果）")
        return

    layers = sorted(result["layers"].keys())
    ctx = result.get("session_context_len", {})
    session_ids = sorted(ctx.keys(), key=lambda sid: ctx.get(sid, 0))
    if not session_ids:
        # Fallback: take sessions from any layer results
        sids = set()
        for lid in layers:
            sids.update(result["layers"][lid].get("sessions", {}).keys())
        session_ids = sorted(sids)

    y_labels = [str(ctx.get(sid, sid)) for sid in session_ids]

    mean_mat = np.full((len(session_ids), len(layers)), np.nan, dtype=np.float32)
    median_mat = np.full((len(session_ids), len(layers)), np.nan, dtype=np.float32)
    min_mat = np.full((len(session_ids), len(layers)), np.nan, dtype=np.float32)

    for j, lid in enumerate(layers):
        sessions_map = result["layers"][lid].get("sessions", {})
        for i, sid in enumerate(session_ids):
            s = sessions_map.get(int(sid))
            if not s:
                continue
            mean_mat[i, j] = float(s.get("mean", np.nan))
            median_mat[i, j] = float(s.get("median", np.nan))
            min_mat[i, j] = float(s.get("min", np.nan))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot_one(mat, metric_name: str, file_name: str):
        fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.45), max(4, len(session_ids) * 0.45)))
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_xlabel("Layer ID")
        ax.set_ylabel("Context Length (session max position)")
        ax.set_title(f"Adjacent-step overlap ({metric_name}) | mode={result.get('mode')} | include_gaps={result.get('include_gaps')}")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(len(session_ids)))
        ax.set_yticklabels(y_labels)

        plt.colorbar(im, ax=ax, label=metric_name)
        plt.tight_layout()
        out = output_dir / file_name
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图表已保存到: {out}")

    base = f"adjacent_step_overlap_mode{str(result.get('mode', 'NA')).lower()}" + ("_gaps" if result.get("include_gaps") else "")
    _plot_one(mean_mat, "mean", f"{base}_mean.png")
    _plot_one(median_mat, "median", f"{base}_median.png")
    _plot_one(min_mat, "min", f"{base}_min.png")


def plot_adjacent_step_overlap(
    data: Dict,
    output_dir: str = ".",
    layer_id: Optional[int] = None,
    session_id: Optional[int] = None,
    mode: str = "decode",
    include_gaps: bool = False,
):
    """绘制“同一层相邻 step 重叠率”热力图（按 session 分行）。

    - X 轴: layer_id
    - Y 轴: session 的上下文长度（该 session 的最大 position）
    - 格子值: 该 (session, layer) 下相邻 step overlap 的统计量
    - 输出三张图: mean / median / min
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    records = data.get("records", [])
    all_layer_ids = get_layer_ids(data)
    if not all_layer_ids:
        print("没有找到任何 layer 记录")
        return

    layers_to_plot = [layer_id] if layer_id is not None else all_layer_ids

    mode_upper = mode.upper()
    if mode_upper not in ("DECODE", "EXTEND", "ALL"):
        raise ValueError(f"Invalid mode: {mode}")

    session_ids = get_session_ids(data)
    if session_id is not None:
        if session_id not in session_ids:
            print(f"错误: session_id={session_id} 不存在，可用的 session IDs: {session_ids}")
            return
        target_sessions = [session_id]
    else:
        target_sessions = session_ids

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("绘制 同层相邻 step 重叠率热力图 (按 session 分行)")
    print("=" * 60)
    print(f"layers={len(layers_to_plot)} (example: {layers_to_plot[0]}..{layers_to_plot[-1]}), mode={mode_upper}, include_gaps={include_gaps}")

    layers_set = set(layers_to_plot)
    sessions_set = set(target_sessions)

    # 预处理：
    # 1) session_context_len: 每个 session 的最大 position
    # 2) (sid, lid) -> record indices (避免后面重复扫全量 records)
    session_context_len: Dict[int, int] = {sid: 0 for sid in target_sessions}
    sl_record_idxs: Dict[tuple, List[int]] = defaultdict(list)

    for ridx, r in enumerate(records):
        lid = r.get("layer_id")
        if lid not in layers_set:
            continue
        sid = int(r.get("session_id", 0))
        if sid not in sessions_set:
            continue

        fm = str(r.get("forward_mode", "unknown")).upper()
        if mode_upper != "ALL" and fm != mode_upper:
            continue

        positions = r.get("positions")
        indices = r.get("topk_indices")
        if positions is None or indices is None:
            continue

        # 记录 idx
        sl_record_idxs[(sid, lid)].append(ridx)

        # 更新上下文长度（max position）
        try:
            max_pos = int(positions.max().item())
            if max_pos > session_context_len.get(sid, 0):
                session_context_len[sid] = max_pos
        except Exception:
            # positions 可能为空/异常，忽略
            pass

    # 过滤掉没有任何数据的 session
    valid_sessions = [sid for sid in target_sessions if session_context_len.get(sid, 0) > 0 or any((sid, lid) in sl_record_idxs for lid in layers_to_plot)]
    if not valid_sessions:
        print("没有足够的数据绘图（没有找到满足条件的 session/layer 记录）")
        return

    # 按上下文长度排序 session（Y 轴）
    valid_sessions = sorted(valid_sessions, key=lambda s: session_context_len.get(s, 0))

    # heatmaps: rows=session, cols=layer
    h_mean = np.full((len(valid_sessions), len(layers_to_plot)), np.nan, dtype=np.float32)
    h_median = np.full_like(h_mean, np.nan)
    h_min = np.full_like(h_mean, np.nan)

    for i, sid in enumerate(valid_sessions):
        for j, lid in enumerate(layers_to_plot):
            rec_idxs = sl_record_idxs.get((sid, lid), [])
            if not rec_idxs:
                continue

            # pos -> indices_row (同 pos 取最后一次)
            pos_to_row: Dict[int, torch.Tensor] = {}
            for ridx in rec_idxs:
                r = records[ridx]
                positions = r.get("positions")
                indices = r.get("topk_indices")
                if positions is None or indices is None:
                    continue
                for k in range(int(positions.shape[0])):
                    pos = int(positions[k].item())
                    pos_to_row[pos] = indices[k]

            if len(pos_to_row) < 2:
                continue

            sorted_pos = sorted(pos_to_row.keys())
            overlaps: List[float] = []
            prev_pos = sorted_pos[0]
            prev_set = set(int(x) for x in pos_to_row[prev_pos].tolist())
            for curr_pos in sorted_pos[1:]:
                curr_set = set(int(x) for x in pos_to_row[curr_pos].tolist())
                delta = curr_pos - prev_pos
                if delta <= 0:
                    prev_pos, prev_set = curr_pos, curr_set
                    continue
                if (not include_gaps) and delta != 1:
                    prev_pos, prev_set = curr_pos, curr_set
                    continue
                denom = len(prev_set | curr_set)
                jaccard = (len(prev_set & curr_set) / denom) if denom else 1.0
                overlaps.append(jaccard)
                prev_pos, prev_set = curr_pos, curr_set

            if not overlaps:
                continue

            t = torch.tensor(overlaps, dtype=torch.float)
            h_mean[i, j] = float(t.mean().item())
            h_median[i, j] = float(t.median().item())
            h_min[i, j] = float(t.min().item())

    y_labels = [str(session_context_len.get(sid, 0)) for sid in valid_sessions]

    def _plot_heatmap(values: "np.ndarray", title: str, out_name: str):
        fig, ax = plt.subplots(figsize=(max(10, len(layers_to_plot) * 0.45), max(4, len(valid_sessions) * 0.45)))
        im = ax.imshow(values, aspect="auto", vmin=0, vmax=1, interpolation="nearest", cmap="YlGnBu")
        ax.set_xlabel("Layer ID")
        ax.set_ylabel("Context Length (max position per session)")
        ax.set_title(title)
        ax.set_xticks(range(len(layers_to_plot)))
        ax.set_xticklabels(layers_to_plot)
        ax.set_yticks(range(len(valid_sessions)))
        ax.set_yticklabels(y_labels)
        plt.colorbar(im, ax=ax, label="Adjacent-step overlap")
        plt.tight_layout()
        out_file = output_dir / out_name
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图表已保存到: {out_file}")

    suffix = f"mode_{mode.lower()}" + ("_gaps" if include_gaps else "")
    if layer_id is not None:
        suffix += f"_layer{layer_id}"
    if session_id is not None:
        suffix += f"_session{session_id}"

    _plot_heatmap(h_mean, f"Adjacent-step overlap (mean) | {suffix}", f"adjacent_step_overlap_mean_{suffix}.png")
    _plot_heatmap(h_median, f"Adjacent-step overlap (median) | {suffix}", f"adjacent_step_overlap_median_{suffix}.png")
    _plot_heatmap(h_min, f"Adjacent-step overlap (min) | {suffix}", f"adjacent_step_overlap_min_{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description="分析 TopK Locality 数据")
    parser.add_argument("--file", nargs="?", default="topk_locality_data/topk_raw_data.pt",
                        help="数据文件路径")
    parser.add_argument("--records", "-r", type=int, default=0,
                        help="显示的记录数量 (0=不显示)")
    parser.add_argument("--layer", "-l", type=int, default=None,
                        help="分析的层 ID (不指定则分析所有层)")
    parser.add_argument("--session", "-s", type=int, default=None,
                        help="只分析指定的 session ID (默认=全部)")
    parser.add_argument("--plot", "-p", action="store_true",
                        help="生成可视化图表")
    parser.add_argument("--output", "-o", type=str, default=".",
                        help="图表输出目录")
    parser.add_argument("--mode", type=str, default="all", choices=["decode", "extend", "all"],
                        help="相邻 step 重叠率统计使用的 forward_mode (默认: all)")
    parser.add_argument("--include-gaps", action="store_true",
                        help="把 position 不连续(delta>1)的步也计入重叠率统计")
    parser.add_argument("--no-per-session", action="store_true",
                        help="不打印每个 session 的相邻 step 重叠率统计 (默认会打印)")
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

    # 打印摘要（全量数据）
    print_summary(data)

    # 如果指定了 session，过滤数据
    if args.session is not None:
        session_ids = get_session_ids(data)
        if args.session not in session_ids:
            print(f"错误: session_id={args.session} 不存在，可用的 session IDs: {session_ids}")
            return
        print(f">>> 过滤 Session {args.session} 的数据进行分析")
        data = filter_by_session(data, args.session)
        print(f">>> 过滤后记录数: {len(data['records'])}")
        print()

    # 打印原始记录
    if args.records > 0:
        print_records(data, args.records)

    # 我们关心的核心指标：同一层相邻 step 重叠率（cache 潜力）
    analysis_result = analyze_adjacent_step_overlap(
        data,
        layer_id=args.layer,
        session_id=args.session,
        mode=args.mode,
        include_gaps=args.include_gaps,
        per_session=not args.no_per_session,
        collect_results=args.plot,
    )

    # # 距离分析
    # analyze_distance(data, args.layer)

    # # 跨层一致性
    # analyze_cross_layer(data)

    # # Decode 稳定性
    # analyze_decode_stability(data, args.layer)

    # 生成图表
    if args.plot:
        plot_adjacent_step_overlap_heatmaps(analysis_result, output_dir=args.output)


if __name__ == "__main__":
    main()

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


def _mp_worker_compute_layer_overlap_from_record_idxs(packed_args):
    """Worker (fork path): compute one layer from record indices.

    packed_args: (layer_id, record_indices)
    Uses global _MP_RECORDS inherited via fork.
    """
    layer_id, record_indices = packed_args
    records = _MP_RECORDS
    include_gaps = _MP_INCLUDE_GAPS
    per_session = _MP_PER_SESSION

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

        if per_session and overlaps:
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

    if per_session:
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
    global _MP_LAYER_ENTRIES, _MP_INCLUDE_GAPS, _MP_PER_SESSION
    _MP_LAYER_ENTRIES = {lid: entries}
    _MP_INCLUDE_GAPS = include_gaps_local
    _MP_PER_SESSION = per_session_local
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
    layers_set = set(layers_to_analyze)
    sessions_set = set(target_sessions)
    layer_record_idxs: Dict[int, List[int]] = {lid: [] for lid in layers_to_analyze}

    print("[MP] Building layer->record buckets ...", flush=True)
    for ridx, r in enumerate(records):
        lid = r.get("layer_id")
        if lid not in layers_set:
            continue
        sid = r.get("session_id", 0)
        if sid not in sessions_set:
            continue
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
        global _MP_RECORDS, _MP_INCLUDE_GAPS, _MP_PER_SESSION
        _MP_RECORDS = records
        _MP_INCLUDE_GAPS = include_gaps
        _MP_PER_SESSION = per_session

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
    for lid in sorted(layers_to_analyze):
        r = results_by_layer.get(lid)
        if not r or not r.get("has_data"):
            continue
        any_layer_has_data = True

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
        return


def plot_intra_layer_similarity(data: Dict, output_dir: str = "."):
    """
    绘制论文风格的 Intra-Layer Similarity 热力图
    复现 ESS 论文 (arxiv 2512.10576) 的图表风格

    X 轴: Layer IDs
    Y 轴: Context Length (不同 session 对应不同的 prompt 长度)
    颜色: Jaccard Similarity (每个 session 在该 layer 的平均相似度)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    records = data.get("records", [])
    layer_ids = get_layer_ids(data)
    session_ids = get_session_ids(data)
    output_dir = Path(output_dir)

    print("=" * 60)
    print("绘制 Intra-Layer Similarity 热力图 (论文风格)")
    print("=" * 60)

    # 为每个 session 的每层计算平均 Jaccard 相似度
    # 结构: {session_id: {layer_id: mean_similarity}}
    session_layer_similarity = {sid: {} for sid in session_ids}

    # 同时记录每个 session 的 context length (用于 Y 轴标签)
    session_context_len = {}

    for session_id in session_ids:
        for layer_id in layer_ids:
            # 获取该 session 该 layer 的所有记录 (EXTEND + DECODE)
            layer_records = [r for r in records
                              if r.get("layer_id") == layer_id
                              and r.get("session_id", 0) == session_id]

            if not layer_records:
                continue

            # 收集所有 (position, indices) 对，然后按 position 排序
            pos_indices_list = []  # [(pos, indices_set), ...]

            for record in layer_records:
                indices = record.get("topk_indices")
                positions = record.get("positions")

                if indices is None or positions is None:
                    continue

                # EXTEND: 多个 position; DECODE: 单个 position
                for i in range(positions.shape[0]):
                    pos = positions[i].item()
                    idx_set = set(indices[i].tolist())
                    pos_indices_list.append((pos, idx_set))

            # 按 position 排序
            pos_indices_list.sort(key=lambda x: x[0])

            if len(pos_indices_list) < 2:
                continue

            # 计算相邻 position 的 Jaccard 相似度
            similarities = []
            max_pos = 0

            for i in range(1, len(pos_indices_list)):
                prev_pos, prev_indices = pos_indices_list[i - 1]
                curr_pos, curr_indices = pos_indices_list[i]
                max_pos = max(max_pos, curr_pos)

                if prev_indices or curr_indices:
                    jaccard = len(prev_indices & curr_indices) / len(prev_indices | curr_indices)
                else:
                    jaccard = 1.0
                similarities.append(jaccard)

            if similarities:
                session_layer_similarity[session_id][layer_id] = np.mean(similarities)
                session_context_len[session_id] = max_pos

    # 过滤掉没有数据的 session
    valid_sessions = [sid for sid in session_ids if session_layer_similarity[sid]]
    if not valid_sessions:
        print("没有足够的数据 (需要至少 2 个 position)")
        return

    # 按 context length 排序 session
    valid_sessions = sorted(valid_sessions, key=lambda s: session_context_len.get(s, 0))

    print(f"有效 Session 数: {len(valid_sessions)}")
    print(f"Layer 数量: {len(layer_ids)}")
    for sid in valid_sessions:
        ctx_len = session_context_len.get(sid, 0)
        print(f"  Session {sid}: Context Length ≈ {ctx_len}")

    # 构建热力图矩阵
    # 行: session (context length), 列: layer id
    heatmap_data = np.zeros((len(valid_sessions), len(layer_ids)))

    for i, sid in enumerate(valid_sessions):
        for j, lid in enumerate(layer_ids):
            heatmap_data[i, j] = session_layer_similarity[sid].get(lid, np.nan)

    # ==================== 绘制热力图 ====================
    fig, ax = plt.subplots(figsize=(max(12, len(layer_ids) * 0.5), max(4, len(valid_sessions) * 0.5)))

    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlGnBu',
                   vmin=0, vmax=1, interpolation='nearest')

    # 设置坐标轴
    ax.set_xlabel("Layer IDs", fontsize=12)
    ax.set_ylabel("Context Length", fontsize=12)
    ax.set_title("Intra-Layer Similarity Across Different Context Lengths", fontsize=14)

    # X 轴: Layer IDs
    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids)

    # Y 轴: Context Length (来自 session)
    ax.set_yticks(range(len(valid_sessions)))
    y_labels = [str(session_context_len.get(sid, f"S{sid}")) for sid in valid_sessions]
    ax.set_yticklabels(y_labels)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, label="Similarity")

    plt.tight_layout()
    output_file = output_dir / "intra_layer_similarity.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"热力图已保存到: {output_file}")
    plt.close()

    # ==================== 打印统计信息 ====================
    print("\n各层平均相似度 (跨所有 context length):")
    for j, lid in enumerate(layer_ids):
        col_data = heatmap_data[:, j]
        valid_data = col_data[~np.isnan(col_data)]
        if len(valid_data) > 0:
            print(f"  Layer {lid}: {np.mean(valid_data):.1%}")


def plot_decode_stability(data: Dict, output_dir: str = "."):
    """绘制多层 Decode 稳定性图 (Jaccard 重叠度)"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    records = data.get("records", [])
    layer_ids = get_layer_ids(data)
    output_dir = Path(output_dir)

    print("=" * 60)
    print("绘制 Decode 稳定性图 (多层叠加)")
    print("=" * 60)

    # 为每层计算 decode 稳定性
    layer_data = {}  # layer_id -> {"positions": [...], "overlaps": [...]}

    for layer_id in layer_ids:
        # 获取该层的 DECODE 记录，按 position 排序
        decode_records = [r for r in records
                          if r.get("forward_mode") == "DECODE" and r.get("layer_id") == layer_id]
        decode_records = sorted(decode_records,
                                 key=lambda r: r.get("positions", torch.tensor([0]))[0].item())

        if len(decode_records) < 2:
            continue

        positions = []
        overlaps = []
        prev_indices = None

        for record in decode_records:
            indices = record.get("topk_indices")
            pos = record.get("positions")

            if indices is not None and pos is not None and indices.shape[0] == 1:
                curr_indices = set(indices[0].tolist())
                curr_pos = pos[0].item()

                if prev_indices is not None:
                    # 计算 Jaccard 相似度
                    if prev_indices or curr_indices:
                        jaccard = len(prev_indices & curr_indices) / len(prev_indices | curr_indices)
                    else:
                        jaccard = 1.0
                    positions.append(curr_pos)
                    overlaps.append(jaccard)

                prev_indices = curr_indices

        if positions:
            layer_data[layer_id] = {"positions": positions, "overlaps": overlaps}

    if not layer_data:
        print("没有足够的 DECODE 数据")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 6))

    # 使用不同颜色绘制每层
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_ids)))

    for idx, layer_id in enumerate(sorted(layer_data.keys())):
        data_layer = layer_data[layer_id]
        positions = data_layer["positions"]
        overlaps = data_layer["overlaps"]

        ax.plot(positions, overlaps,
                label=f"Layer {layer_id}",
                color=colors[idx],
                alpha=0.7,
                linewidth=1.5)

    ax.set_xlabel("Query Position (Decode Step)", fontsize=12)
    ax.set_ylabel("Jaccard Similarity with Previous Step", fontsize=12)
    ax.set_title("Decode Stability: Consecutive Step Overlap (All Layers)", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 添加平均线
    all_overlaps = []
    for layer_id in layer_data:
        all_overlaps.extend(layer_data[layer_id]["overlaps"])
    if all_overlaps:
        avg = np.mean(all_overlaps)
        ax.axhline(y=avg, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {avg:.1%}')
        # 更新图例
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    output_file = output_dir / "decode_stability.png"
    plt.savefig(output_file, dpi=150)
    print(f"图表已保存到: {output_file}")
    plt.close()

    # 打印统计信息
    print("\n各层统计:")
    for layer_id in sorted(layer_data.keys()):
        overlaps = layer_data[layer_id]["overlaps"]
        print(f"  Layer {layer_id}: mean={np.mean(overlaps):.1%}, std={np.std(overlaps):.1%}, n={len(overlaps)}")


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
    analyze_adjacent_step_overlap(
        data,
        layer_id=args.layer,
        session_id=args.session,
        mode=args.mode,
        include_gaps=args.include_gaps,
        per_session=not args.no_per_session,
    )

    # # 距离分析
    # analyze_distance(data, args.layer)

    # # 跨层一致性
    # analyze_cross_layer(data)

    # # Decode 稳定性
    # analyze_decode_stability(data, args.layer)

    # 生成图表
    if args.plot:
        plot_decode_stability(data, args.output)
        plot_intra_layer_similarity(data, args.output)


if __name__ == "__main__":
    main()

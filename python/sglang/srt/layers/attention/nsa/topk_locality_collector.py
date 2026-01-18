"""
TopK Indices Collector for NSA (Native Sparse Attention)

收集 indexer 选择的 KV cache 索引的原始数据，用于局部性分析和绘图。

使用方法:
1. 设置环境变量启用收集: NSA_COLLECT_TOPK=1
2. 可选设置保存路径: NSA_TOPK_SAVE_PATH=/path/to/save
3. 运行推理
4. 数据自动保存为 .pt 文件，包含原始索引数据

数据格式:
- 按时间顺序记录每次 topk 选择
- 保留完整的 topk_indices tensor
- 可直接用于 matplotlib/seaborn 绘图
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

import torch


class TopKLocalityCollector:
    """
    收集 topk indices 原始数据（简化版，单 batch 模式）

    保存的数据结构:
    {
        "records": [  # 按时间顺序的记录（每层每次 forward 一条）
            {
                "session_id": int,                      # 会话 ID（每次 prefill 开始新会话）
                "layer_id": int,                        # 层编号
                "topk_indices": Tensor (num_tokens, topk),  # 选中的 KV block 索引
                "positions": Tensor (num_tokens,),      # 每个 query token 的位置
                "num_tokens": int,                      # 本次处理的 token 数
                "topk": int,                            # top-k 值
                "seq_len": int,                         # KV cache 总长度
                "forward_mode": str,                    # "EXTEND" (prefill) 或 "DECODE"
                "timestamp": float,
            }
        ],
        "metadata": {
            "total_records": int,
            "total_sessions": int,
            "start_time": float,
            "save_time": float,
            "duration": float,
        }
    }

    数据解读:
    - session_id: 每次 prefill (EXTEND) 开始时自增，用于区分不同 prompt
    - topk_indices[i] 是 positions[i] 位置的 query token 选择的 KV block 索引
    - Prefill: positions = [0, 1, ..., prompt_len-1], 一次处理所有 prompt tokens
    - Decode: positions = [current_pos], 每次处理一个新 token
    - seq_len 是 KV cache 总长度，即 query 可以 attend 到的范围
    """

    _instance: Optional["TopKLocalityCollector"] = None

    def __init__(self):
        self.enabled = os.getenv("NSA_COLLECT_TOPK", "1") == "1"
        self.save_path = Path(os.getenv("NSA_TOPK_SAVE_PATH", "./topk_locality_data"))
        self.max_records = int(os.getenv("NSA_MAX_RECORDS", "10000"))
        self.save_interval = int(os.getenv("NSA_SAVE_INTERVAL", "500"))
        self.log_to_file = os.getenv("NSA_LOG_TO_FILE", "1") == "1"  # 是否输出可读日志
        print(f"[TopKCollector] Initializing TopKLocalityCollector...")

        # 数据存储
        self.records: List[Dict] = []

        # 统计
        self.total_records = 0
        self.start_time = time.time()
        self.file_counter = 0

        # Session 追踪: 每次 prefill (EXTEND) 开始时递增
        self.current_session_id = 0
        self.total_sessions = 0
        self._last_mode = None  # 用于检测 prefill 开始

        # 可读日志文件
        self.log_file = None
        if self.enabled:
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"[TopKCollector] Enabled, saving to {self.save_path}")
            print(f"[TopKCollector] Max records: {self.max_records}, save interval: {self.save_interval}")
            if self.log_to_file:
                log_path = self.save_path / "topk_readable.log"
                self.log_file = open(log_path, "a")  # 追加模式
                # 写入分隔符，区分不同运行
                from datetime import datetime
                self.log_file.write(f"\n{'='*60}\n")
                self.log_file.write(f"New session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write(f"{'='*60}\n")
                self.log_file.flush()
                print(f"[TopKCollector] Readable log (append): {log_path}")

    @classmethod
    def get_instance(cls) -> "TopKLocalityCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(
        self,
        layer_id: int,
        topk_indices: torch.Tensor,
        positions: torch.Tensor,
        seq_len: int = 0,
        forward_mode: Optional["ForwardMode"] = None,
    ):
        """记录一次 topk 选择结果

        Args:
            layer_id: 层编号
            topk_indices: 选中的 KV block 索引，形状 (num_tokens, topk)
            positions: 每个 query token 的位置，形状 (num_tokens,)
                       topk_indices[i] 对应 positions[i] 这个位置的 query
            seq_len: KV cache 总长度（query 可以 attend 到的范围）
            forward_mode: EXTEND (prefill) 或 DECODE
        """
        # 在 CUDA graph capture 期间不能执行 .cpu() 操作，跳过记录
        if torch.cuda.is_current_stream_capturing():
            return

        # 获取 forward_mode 的可读名称
        mode_name = forward_mode.name if forward_mode is not None else "unknown"

        # 检测新 session: 当从 DECODE 切换到 EXTEND，或首次 EXTEND 时
        # 只在 layer_id=0 时检测，避免同一个 prefill 的多层重复触发
        if layer_id == 0 and mode_name == "EXTEND":
            if self._last_mode != "EXTEND":
                self.current_session_id += 1
                self.total_sessions += 1
                print(f"[TopKCollector] New session started: session_id={self.current_session_id}")
        if layer_id == 0:
            self._last_mode = mode_name

        # DEBUG: 打印调用信息
        print(f"[TopKCollector DEBUG] record(): session={self.current_session_id}, layer={layer_id}, mode={mode_name}, seq_len={seq_len}, num_tokens={positions.shape[0]}, total={self.total_records}")

        if not self.enabled:
            return

        if self.total_records >= self.max_records:
            if self.total_records == self.max_records:
                print(f"[TopKCollector] Reached max records {self.max_records}, saving and stopping...")
                self.save()
                self.total_records += 1  # 防止重复打印
            return

        timestamp = time.time()

        # 保存原始 tensor (clone 到 CPU)
        indices_cpu = topk_indices.detach().cpu().clone()
        positions_cpu = positions.detach().cpu().clone()

        # 保存记录
        self.records.append({
            "session_id": self.current_session_id,  # 会话 ID
            "layer_id": layer_id,
            "topk_indices": indices_cpu,           # (num_tokens, topk)
            "positions": positions_cpu,             # (num_tokens,) query 位置
            "num_tokens": indices_cpu.shape[0],
            "topk": indices_cpu.shape[1] if len(indices_cpu.shape) > 1 else 0,
            "seq_len": seq_len,                     # KV cache 总长度
            "forward_mode": mode_name,
            "timestamp": timestamp,
        })

        # 写入可读日志
        if self.log_file is not None:
            self._write_readable_log(self.current_session_id, layer_id, indices_cpu, positions_cpu, seq_len, mode_name)

        self.total_records += 1

        # 定期自动保存
        if self.total_records % self.save_interval == 0:
            print(f"[TopKCollector] Auto-saving at {self.total_records} records...")
            self.save()

    def _write_readable_log(
        self,
        session_id: int,
        layer_id: int,
        indices: torch.Tensor,
        positions: torch.Tensor,
        seq_len: int,
        mode_name: str,
    ):
        """写入人类可读的日志"""
        num_tokens = indices.shape[0]
        topk = indices.shape[1] if len(indices.shape) > 1 else 0

        # 写入头部信息（每个 forward 一行）
        self.log_file.write(f"\n[Session {session_id}][Layer {layer_id}] {mode_name} | seq_len={seq_len} | tokens={num_tokens}\n")

        # 每个 token 一行：pos -> top10 indices
        for i in range(num_tokens):
            pos = positions[i].item()
            # 取前 10 个 indices（如果不足 10 个就全部显示）
            top_indices = indices[i][:min(10, topk)].tolist()
            indices_str = ", ".join(map(str, top_indices))
            self.log_file.write(f"  pos={pos:4d} -> [{indices_str}]\n")

        self.log_file.flush()  # 立即刷新，方便实时查看

    def save(self, filename: Optional[str] = None):
        """保存收集的原始数据（单文件追加模式）"""
        if not self.enabled or len(self.records) == 0:
            return

        if filename is None:
            filename = "topk_raw_data.pt"

        save_file = self.save_path / filename

        self.file_counter += 1
        appended_count = len(self.records)

        # 如果文件已存在，加载并合并
        if save_file.exists():
            existing = torch.load(save_file, weights_only=False)
            existing["records"].extend(self.records)
            existing["metadata"]["total_records"] = len(existing["records"])
            # 统计总 session 数
            all_sessions = set(r.get("session_id", 0) for r in existing["records"])
            existing["metadata"]["total_sessions"] = len(all_sessions)
            existing["metadata"]["save_time"] = time.time()
            existing["metadata"]["duration"] = time.time() - existing["metadata"]["start_time"]
            data = existing
            print(f"[TopKCollector] Appended {appended_count} records, total: {len(existing['records'])} ({len(all_sessions)} sessions)")
        else:
            data = {
                "records": self.records,
                "metadata": {
                    "total_records": len(self.records),
                    "total_sessions": self.total_sessions,
                    "start_time": self.start_time,
                    "save_time": time.time(),
                    "duration": time.time() - self.start_time,
                }
            }
            print(f"[TopKCollector] Saved {len(self.records)} records ({self.total_sessions} sessions) to {save_file}")

        torch.save(data, save_file)

        # 保存一份轻量级的索引信息
        self._save_index_file(save_file, data, appended_count)

        # 清空内存中的数据
        self.clear()

    def _save_index_file(self, main_file: Path, data: Dict, appended_count: int = 0):
        """保存索引文件，记录每次保存的历史"""
        index_file = main_file.with_suffix(".index.txt")

        records = data["records"]
        metadata = data["metadata"]

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 统计每层的记录数
        layer_counts = {}
        for record in records:
            layer_id = record["layer_id"]
            layer_counts[layer_id] = layer_counts.get(layer_id, 0) + 1

        # 统计 forward_mode
        mode_counts = {}
        for record in records:
            mode = record["forward_mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # 统计 session
        session_ids = set(r.get("session_id", 0) for r in records)

        # 构建本次保存的记录
        lines = [
            f"",
            f"{'=' * 50}",
            f"Save #{self.file_counter} ({timestamp})",
            f"{'=' * 50}",
            f"Main file: {main_file.name}",
            f"Total records: {len(records)}" + (f" (appended {appended_count})" if appended_count > 0 else ""),
            f"Total sessions: {len(session_ids)} (session_ids: {sorted(session_ids)})",
            f"Duration: {metadata['duration']:.2f}s",
            f"",
            f"Per-layer counts:",
        ]

        for layer_id in sorted(layer_counts.keys()):
            lines.append(f"  Layer {layer_id}: {layer_counts[layer_id]} records")

        lines.append(f"")
        lines.append(f"Per-mode counts:")
        for mode, count in sorted(mode_counts.items()):
            lines.append(f"  {mode}: {count} records")

        # 追加到文件
        with open(index_file, "a") as f:
            f.write("\n".join(lines) + "\n")

    def clear(self, reset_counter: bool = False):
        """清空内存中的数据"""
        self.records.clear()
        if reset_counter:
            self.total_records = 0

    def close(self):
        """关闭日志文件"""
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
            print(f"[TopKCollector] Log file closed")

    def __del__(self):
        """析构时关闭文件"""
        self.close()


# 全局访问函数
def get_collector() -> TopKLocalityCollector:
    return TopKLocalityCollector.get_instance()

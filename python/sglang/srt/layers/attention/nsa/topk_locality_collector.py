"""
TopK Indices Collector for NSA (Native Sparse Attention)

收集 indexer 选择的 KV cache 索引的原始数据，用于局部性分析和绘图。

使用方法:
1. 设置环境变量启用收集: NSA_COLLECT_TOPK=1
2. 可选设置保存路径: NSA_TOPK_SAVE_PATH=/path/to/save
3. 运行推理
4. 数据自动保存为 .pt 文件，包含原始索引数据

数据格式:
- 按 request -> layer -> token 组织
- 保留完整的 topk_indices tensor
- 可直接用于 matplotlib/seaborn 绘图
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

import torch


class TopKLocalityCollector:
    """
    收集 topk indices 原始数据

    保存的数据结构:
    {
        "requests": {
            request_id: {
                "layers": {
                    layer_id: {
                        "topk_indices": Tensor (num_tokens, topk),
                        "seq_lens": List[int],
                        "forward_mode": str,
                        "timestamp": float,
                    }
                },
                "metadata": {...}
            }
        },
        "raw_records": [  # 按时间顺序的原始记录，方便时序分析
            {
                "request_id": str,
                "layer_id": int,
                "step": int,  # decode step 序号
                "topk_indices": Tensor,
                ...
            }
        ]
    }
    """

    _instance: Optional["TopKLocalityCollector"] = None

    def __init__(self):
        self.enabled = os.getenv("NSA_COLLECT_TOPK", "1") == "1"
        self.save_path = Path(os.getenv("NSA_TOPK_SAVE_PATH", "./topk_locality_data"))
        self.max_records = int(os.getenv("NSA_MAX_RECORDS", "10000"))
        self.save_interval = int(os.getenv("NSA_SAVE_INTERVAL", "500"))
        print(f"[TopKCollector] Initializing TopKLocalityCollector...")

        # 数据存储 - 保存原始 tensor
        self.requests: Dict[str, Dict] = {}  # request_id -> layer data
        self.raw_records: List[Dict] = []  # 按时间顺序的所有记录
        self.decode_steps: Dict[str, int] = defaultdict(int)  # request_id -> step counter

        # 统计
        self.total_records = 0
        self.start_time = time.time()
        self.file_counter = 0

        if self.enabled:
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"[TopKCollector] Enabled, saving to {self.save_path}")
            print(f"[TopKCollector] Max records: {self.max_records}, save interval: {self.save_interval}")

    @classmethod
    def get_instance(cls) -> "TopKLocalityCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(
        self,
        request_id: str,
        layer_id: int,
        topk_indices: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
        forward_mode: Optional["ForwardMode"] = None,
    ):
        """记录一次 topk 选择结果，保存原始索引数据"""
        # 在 CUDA graph capture 期间不能执行 .cpu() 操作，跳过记录
        if torch.cuda.is_current_stream_capturing():
            return

        # 获取 forward_mode 的可读名称
        mode_name = forward_mode.name if forward_mode is not None else "unknown"

        # DEBUG: 打印调用信息
        print(f"[TopKCollector DEBUG] record() called: layer={layer_id}, mode={mode_name}, seq_lens={seq_lens}, total_records={self.total_records}, enabled={self.enabled}")

        if not self.enabled:
            print(f"[TopKCollector DEBUG] skipped: not enabled")
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

        # 判断是否是 decode 模式
        is_decode = forward_mode is not None and forward_mode.is_decode()
        step = -1
        if is_decode:
            step = self.decode_steps[request_id]
            self.decode_steps[request_id] += 1

        # 按 request/layer 组织的数据
        if request_id not in self.requests:
            self.requests[request_id] = {
                "layers": {},
                "metadata": {
                    "first_seen": timestamp,
                    "forward_mode": mode_name,
                }
            }

        # 保存到 layer 字典 (会覆盖同一 layer 的旧数据，只保留最新)
        self.requests[request_id]["layers"][layer_id] = {
            "topk_indices": indices_cpu,
            "seq_lens": seq_lens or [],
            "num_tokens": indices_cpu.shape[0],
            "topk": indices_cpu.shape[1] if len(indices_cpu.shape) > 1 else 0,
            "timestamp": timestamp,
            "step": step,
        }

        # 同时保存到原始记录列表 (保留所有历史，用于时序分析)
        self.raw_records.append({
            "request_id": request_id,
            "layer_id": layer_id,
            "step": step,
            "topk_indices": indices_cpu,
            "seq_lens": seq_lens or [],
            "forward_mode": mode_name,
            "timestamp": timestamp,
        })

        self.total_records += 1

        # 定期自动保存
        if self.total_records % self.save_interval == 0:
            print(f"[TopKCollector] Auto-saving at {self.total_records} records...")
            self.save()

    def save(self, filename: Optional[str] = None):
        """保存收集的原始数据（单文件追加模式）"""
        if not self.enabled or len(self.raw_records) == 0:
            return

        if filename is None:
            filename = "topk_raw_data.pt"

        save_file = self.save_path / filename

        self.file_counter += 1
        appended_count = len(self.raw_records)

        # 如果文件已存在，加载并合并
        if save_file.exists():
            existing = torch.load(save_file, weights_only=False)
            existing["requests"].update(self.requests)
            existing["raw_records"].extend(self.raw_records)
            existing["metadata"]["total_records"] = len(existing["raw_records"])
            existing["metadata"]["num_requests"] = len(existing["requests"])
            existing["metadata"]["save_time"] = time.time()
            existing["metadata"]["duration"] = time.time() - existing["metadata"]["start_time"]
            data = existing
            print(f"[TopKCollector] Appended {appended_count} records, total: {len(existing['raw_records'])}")
        else:
            data = {
                "requests": self.requests,
                "raw_records": self.raw_records,
                "metadata": {
                    "total_records": len(self.raw_records),
                    "num_requests": len(self.requests),
                    "start_time": self.start_time,
                    "save_time": time.time(),
                    "duration": time.time() - self.start_time,
                }
            }
            appended_count = 0  # 首次保存，不是追加
            print(f"[TopKCollector] Saved {len(self.raw_records)} records to {save_file}")

        torch.save(data, save_file)

        # 保存一份轻量级的索引信息 (用于快速预览)
        self._save_index_file(save_file, data, appended_count)

        # 清空内存中的数据
        self.clear()

    def _save_index_file(self, main_file: Path, data: Dict, appended_count: int = 0):
        """保存索引文件，记录每次保存的历史"""
        index_file = main_file.with_suffix(".index.txt")

        requests = data["requests"]
        raw_records = data["raw_records"]
        metadata = data["metadata"]

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建本次保存的记录
        lines = [
            f"",
            f"{'=' * 50}",
            f"Save #{self.file_counter} ({timestamp})",
            f"{'=' * 50}",
            f"Main file: {main_file.name}",
            f"Total records: {len(raw_records)}" + (f" (appended {appended_count})" if appended_count > 0 else ""),
            f"Num requests: {len(requests)}",
            f"Duration: {metadata['duration']:.2f}s",
            f"",
            f"Requests:",
        ]

        for req_id, req_data in requests.items():
            layers = sorted(req_data["layers"].keys())
            num_layers = len(layers)
            sample_layer = req_data["layers"][layers[0]] if layers else {}
            num_tokens = sample_layer.get("num_tokens", 0)
            topk = sample_layer.get("topk", 0)

            lines.append(f"  {req_id}: {num_layers} layers, {num_tokens} tokens, topk={topk}")

        # 追加到文件
        with open(index_file, "a") as f:
            f.write("\n".join(lines) + "\n")

    def get_layer_indices(self, request_id: str, layer_id: int) -> Optional[torch.Tensor]:
        """获取指定 request 和 layer 的 topk indices"""
        if request_id in self.requests:
            layers = self.requests[request_id]["layers"]
            if layer_id in layers:
                return layers[layer_id]["topk_indices"]
        return None

    def get_all_layers_indices(self, request_id: str) -> Dict[int, torch.Tensor]:
        """获取指定 request 所有层的 topk indices"""
        if request_id not in self.requests:
            return {}
        return {
            layer_id: data["topk_indices"]
            for layer_id, data in self.requests[request_id]["layers"].items()
        }

    def clear(self, reset_counter: bool = False):
        """清空内存中的数据（不重置 total_records 计数器，除非显式指定）"""
        self.requests.clear()
        self.raw_records.clear()
        self.decode_steps.clear()
        if reset_counter:
            self.total_records = 0


# 全局访问函数
def get_collector() -> TopKLocalityCollector:
    return TopKLocalityCollector.get_instance()


def record_topk(
    request_id: str,
    layer_id: int,
    topk_indices: torch.Tensor,
    seq_lens: Optional[List[int]] = None,
    forward_mode: Optional["ForwardMode"] = None,
):
    """便捷函数：记录 topk 选择结果"""
    get_collector().record(
        request_id=request_id,
        layer_id=layer_id,
        topk_indices=topk_indices,
        seq_lens=seq_lens,
        forward_mode=forward_mode,
    )

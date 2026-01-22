import requests
import time
import threading
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable


class Analyzer:
    """每个 analyzer 返回 (notes, derived_fields)."""
    name: str = "base"

    def analyze(self, current: Dict[str, float], deltas: Dict[str, float]) -> (List[str], Dict[str, Any]):
        return [], {}


class TokenRateAnalyzer(Analyzer):
    """
    关注 token 增量 & token/s。
    注意:token_total 的 delta 本身不是“命中/不命中”,只是 workload 的变化。
    """
    name = "token_rate"

    def __init__(self, interval_s: float):
        self.interval_s = max(interval_s, 1e-6)

    def analyze(self, current, deltas):
        notes = []
        derived = {}

        dp = float(deltas.get("prompt", 0.0))
        dg = float(deltas.get("gen", 0.0))

        derived["prompt_toks_delta"] = dp
        derived["gen_toks_delta"] = dg
        derived["prompt_toks_per_s"] = dp / self.interval_s
        derived["gen_toks_per_s"] = dg / self.interval_s

        if dg > 0 and dp == 0:
            notes.append("🧠 gen_only")
        if dp > 0 and dg == 0:
            notes.append("📥 prefill_only")

        return notes, derived


class QueueAnalyzer(Analyzer):
    """关注 waiting / running."""
    name = "queue"

    def analyze(self, current, deltas):
        notes = []
        derived = {}

        waiting = int(current.get("waiting", 0))
        running = int(current.get("running", 0))

        derived["waiting"] = waiting
        derived["running"] = running

        if waiting > 0 and running > 0:
            notes.append("⚠️ queueing")
        elif waiting > 0 and running == 0:
            notes.append("⏳ backlog_no_runner")

        return notes, derived


class GpuCacheAnalyzer(Analyzer):
    """
    关注 gpu_cache_usage_perc(通常 0~1)。
    """
    name = "gpu_cache"

    def __init__(self, high: float = 0.90, warn: float = 0.80):
        self.warn = warn
        self.high = high

    def analyze(self, current, deltas):
        notes = []
        derived = {}

        usage = float(current.get("gpu_cache", 0.0))
        derived["gpu_cache_usage"] = usage

        if usage >= self.high:
            notes.append("🔥 kv_cache_high")
        elif usage >= self.warn:
            notes.append("🟠 kv_cache_warn")

        return notes, derived


class CacheHintAnalyzer(Analyzer):
    """
    只给出 “可能命中/可能重算” 的 hint,
    并把阈值做成可配置 + 输出到 CSV 便于后验校准。
    """
    name = "cache_hint"

    def __init__(self, hit_threshold: float = 50.0, recompute_threshold: float = 500.0):
        self.hit_threshold = hit_threshold
        self.recompute_threshold = recompute_threshold

    def analyze(self, current, deltas):
        notes = []
        derived = {}

        dp = float(deltas.get("prompt", 0.0))

        # 输出阈值,便于 CSV 后处理/回归校准
        derived["cache_hit_threshold"] = self.hit_threshold
        derived["cache_recompute_threshold"] = self.recompute_threshold

        # 只在确实发生了 prefill 时才谈“hint”
        if dp > 0:
            if dp <= self.hit_threshold:
                notes.append("✅ cache_hint_hit")
                derived["cache_hint"] = "hit"
            elif dp >= self.recompute_threshold:
                notes.append("🧊 cache_hint_recompute")
                derived["cache_hint"] = "recompute"
            else:
                derived["cache_hint"] = "unknown"
        else:
            derived["cache_hint"] = "n/a"

        return notes, derived


# ----------------------------
# 主监控类
# ----------------------------

@dataclass
class MetricSpec:
    prom_name: str
    reduce: str = "sum"  # "sum" | "max" | "last"


class VLLMMonitor:
    def __init__(
        self,
        url: str = "http://localhost:8000/metrics",
        interval: float = 0.5,
        csv_path: str = "vllm_benchmark.csv",
        flush_every: int = 1,
    ):
        self.url = url
        self.interval = interval
        self.running = False

        # 状态追踪
        self.last_prompt_tokens: Optional[float] = None
        self.last_gen_tokens: Optional[float] = None

        # Prometheus 指标定义(不同 vLLM 版本可能名字不同,改这里)
        self.METRICS: Dict[str, MetricSpec] = {
            "prompt_total": MetricSpec("vllm:prompt_tokens_total", reduce="sum"),
            "gen_total": MetricSpec("vllm:generation_tokens_total", reduce="sum"),
            "running": MetricSpec("vllm:num_requests_running", reduce="last"),
            "waiting": MetricSpec("vllm:num_requests_waiting", reduce="last"),
            "gpu_cache": MetricSpec("vllm:kv_cache_usage_perc", reduce="last"),
        }

        # analyzers:可以自由增删/调整顺序
        self.analyzers: List[Analyzer] = [
            TokenRateAnalyzer(interval_s=self.interval),
            QueueAnalyzer(),
            GpuCacheAnalyzer(warn=0.80, high=0.90),
            CacheHintAnalyzer(hit_threshold=50.0, recompute_threshold=500.0),
        ]

        # CSV 实时写入
        self.csv_path = csv_path
        self.flush_every = max(int(flush_every), 1)
        self._csv_f = None
        self._csv_writer = None
        self._csv_header_written = False
        self._rows_since_flush = 0

        # 记录字段模板(第一次运行后动态扩展)
        self._base_fields = ["time"]
        self._last_fieldnames: Optional[List[str]] = None

    # ---------- lifecycle ----------

    def start(self):
        if self.running:
            return
        self.running = True
        self._open_csv()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"🚀 vLLM 监控已启动 (freq={self.interval}s, csv={self.csv_path})")

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)
        self._close_csv()
        print("🛑 vLLM 监控已停止")

    # ---------- core loop ----------

    def _loop(self):
        while self.running:
            try:
                self._fetch_process_record()
            except requests.exceptions.ConnectionError:
                print(f"⚠️ 无法连接 vLLM ({self.url}),请检查服务是否启动...")
            except Exception as e:
                print(f"Monitor Error: {e}")
            time.sleep(self.interval)

    def _fetch_process_record(self):
        raw = self._get_raw_data()
        if not raw:
            return

        current = self._parse_prometheus(raw)
        deltas = self._compute_deltas(current)
        if deltas is None:
            # 初始化阶段:不输出/不写 CSV(避免第一行 delta 巨大/无意义)
            return

        notes, derived = self._run_analyzers(current, deltas)
        self._record_entry(current, deltas, notes, derived)

    # ---------- IO & parsing ----------

    def _get_raw_data(self) -> Optional[str]:
        resp = requests.get(self.url, timeout=2)
        if resp.status_code != 200:
            return None
        return resp.text

    def _parse_prometheus(self, text: str) -> Dict[str, float]:
        """
        Prometheus exposition 格式:
          metric_name{label="x"} 123
          metric_name 456
        同名多行时按 reduce 聚合。
        """
        # 收集每个 prom_name 的所有 value
        bucket: Dict[str, List[float]] = {spec.prom_name: [] for spec in self.METRICS.values()}

        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            # 以空格分隔:左边是 name{labels} 或 name,右边是 value
            try:
                left, value_str = line.rsplit(" ", 1)
                value = float(value_str)
            except Exception:
                continue

            # 去掉 labels,只保留 name
            name = left.split("{", 1)[0].strip()

            # 只收集我们关心的
            for spec in self.METRICS.values():
                if name == spec.prom_name:
                    bucket[spec.prom_name].append(value)

        parsed: Dict[str, float] = {}
        for key, spec in self.METRICS.items():
            values = bucket.get(spec.prom_name, [])
            if not values:
                parsed[key] = 0.0
                continue
            if spec.reduce == "max":
                parsed[key] = max(values)
            elif spec.reduce == "last":
                parsed[key] = values[-1]
            else:  # sum
                parsed[key] = sum(values)

        return parsed

    # ---------- state/delta ----------

    def _compute_deltas(self, current: Dict[str, float]) -> Optional[Dict[str, float]]:
        pt = float(current.get("prompt_total", 0.0))
        gt = float(current.get("gen_total", 0.0))

        if self.last_prompt_tokens is None:
            self.last_prompt_tokens = pt
            self.last_gen_tokens = gt
            return None

        deltas = {
            "prompt": pt - float(self.last_prompt_tokens),
            "gen": gt - float(self.last_gen_tokens),
        }

        self.last_prompt_tokens = pt
        self.last_gen_tokens = gt
        return deltas

    # ---------- analysis ----------

    def _run_analyzers(self, current, deltas) -> (List[str], Dict[str, Any]):
        notes_all: List[str] = []
        derived_all: Dict[str, Any] = {}

        for a in self.analyzers:
            notes, derived = a.analyze(current, deltas)
            if notes:
                notes_all.extend(notes)
            if derived:
                # analyzer 字段名冲突时,后者覆盖前者(也可以改成报错)
                derived_all.update(derived)

        return notes_all, derived_all

    # ---------- record & csv ----------

    def _record_entry(self, current, deltas, notes: List[str], derived: Dict[str, Any]):
        now = datetime.now().strftime("%H:%M:%S")

        # 1) 统一打印(尽量稳定、简洁)
        gpu_pct = float(current.get("gpu_cache", 0.0)) * 100.0
        log = (
            f"[{now}] "
            f"ΔP:{int(deltas['prompt'])} | ΔG:{int(deltas['gen'])} | "
            f"Run:{int(current.get('running', 0))} Wait:{int(current.get('waiting', 0))} | "
            f"KV:{gpu_pct:.1f}%"
        )
        if notes:
            log += " | " + " ".join(notes)
        print(log)

        # 2) 行数据(CSV:建议“原始 + delta + derived + notes”)
        row: Dict[str, Any] = {}
        row["time"] = now

        # 原始值
        row.update({
            "prompt_total": float(current.get("prompt_total", 0.0)),
            "gen_total": float(current.get("gen_total", 0.0)),
            "running_raw": float(current.get("running", 0.0)),
            "waiting_raw": float(current.get("waiting", 0.0)),
            "gpu_cache_raw": float(current.get("gpu_cache", 0.0)),
        })

        # delta
        row.update({
            "delta_prompt": float(deltas.get("prompt", 0.0)),
            "delta_gen": float(deltas.get("gen", 0.0)),
        })

        # derived
        row.update(derived or {})

        # notes 串
        row["notes"] = " ".join(notes) if notes else ""

        self._append_csv_row(row)

    def _open_csv(self):
        # 确保目录存在
        csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
        os.makedirs(csv_dir, exist_ok=True)

        self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
        # writer 需要 fieldnames,首次写入时再确定
        self._csv_writer = None
        self._csv_header_written = os.path.getsize(self.csv_path) > 0
        self._rows_since_flush = 0

    def _close_csv(self):
        if self._csv_f:
            try:
                self._csv_f.flush()
            except Exception:
                pass
            try:
                self._csv_f.close()
            except Exception:
                pass
        self._csv_f = None
        self._csv_writer = None

    def _append_csv_row(self, row: Dict[str, Any]):
        """
        关键点:字段名可能随着 analyzer 增加而扩展。
        这里做一个“动态表头”策略:
        - 如果当前 row 有新字段 -> 重新写一个新文件最标准
          但实现复杂。
        - 这里采取“保守策略”:fieldnames = 历史字段 ∪ 当前字段(排序后稳定)
          若发现新字段,会在后续行包含这些字段(旧行自然为空)。
        """
        if not self._csv_f:
            return

        # 计算 fieldnames(稳定排序:time 放最前,其余按字母)
        keys = list(row.keys())
        if self._last_fieldnames is None:
            fieldnames = ["time"] + sorted([k for k in keys if k != "time"])
        else:
            merged = set(self._last_fieldnames) | set(keys)
            merged.discard("time")
            fieldnames = ["time"] + sorted(list(merged))

        # 若 fieldnames 变化,需要重新构建 writer
        if self._csv_writer is None or fieldnames != self._last_fieldnames:
            self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=fieldnames)
            self._last_fieldnames = fieldnames
            if not self._csv_header_written:
                self._csv_writer.writeheader()
                self._csv_header_written = True
            else:
                # 已有旧文件但表头可能不同:这里不重写表头,避免破坏文件结构
                #(如果希望严格一致,建议“新字段出现就新起一个文件”)
                pass

        self._csv_writer.writerow(row)
        self._rows_since_flush += 1

        if self._rows_since_flush >= self.flush_every:
            try:
                self._csv_f.flush()
            except Exception:
                pass
            self._rows_since_flush = 0


if __name__ == "__main__":
    mon = VLLMMonitor(
        url="http://localhost:8000/metrics",
        interval=0.5,
        csv_path="vllm_benchmark.csv",
        flush_every=1,
    )
    mon.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mon.stop()

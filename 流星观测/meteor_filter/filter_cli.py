# -*- coding: utf-8 -*-
"""流星图片批量筛选命令行工具。

一次性批量筛选：
    python filter_cli.py --input D:\\captures\\20261023_night --mode copy

持续监听目录、自动筛选新图片（接入流水线推荐）：
    python filter_cli.py --input D:\\captures\\live --mode move --watch --interval 15
    # 新图片落盘后自动判定：流星图移入 <input>_meteor/，其余移入 <input>_noise/
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from meteor_detector import MeteorDetector
from batch_filter import (DetectorPool, auto_workers, build_pool,
                          BatchResult, archive_file, classify_one,
                          iter_images, result_to_json, run_batch,
                          save_annotated_image, wait_file_stable)

log = logging.getLogger("filter_cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="流星图片自动筛选（MetDetPy YOLOv5s）")
    p.add_argument("--input", "-i", required=True, help="图片目录")
    p.add_argument("--mode", default="copy", choices=["report", "copy", "move"],
                   help="report=只报告；copy=复制归档；move=移动归档（默认 copy）")
    p.add_argument("--meteor-dir", default=None, help="流星图归档目录（默认 <input>_meteor）")
    p.add_argument("--noise-dir", default=None, help="非流星图归档目录（默认 <input>_noise）")
    p.add_argument("--threshold", "-t", type=float, default=0.5,
                   help="正类判定阈值 0~1（默认 0.5，误检多则调高，漏检多则调低）")
    p.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    p.add_argument("--save-annotated", action="store_true", help="额外保存画框图便于人工抽查")
    p.add_argument("--watch", action="store_true",
                   help="持续监听模式：周期扫描目录，自动处理新出现的图片")
    p.add_argument("--interval", type=float, default=10, help="监听扫描间隔秒数（默认 10）")
    p.add_argument("--multiscale", type=int, default=2, choices=[0, 1, 2],
                   help="多尺度档位：0=最快（整图一次），2=最准（默认，小流星召回更好）")
    p.add_argument("--workers", type=int, default=0,
                   help="并行推理 worker 数（0=按 CPU 核数自动，1=单线程；"
                        "多 worker 会各建一个 ONNX session 提高吞吐）")
    return p


def print_progress(done: int, total: int, r: BatchResult):
    flag = "流星" if r.has_meteor else "----"
    mark = "✔" if r.has_meteor else "·"
    err = f"  [错误: {r.error}]" if r.error else ""
    print(f"[{done}/{total}] {mark} {flag} {r.meteor_score:.2f}  {r.path.name}{err}",
          flush=True)


def run_once(detector: MeteorDetector, args, factory) -> None:
    input_dir = Path(args.input).resolve()
    csv_path = input_dir.parent / f"filter_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
    workers = args.workers if args.workers > 0 else auto_workers()
    print(f"开始筛选: {input_dir}  模式={args.mode}  阈值={args.threshold}  "
          f"workers={workers}")
    out = run_batch(detector, input_dir, mode=args.mode,
                    meteor_dir=args.meteor_dir, noise_dir=args.noise_dir,
                    threshold=args.threshold, recursive=not args.no_recursive,
                    save_annotated=args.save_annotated,
                    csv_path=csv_path, progress=print_progress,
                    workers=workers, detector_factory=factory)
    s = out["summary"]
    print("-" * 60)
    print(f"完成：共 {s['total']} 张 | 流星 {s['meteor']} | 非流星 {s['noise']} | "
          f"错误 {s['error']} | 耗时 {s['elapsed_sec']}s")
    print(f"流星目录: {s['meteor_dir'] or '（仅报告模式）'}")
    print(f"CSV 报告: {s.get('csv_report', '')}")


def run_watch(detector: MeteorDetector, args, factory) -> None:
    """持续监听：周期扫描 --input 目录，自动处理新落盘的图片并归档。"""
    input_dir = Path(args.input).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    meteor_dir = Path(args.meteor_dir).resolve() if args.meteor_dir \
        else input_dir.parent / f"{input_dir.name}_meteor"
    noise_dir = Path(args.noise_dir).resolve() if args.noise_dir \
        else input_dir.parent / f"{input_dir.name}_noise"
    mode = args.mode if args.mode != "report" else "move"  # 监听模式下归档才有意义
    workers = args.workers if args.workers > 0 else auto_workers()
    pool = build_pool(detector, workers, factory)
    annotated_dir = meteor_dir.parent / "annotated"
    max_retries = 3  # 解码失败的文件最多重试次数（可能是写入中被误判为稳定）
    print(f"监听模式: 每 {args.interval}s 扫描 {input_dir}（workers={pool.size}）")
    print(f"流星图 → {meteor_dir}")
    print(f"非流星图 → {noise_dir}   (Ctrl+C 退出)")
    seen: set[Path] = set(iter_images(input_dir, not args.no_recursive))
    if seen:
        print(f"跳过已存在的 {len(seen)} 张旧图片（只处理新落盘文件）")

    def _classify(path: Path) -> BatchResult:
        with pool.borrow() as det:
            return classify_one(det, path, args.threshold)

    retries: dict[Path, int] = {}
    executor = ThreadPoolExecutor(max_workers=pool.size)
    try:
        while True:
            time.sleep(args.interval)
            # 先等文件写稳，再标记为已处理；仍在写入的留给下一轮
            new_files = []
            for path in iter_images(input_dir, not args.no_recursive):
                if path in seen:
                    continue
                if not wait_file_stable(path):
                    continue
                seen.add(path)
                new_files.append(path)
            for r in executor.map(_classify, new_files):
                # 写入中途被误判为稳定的文件会解码失败：撤销 seen 等下轮重试
                if r.action == "error" and "无法读取" in r.error \
                        and retries.get(r.path, 0) < max_retries:
                    retries[r.path] = retries.get(r.path, 0) + 1
                    seen.discard(r.path)
                    print(f"[重试 {retries[r.path]}/{max_retries}] {r.path.name}: "
                          f"{r.error}", flush=True)
                    continue
                if r.action == "error":
                    print(f"[错误] {r.path.name}: {r.error}", flush=True)
                    continue
                archive_file(r, mode, meteor_dir, noise_dir, input_dir)
                if args.save_annotated:
                    save_annotated_image(r, annotated_dir, input_dir)
                r.img = None
                mark = "✔流星" if r.has_meteor else "·无"
                print(f"{datetime.now():%H:%M:%S} {mark} {r.meteor_score:.2f} "
                      f"{r.path.name} → {r.action}", flush=True)
    except KeyboardInterrupt:
        executor.shutdown(wait=False)
        print("\n已停止监听。")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    workers = args.workers if args.workers > 0 else auto_workers()
    threads_per = max(1, (os.cpu_count() or 4) // workers) if workers > 1 else 0

    def factory() -> MeteorDetector:
        return MeteorDetector(multiscale_pred=args.multiscale,
                              num_threads=threads_per)

    detector = factory()
    if args.watch:
        run_watch(detector, args, factory)
    else:
        run_once(detector, args, factory)


if __name__ == "__main__":
    main()

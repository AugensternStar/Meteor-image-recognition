# -*- coding: utf-8 -*-
"""批量筛选逻辑：扫描目录 → 逐图判定 → 归档（复制/移动/仅报告）。

供 service.py（HTTP API）和 filter_cli.py（命令行）共用。
支持多 ONNX session 并行推理（workers > 1 时吞吐量随 CPU 核数提升）。
"""
from __future__ import annotations

import csv
import logging
import os
import queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from meteor_detector import MeteorDetector, draw_detections, imread_unicode, imwrite_unicode
import cv2

logger = logging.getLogger("batch_filter")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# 输出目录名，扫描时跳过，避免重复处理
OUTPUT_DIR_NAMES = {"meteor", "noise", "review", "annotated"}


def auto_workers() -> int:
    """按 CPU 核数推荐并行 worker 数（每个 worker 独占一个 ONNX session）。"""
    return max(1, min(8, (os.cpu_count() or 4) // 4))


class DetectorPool:
    """线程安全的检测器借用池：一个 session 同一时间只被一个线程使用。"""

    def __init__(self, detectors: Iterable[MeteorDetector]):
        self.detectors = list(detectors)
        if not self.detectors:
            raise ValueError("DetectorPool 需要至少一个检测器")
        self._idle: queue.Queue = queue.Queue()
        for d in self.detectors:
            self._idle.put(d)

    @property
    def size(self) -> int:
        return len(self.detectors)

    @contextmanager
    def borrow(self):
        det = self._idle.get()
        try:
            yield det
        finally:
            self._idle.put(det)


def build_pool(detector: MeteorDetector, workers: int,
               factory: Callable[[], MeteorDetector] | None) -> DetectorPool:
    """以 detector 为首实例构建池；workers>1 且给出 factory 时创建额外 session。"""
    detectors = [detector]
    if workers > 1 and factory is not None:
        detectors.extend(factory() for _ in range(workers - 1))
    return DetectorPool(detectors)


def iter_images(input_dir: str | Path, recursive: bool = True) -> list[Path]:
    """列出目录下所有图片文件（跳过输出子目录）。"""
    input_dir = Path(input_dir)
    it = input_dir.rglob("*") if recursive else input_dir.glob("*")
    files = []
    for p in it:
        if not p.is_file():
            continue
        rel_parts = p.relative_to(input_dir).parts
        if any(part in OUTPUT_DIR_NAMES for part in rel_parts[:-1]):
            continue  # 跳过输出目录里的文件
        if p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def wait_file_stable(path: Path, wait_sec: float = 1.0, timeout: float = 30.0) -> bool:
    """等待文件写入完成（体积在 wait_sec 内不变）。用于监听正在生成的文件。"""
    last_size, t0 = -1, time.time()
    while time.time() - t0 < timeout:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last_size and size > 0:
            return True
        last_size = size
        time.sleep(wait_sec)
    return False


@dataclass
class BatchResult:
    path: Path
    has_meteor: bool = False
    meteor_score: float = 0.0
    top_class: str = ""
    top_score: float = 0.0
    n_detections: int = 0
    action: str = "none"  # copied/moved/kept/error
    error: str = ""
    detections: list = field(default_factory=list)
    binned: str = ""      # 归档位置: meteor / review / noise（report 模式为空）
    img: Optional[object] = field(default=None, repr=False, compare=False)
    # img: 解码后的图像，供归档/画框复用，避免二次读盘；用完置 None 释放内存


def classify_one(detector: MeteorDetector, path: Path, threshold: float) -> BatchResult:
    """处理单张图片，返回判定结果（不做文件操作）。"""
    r = BatchResult(path=path)
    img = imread_unicode(path)
    if img is None:
        r.action, r.error = "error", "无法读取图片"
        return r
    res = detector.filter_image(img, threshold)
    r.has_meteor = res["has_meteor"]
    r.meteor_score = res["meteor_score"]
    r.n_detections = len(res["detections"])
    r.detections = res["detections"]
    r.img = img
    if res["detections"]:
        top = max(res["detections"], key=lambda d: d["score"])
        r.top_class, r.top_score = top["class_name"], top["score"]
    return r


def archive_file(r: BatchResult, mode: str, meteor_dir: Optional[Path],
                 noise_dir: Optional[Path], input_dir: Path,
                 review_dir: Optional[Path] = None,
                 review_min: float = 0.0) -> None:
    """按判定结果复制/移动文件到对应目录，保持相对子路径。

    三档分流：has_meteor → meteor_dir；
    review_min>0 且置信度落在 [review_min, threshold) → review_dir（待人工复核）；
    其余 → noise_dir。
    """
    if r.action == "error":
        return  # 读取失败的文件不归档
    if mode not in ("copy", "move"):
        r.action = "kept"
        return
    if r.has_meteor:
        dst_root, r.binned = meteor_dir, "meteor"
    elif review_min > 0 and r.meteor_score >= review_min and review_dir is not None:
        dst_root, r.binned = review_dir, "review"
    else:
        dst_root, r.binned = noise_dir, "noise"
    if dst_root is None:
        r.action, r.binned = "kept", ""
        return
    rel = r.path.relative_to(input_dir)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        stem, ext = dst.stem, dst.suffix
        dst = dst.with_name(f"{stem}_{int(time.time())}{ext}")
    try:
        if mode == "copy":
            shutil.copy2(r.path, dst)
            r.action = "copied"
        else:
            shutil.move(str(r.path), dst)
            r.action = "moved"
    except OSError as e:
        r.action, r.error, r.binned = "error", str(e), ""


def save_annotated_image(r: BatchResult, annotated_dir: Path, input_dir: Path) -> None:
    """把带检测框的图片另存，便于人工抽查（复用已解码图像，兼容 move 后源文件已移走）。"""
    try:
        img = r.img if r.img is not None else imread_unicode(r.path)
        if img is None:
            return
        from meteor_detector import Detection
        dets = [Detection(d["box"][0], d["box"][1], d["box"][2], d["box"][3],
                          d["score"], d["class_id"], d["class_name"])
                for d in r.detections]
        rel = r.path.relative_to(input_dir)
        dst = annotated_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(dst, draw_detections(img, dets))
    except Exception as e:  # 标注图保存失败不影响主流程
        logger.warning("保存标注图失败 %s: %s", r.path, e)


CSV_HEADER = ["文件", "有流星", "流星置信度", "最高分类别", "最高分",
              "检测目标数", "归档位置", "处理动作", "错误信息"]


def csv_row(r: BatchResult) -> list:
    return [str(r.path), r.has_meteor, f"{r.meteor_score:.4f}",
            r.top_class, f"{r.top_score:.4f}", r.n_detections,
            r.binned or ("meteor" if r.has_meteor else "noise"),
            r.action, r.error]


class CsvReportWriter:
    """增量 CSV 写入：每张图追加一行，避免整体重写的 O(n²) I/O，中断也留记录。"""

    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", newline="", encoding="utf-8-sig")
        self._w = csv.writer(self._f)
        self._w.writerow(CSV_HEADER)
        self._f.flush()

    def write(self, r: BatchResult) -> None:
        self._w.writerow(csv_row(r))
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def write_csv_report(results: list[BatchResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in results:
            w.writerow(csv_row(r))


def run_batch(
    detector: MeteorDetector,
    input_dir: str | Path,
    mode: str = "report",          # report | copy | move
    meteor_dir: str | Path | None = None,
    noise_dir: str | Path | None = None,
    threshold: float = 0.5,
    recursive: bool = True,
    save_annotated: bool = False,
    files: Iterable[Path] | None = None,
    csv_path: str | Path | None = None,
    progress=None,                 # callable(done, total, result) 回调，可空
    workers: int = 1,              # 并行推理 worker 数（>1 需提供 detector_factory）
    detector_factory: Callable[[], MeteorDetector] | None = None,
    review_min: float = 0.0,       # 复核带下限（0=关闭）；[review_min, threshold) 待人工复核
    review_dir: str | Path | None = None,
) -> dict:
    """批量筛选一个目录。

    mode=report: 只判定不移动文件；
    mode=copy/move: 流星图归档到 meteor_dir，其余归档到 noise_dir
                    （默认为 input_dir 同级的 <目录名>_meteor / <目录名>_noise）。
    review_min>0 时置信度落在 [review_min, threshold) 的图片归档到 review_dir
    （默认 <目录名>_review），避免把暗弱流星直接判为非流星而丢失。
    workers>1 且给出 detector_factory 时并行推理（每个 worker 独占一个
    ONNX session）；文件归档仍在主线程串行执行，保证结果顺序稳定。
    返回 {"results": [...], "summary": {...}}。
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    file_list = list(files) if files is not None else iter_images(input_dir, recursive)

    if (mode in ("copy", "move")) and meteor_dir is None:
        meteor_dir = input_dir.parent / f"{input_dir.name}_meteor"
    if (mode in ("copy", "move")) and noise_dir is None:
        noise_dir = input_dir.parent / f"{input_dir.name}_noise"
    if review_min > 0 and review_dir is None:
        review_dir = input_dir.parent / f"{input_dir.name}_review"
    meteor_dir = Path(meteor_dir).resolve() if meteor_dir else None
    noise_dir = Path(noise_dir).resolve() if noise_dir else None
    review_dir = Path(review_dir).resolve() if review_dir else None
    if review_min > 0 and review_min >= threshold:
        review_min = 0.0  # 配置不合理时关闭复核带
        logger.warning("review_min >= threshold，复核带已关闭")
    annotated_dir = None
    if save_annotated:
        base = meteor_dir.parent if meteor_dir else input_dir.parent
        annotated_dir = base / "annotated"

    pool = build_pool(detector, workers, detector_factory)
    if pool.size > 1:
        logger.info("并行模式: %d 个推理 worker", pool.size)

    results: list[BatchResult] = []
    n_meteor = n_review = n_noise = n_error = 0
    t_start = time.time()
    csv_writer = CsvReportWriter(csv_path) if csv_path else None

    def _classify(path: Path) -> BatchResult:
        with pool.borrow() as det:
            return classify_one(det, path, threshold)

    try:
        with ThreadPoolExecutor(max_workers=pool.size) as ex:
            futures = [(p, ex.submit(_classify, p)) for p in file_list]
            for done, (path, fut) in enumerate(futures, 1):
                try:
                    r = fut.result()
                    if r.action != "error":
                        archive_file(r, mode, meteor_dir, noise_dir, input_dir,
                                     review_dir=review_dir, review_min=review_min)
                        if save_annotated and annotated_dir:
                            save_annotated_image(r, annotated_dir, input_dir)
                except Exception as e:
                    r = BatchResult(path=path, action="error", error=repr(e))
                r.img = None  # 释放解码图像
                if r.action == "error":
                    n_error += 1
                elif r.has_meteor:
                    r.binned = r.binned or "meteor"
                    n_meteor += 1
                elif review_min > 0 and r.meteor_score >= review_min:
                    r.binned = r.binned or "review"
                    n_review += 1
                else:
                    r.binned = r.binned or "noise"
                    n_noise += 1
                results.append(r)
                if csv_writer:
                    csv_writer.write(r)
                if progress:
                    try:
                        progress(done, len(futures), r)
                    except Exception:
                        pass
    finally:
        if csv_writer:
            csv_writer.close()

    summary = {
        "input_dir": str(input_dir),
        "total": len(results),
        "meteor": n_meteor,
        "review": n_review,
        "noise": n_noise,
        "error": n_error,
        "mode": mode,
        "threshold": threshold,
        "review_min": review_min,
        "workers": pool.size,
        "elapsed_sec": round(time.time() - t_start, 2),
        "meteor_dir": str(meteor_dir) if meteor_dir else "",
        "review_dir": str(review_dir) if review_dir and review_min > 0 else "",
        "noise_dir": str(noise_dir) if noise_dir else "",
    }
    if csv_path:
        summary["csv_report"] = str(csv_path)
    return {"results": results, "summary": summary}


def result_to_json(r: BatchResult) -> dict:
    return {
        "file": str(r.path),
        "name": r.path.name,
        "has_meteor": r.has_meteor,
        "meteor_score": round(r.meteor_score, 4),
        "top_class": r.top_class,
        "top_score": round(r.top_score, 4),
        "n_detections": r.n_detections,
        "binned": r.binned,
        "action": r.action,
        "error": r.error,
        "detections": r.detections,
    }

# -*- coding: utf-8 -*-
"""批量筛选逻辑：扫描目录 → 逐图判定 → 归档（复制/移动/仅报告）。

供 service.py（HTTP API）和 filter_cli.py（命令行）共用。
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from meteor_detector import MeteorDetector, draw_detections, imread_unicode, imwrite_unicode
import cv2

logger = logging.getLogger("batch_filter")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# 输出目录名，扫描时跳过，避免重复处理
OUTPUT_DIR_NAMES = {"meteor", "noise", "annotated"}


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
    if res["detections"]:
        top = max(res["detections"], key=lambda d: d["score"])
        r.top_class, r.top_score = top["class_name"], top["score"]
    return r


def archive_file(r: BatchResult, mode: str, meteor_dir: Optional[Path],
                 noise_dir: Optional[Path], input_dir: Path) -> None:
    """按判定结果复制/移动文件到对应目录，保持相对子路径。"""
    if r.action == "error":
        return  # 读取失败的文件不归档
    if mode not in ("copy", "move"):
        r.action = "kept"
        return
    dst_root = meteor_dir if r.has_meteor else noise_dir
    if dst_root is None:
        r.action = "kept"
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
        r.action, r.error = "error", str(e)


def save_annotated_image(r: BatchResult, annotated_dir: Path, input_dir: Path) -> None:
    """把带检测框的图片另存，便于人工抽查。"""
    try:
        img = imread_unicode(r.path)
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


def write_csv_report(results: list[BatchResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件", "有流星", "流星置信度", "最高分类别", "最高分",
                    "检测目标数", "处理动作", "错误信息"])
        for r in results:
            w.writerow([str(r.path), r.has_meteor, f"{r.meteor_score:.4f}",
                        r.top_class, f"{r.top_score:.4f}", r.n_detections,
                        r.action, r.error])


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
) -> dict:
    """批量筛选一个目录。

    mode=report: 只判定不移动文件；
    mode=copy/move: 流星图归档到 meteor_dir，其余归档到 noise_dir
                    （默认为 input_dir 同级的 <目录名>_meteor / <目录名>_noise）。
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
    meteor_dir = Path(meteor_dir).resolve() if meteor_dir else None
    noise_dir = Path(noise_dir).resolve() if noise_dir else None
    annotated_dir = None
    if save_annotated:
        base = meteor_dir.parent if meteor_dir else input_dir.parent
        annotated_dir = base / "annotated"

    results: list[BatchResult] = []
    n_meteor = n_noise = n_error = 0
    t_start = time.time()
    for done, path in enumerate(file_list, 1):
        try:
            r = classify_one(detector, path, threshold)
            if r.action != "error":
                archive_file(r, mode, meteor_dir, noise_dir, input_dir)
                if save_annotated and annotated_dir:
                    save_annotated_image(r, annotated_dir, input_dir)
        except Exception as e:
            r = BatchResult(path=path, action="error", error=repr(e))
        if r.action == "error":
            n_error += 1
        elif r.has_meteor:
            n_meteor += 1
        else:
            n_noise += 1
        results.append(r)
        if progress:
            try:
                progress(done, len(file_list), r)
            except Exception:
                pass
        if csv_path:
            write_csv_report(results, Path(csv_path))  # 边跑边写，中断也留记录

    summary = {
        "input_dir": str(input_dir),
        "total": len(results),
        "meteor": n_meteor,
        "noise": n_noise,
        "error": n_error,
        "mode": mode,
        "threshold": threshold,
        "elapsed_sec": round(time.time() - t_start, 2),
        "meteor_dir": str(meteor_dir) if meteor_dir else "",
        "noise_dir": str(noise_dir) if noise_dir else "",
    }
    if csv_path:
        write_csv_report(results, Path(csv_path))
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
        "action": r.action,
        "error": r.error,
        "detections": r.detections,
    }

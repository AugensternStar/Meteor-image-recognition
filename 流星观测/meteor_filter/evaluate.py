# -*- coding: utf-8 -*-
"""模型准确率评估工具：基于 MetDetPy 标注测试视频量化召回/误报。

测试素材为 ref/MetDetPy-main/test/ 下的 20220413Red.mp4 与
20220413_annotation.json（帧级流星真值：时间窗 + 真值框）。

用法：
    python evaluate.py                     # 默认 multiscale=2，扫描多档阈值
    python evaluate.py --multiscale 0      # 评估其他档位
    python evaluate.py --sweep-multiscale  # 对比 0/1/2 三档
    python evaluate.py --no-mask           # 不使用标注自带的区域掩码
    python evaluate.py --save-frames out   # 输出画框帧供人工核对

指标说明（帧级）：
    召回率   = 时间窗内检出流星的帧数 / 时间窗内总帧数
    误报帧数 = 时间窗外被判为流星（≥阈值）的帧数
    定位命中 = 检出框与真值框有交叠（仅统计，不影响召回口径）
注意：单个测试视频只有 1 个流星事件，结果用于相对对比与回归验证，
不适合作为绝对性能结论。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from meteor_detector import MeteorDetector, POSITIVE_CLASSES

ROOT = Path(__file__).parent
DEFAULT_VIDEO = ROOT / "ref" / "MetDetPy-main" / "test" / "20220413Red.mp4"
DEFAULT_ANNO = ROOT / "ref" / "MetDetPy-main" / "test" / "20220413_annotation.json"

SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
POS_STEP, NEG_STEP, NEG_MARGIN = 5, 15, 1.0  # 正/负样本抽帧步长与窗外余量(秒)


def parse_time(s: str) -> float:
    return float((datetime.strptime(s, "%H:%M:%S.%f")
                  - datetime(1900, 1, 1)).total_seconds())


def load_annotation(anno_path: Path):
    anno = json.load(open(anno_path, encoding="utf-8"))
    ah, aw = anno["anno_size"][1], anno["anno_size"][0]
    events = []
    for m in anno["meteors"]:
        t0, t1 = parse_time(m["start_time"]), parse_time(m["end_time"])
        x1, x2 = sorted((m["pt1"][0], m["pt2"][0]))
        y1, y2 = sorted((m["pt1"][1], m["pt2"][1]))
        events.append({"t0": t0, "t1": t1, "box": (x1, y1, x2, y2)})
    return events, aw, ah, anno.get("mask")


def sample_frames(cap, events, fps, n_frames):
    """抽取正/负样本帧号：窗内为正，窗外(留余量)为负。"""
    positives, negatives = [], []
    for i in range(n_frames):
        t = i / fps
        if any(e["t0"] <= t <= e["t1"] for e in events):
            positives.append(i)
        elif all(t < e["t0"] - NEG_MARGIN or t > e["t1"] + NEG_MARGIN
                 for e in events):
            negatives.append(i)
    return positives[::POS_STEP], negatives[::NEG_STEP]


def box_overlap(a, b) -> bool:
    """两个 xyxy 框是否有交叠（帧内流星段是真值路径的一部分，放宽为有交叠即命中）。"""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return ix > 0 and iy > 0


def eval_level(detector: MeteorDetector, cap, events, sx, sy, fps,
               save_dir: Path | None):
    """对一个 multiscale 档位跑完所有抽帧，返回逐帧行 (is_pos, score, hit, train)。

    train=True 表示负例帧（标注窗外）中检出位置与真值框重合——
    通常为流星余迹（persistent train），不计为误报，单独统计。
    """
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pos_ids, neg_ids = sample_frames(cap, events, fps, n_frames)
    rows = []
    print(f"  抽样: 正例 {len(pos_ids)} 帧 {pos_ids} | 负例 {len(neg_ids)} 帧")
    for fid in pos_ids + neg_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = detector.detect(frame)
        pos_dets = [d for d in dets if d.class_name in POSITIVE_CLASSES]
        score = max((d.score for d in pos_dets), default=0.0)
        is_pos = any(e["t0"] <= fid / fps <= e["t1"] for e in events)
        hit, train = False, False
        scaled = [(e["box"][0] * sx, e["box"][1] * sy,
                   e["box"][2] * sx, e["box"][3] * sy) for e in events]
        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in pos_dets]
        if is_pos:
            hit = any(box_overlap(b, g) for b in boxes for g in scaled)
        elif score > 0:
            # 窗外检出位置与真值重合 → 余迹，不算误报
            train = any(box_overlap(b, g) for b in boxes for g in scaled)
        rows.append({"fid": fid, "is_pos": is_pos, "score": score, "hit": hit,
                     "train": train,
                     "top_neg": max(((d.class_name, d.score) for d in dets
                                     if d.class_name not in POSITIVE_CLASSES),
                                    key=lambda x: x[1], default=("", 0.0))})
        if save_dir:
            vis = frame.copy()
            for d in dets:
                color = (0, 255, 0) if d.is_positive else (0, 0, 255)
                cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), color, 3)
                cv2.putText(vis, f"{d.class_name} {d.score:.2f}",
                            (d.x1, max(30, d.y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            for g in scaled:
                cv2.rectangle(vis, (int(g[0]), int(g[1])),
                              (int(g[2]), int(g[3])), (0, 255, 255), 3)
            cv2.imwrite(str(save_dir / f"f{fid:04d}_"
                                     f"{'pos' if is_pos else 'neg'}_s{score:.2f}.jpg"),
                        vis)
    return rows


def report(rows, label: str):
    n_pos = sum(1 for r in rows if r["is_pos"])
    n_neg = len(rows) - n_pos
    n_train = sum(1 for r in rows if r["train"])
    print(f"\n  [{label}] 阈值扫描（正例 {n_pos} 帧 / 负例 {n_neg} 帧，"
          f"其中余迹帧 {n_train}）")
    print(f"  {'阈值':>5} {'检出':>4} {'漏检':>4} {'误报':>4} {'召回率':>7} {'定位命中':>8}")
    for t in SWEEP_THRESHOLDS:
        tp = sum(1 for r in rows if r["is_pos"] and r["score"] >= t)
        fp = sum(1 for r in rows if not r["is_pos"] and r["score"] >= t
                 and not r["train"])
        fn = n_pos - tp
        recall = tp / n_pos if n_pos else 0.0
        loc = sum(1 for r in rows if r["is_pos"] and r["score"] >= t and r["hit"])
        print(f"  {t:>5.2f} {tp:>4} {fn:>4} {fp:>4} {recall:>7.1%} "
              f"{loc:>6}/{tp or 1:<4}")
    scores = [f"f{r['fid']}={r['score']:.2f}" for r in rows if r["is_pos"]]
    print(f"  正例逐帧分数: {', '.join(scores)}")
    trains = [f"f{r['fid']}={r['score']:.2f}" for r in rows if r["train"]]
    if trains:
        print(f"  余迹检出(窗外、位置与真值重合): {', '.join(trains)}")
    fp_detail = [f"f{r['fid']}({r['top_neg'][0]} {r['top_neg'][1]:.2f})"
                 for r in rows if not r["is_pos"] and not r["train"]
                 and max(r["score"], r["top_neg"][1]) >= 0.3]
    if fp_detail:
        print(f"  真误报(窗外且位置不符): {', '.join(fp_detail)}")


def main():
    ap = argparse.ArgumentParser(description="流星检测准确率评估（标注视频）")
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--anno", default=str(DEFAULT_ANNO))
    ap.add_argument("--multiscale", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--sweep-multiscale", action="store_true",
                    help="依次评估 multiscale 0/1/2 三档")
    ap.add_argument("--no-mask", action="store_true",
                    help="不使用标注自带的区域掩码")
    ap.add_argument("--save-frames", default=None, help="保存画框帧的目录")
    args = ap.parse_args()

    events, aw, ah, mask_name = load_annotation(Path(args.anno))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频: {int(vw)}x{int(vh)} @ {fps:.0f}fps | 流星事件 {len(events)} 个: "
          + ", ".join(f"[{e['t0']:.1f}s~{e['t1']:.1f}s]" for e in events))

    mask_path = None
    if mask_name and not args.no_mask:
        cand = Path(args.anno).parent / mask_name
        if cand.exists():
            mask_path = cand
    save_dir = Path(args.save_frames) if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    levels = [0, 1, 2] if args.sweep_multiscale else [args.multiscale]
    for level in levels:
        print(f"\n===== multiscale={level}"
              f"{'（带掩码）' if mask_path else ''} =====")
        det = MeteorDetector(multiscale_pred=level, mask_path=mask_path)
        rows = eval_level(det, cap, events, vw / aw, vh / ah, fps, save_dir)
        report(rows, f"multiscale={level}")
    cap.release()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""流星图片检测核心模块。

基于 MetDetPy 项目提供的 YOLOv5s 预训练模型（weights/yolov5s_v2.onnx），
对单张图片推理，返回图中所有目标（流星/飞机卫星/飞虫/闪电等）的
边界框与置信度。算法流程（多尺度切块推理 + 类别感知 NMS +
sqrt(objectness × class_score) 置信度）与 MetDetPy 保持一致。

独立可用：
    from meteor_detector import MeteorDetector
    det = MeteorDetector()
    results = det.detect_file("some_image.jpg")
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("meteor_detector")

# ---------------------------------------------------------------------------
# 类别定义（与 MetDetPy global/class_name.txt 及模型输出列一一对应）
# 第 8 列 DROPPED 是模型自动生成的"应当丢弃"类别，判定时忽略
# ---------------------------------------------------------------------------
CLASSES = [
    "METEOR",          # 0 流星
    "PLANE/SATELLITE", # 1 飞机或卫星
    "RED_SPRITE",      # 2 红色精灵（高层大气放电现象，正类）
    "LIGHTNING",       # 3 闪电
    "JET",             # # 4 喷气式飞机（凝结尾迹）
    "RARE_SPRITE",     # 5 罕见精灵
    "SPACECRAFT",      # 6 航天器
    "BUGS",            # 7 飞虫
    "DROPPED",         # 8 模型自动生成的丢弃类（不参与判定）
]
POSITIVE_CLASSES = ("METEOR", "RED_SPRITE")  # 判定为"有流星/值得保留"的类别

PARTITION_MIN_OVERLAP = 0.2  # 多尺度切块时相邻块的最小重叠比例（与 MetDetPy 一致）

# 推理后端优先级：CUDA > DirectML(任意 Windows GPU) > CPU。
# 装了 onnxruntime-gpu / onnxruntime-directml 后自动启用，无需改代码。
_PROVIDER_PRIORITY = ("CUDAExecutionProvider", "DmlExecutionProvider",
                      "CPUExecutionProvider")


def pick_providers(preferred: str | None = None) -> list[str]:
    """按优先级选择可用的 ONNX Runtime 执行后端。"""
    available = set(ort.get_available_providers())
    if preferred:
        candidates = [preferred]
    else:
        candidates = list(_PROVIDER_PRIORITY)
    chosen = [p for p in candidates if p in available]
    if not chosen:
        chosen = ["CPUExecutionProvider"]
    # GPU 后端之后补充 CPU 兜底，避免算子不支持时报错
    if chosen != ["CPUExecutionProvider"] and "CPUExecutionProvider" in available:
        chosen.append("CPUExecutionProvider")
    return chosen


def imread_unicode(path: str | Path):
    """读取图片，兼容 Windows 中文路径（cv2.imread 不支持非 ASCII 路径）。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray) -> bool:
    """保存图片，兼容 Windows 中文路径。"""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


@dataclass
class Detection:
    """单个检测结果。坐标为原图像素坐标 (x1, y1, x2, y2)。"""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float        # sqrt(objectness * class_score)，范围 0~1
    class_id: int
    class_name: str

    @property
    def is_positive(self) -> bool:
        return self.class_name in POSITIVE_CLASSES

    def to_dict(self) -> dict:
        return {
            "box": [self.x1, self.y1, self.x2, self.y2],
            "score": round(self.score, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "is_positive": self.is_positive,
        }


def _class_aware_nms(boxes_tlwh: np.ndarray, class_scores: np.ndarray,
                     score_threshold: float, nms_threshold: float) -> np.ndarray:
    """类别感知 NMS：只抑制同类的重叠框。返回按分数降序的选中下标。"""
    if len(boxes_tlwh) == 0:
        return np.empty(0, dtype=np.intp)
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    try:
        selected = cv2.dnn.NMSBoxesBatched(
            bboxes=boxes_tlwh.tolist(), scores=scores.tolist(),
            class_ids=class_ids.tolist(),
            score_threshold=score_threshold, nms_threshold=nms_threshold)
    except AttributeError:  # 旧版 OpenCV 无 NMSBoxesBatched，退化为逐类 NMS
        keep_all = []
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            sel = cv2.dnn.NMSBoxes(
                bboxes=boxes_tlwh[idx].tolist(), scores=scores[idx].tolist(),
                score_threshold=score_threshold, nms_threshold=nms_threshold)
            keep_all.extend(idx[np.asarray(sel).reshape(-1)])
        selected = np.asarray(sorted(keep_all), dtype=np.intp)
    selected = np.asarray(selected, dtype=np.intp).reshape(-1)
    return selected[np.argsort(scores[selected])[::-1]]


class MeteorDetector:
    """封装 YOLOv5s ONNX 模型的流星检测器。

    参数与 MetDetPy config/dldet.json 的模型节保持一致，默认值即为其
    recheck（复核）配置，适合离线批量筛选图片的场景。
    """

    def __init__(
        self,
        weight_path: str | Path = None,
        multiscale_pred: int = 2,
        multiscale_partition: int = 2,
        objectness_thre: float = 0.25,
        pos_thre: float = 0.25,
        nms_thre: float = 0.45,
        hw_tolerance: float = 0.2,
        num_threads: int = 0,
        providers: list[str] | None = None,
        mask_path: str | Path | None = None,
    ):
        if weight_path is None:
            weight_path = Path(__file__).parent / "weights" / "yolov5s_v2.onnx"
        self.weight_path = str(weight_path)
        self.multiscale_pred = multiscale_pred
        self.multiscale_partition = multiscale_partition
        self.objectness_thre = objectness_thre
        self.pos_thre = pos_thre
        self.nms_thre = nms_thre
        self.hw_tolerance = hw_tolerance

        session_options = ort.SessionOptions()
        if num_threads > 0:
            session_options.intra_op_num_threads = num_threads
        self.providers = pick_providers() if providers is None else providers
        self.session = ort.InferenceSession(
            self.weight_path, sess_options=session_options,
            providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        # 模型输入形如 [1, 3, 544, 960]
        self.batch, self.channels, self.h, self.w = self.session.get_inputs()[0].shape
        self.hw_ratio = self.h / self.w
        # 单个 ONNX session 非线程安全，用锁串行化推理
        self._lock = threading.Lock()
        # 预热
        self._run_model(np.zeros((1, self.channels, self.h, self.w), np.float32))
        device = "GPU" if any("CPU" not in p for p in self.providers) else "CPU"
        logger.info("模型已加载: %s, 输入 %dx%d, 后端 %s (%s)",
                    self.weight_path, self.w, self.h, device,
                    ",".join(self.providers))
        self.region_mask: Optional[np.ndarray] = None
        if mask_path:
            self.set_mask(mask_path)

    def set_mask(self, mask_path: str | Path) -> None:
        """设置有效区域掩码（白=检测有效，黑=忽略，如建筑/地平线灯）。

        掩码尺寸任意，检测时自动缩放到图片大小；检测框中心落在黑色
        区域内的目标将被丢弃。传不存在路径会抛错以便及早发现配置问题。
        """
        m = imread_unicode(mask_path)
        if m is None:
            raise ValueError(f"无法读取掩码图片: {mask_path}")
        self.region_mask = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) \
            if m.ndim == 3 else m
        logger.info("已启用区域掩码: %s (有效区占比 %.1f%%)", mask_path,
                    100.0 * (self.region_mask >= 128).mean())

    def _apply_region_mask(self, detections: list, h: int, w: int) -> list:
        """丢弃检测框中心落在掩码黑色区域内的目标。"""
        if not detections or self.region_mask is None:
            return detections
        mask = self.region_mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        keep = []
        for d in detections:
            cx, cy = (d.x1 + d.x2) // 2, (d.y1 + d.y2) // 2
            if mask[max(0, min(h - 1, cy)), max(0, min(w - 1, cx))] >= 128:
                keep.append(d)
        return keep

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _run_model(self, tensor: np.ndarray) -> np.ndarray:
        """执行一次 ONNX 前向，返回 (N, 4+1+num_cls) 的原始输出。"""
        with self._lock:
            out = self.session.run(None, {self.input_name: tensor})
        return out[0][0]

    def _forward_tile(self, x: np.ndarray):
        """对一块 uint8 RGB 图 (H,W,3) 推理（先缩放后转 float，省内存带宽）。

        返回 (boxes_xyxy_int, joint_scores)，坐标已映射回该块原始尺寸。
        """
        h, w = x.shape[:2]
        if h != self.h or w != self.w:
            x = cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        tensor = np.ascontiguousarray(
            x.transpose(2, 0, 1)).astype(np.float32)
        tensor *= np.float32(1.0 / 255.0)
        tensor = tensor[None]
        results = self._run_model(tensor)

        # objectness 独立门槛：类别分数再高也救不了低 objectness 的框
        if self.objectness_thre > 0:
            results = results[results[:, 4] > self.objectness_thre]
        if len(results) == 0:
            return (np.zeros((0, 4), dtype=np.int64),
                    np.zeros((0, len(CLASSES)), dtype=np.float64))

        objectness = results[:, 4]
        joint = results[:, 5:] * objectness[:, None]  # (N, num_cls)

        # 类别感知 NMS（分数门槛用 pos_thre 的平方，与 MetDetPy 语义一致）
        tlwh = results[:, :4].copy()
        tlwh[:, 0] -= tlwh[:, 2] / 2
        tlwh[:, 1] -= tlwh[:, 3] / 2
        selected = _class_aware_nms(tlwh, joint,
                                    self.pos_thre ** 2, self.nms_thre)
        results, joint = results[selected], joint[selected]

        boxes = results[:, :4].copy()
        # cxcywh -> xyxy（x2/y2 必须由 x1/y1 加上宽高得到）
        boxes[:, 0] -= results[:, 2] / 2
        boxes[:, 1] -= results[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + results[:, 2]
        boxes[:, 3] = boxes[:, 1] + results[:, 3]
        # 映射回切块原始尺寸
        if h != self.h or w != self.w:
            boxes[:, 0] *= w / self.w
            boxes[:, 2] *= w / self.w
            boxes[:, 1] *= h / self.h
            boxes[:, 3] *= h / self.h
        return np.array(boxes, dtype=np.int64), joint

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """对一张 BGR uint8 图片检测，返回所有目标（含负类）。"""
        h, w = image_bgr.shape[:2]

        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        x = np.ascontiguousarray(image_bgr[..., ::-1])  # BGR -> RGB（模型按 RGB 训练）

        if self.multiscale_pred == 0:
            pos, joint = self._forward_tile(x)
        else:
            pos, joint = self._multiscale_forward(x, h, w)

        scores = np.sqrt(joint)  # 与 MetDetPy 一致：sqrt 修正低联合分数
        detections = []
        for box, sc in zip(pos, scores):
            cid = int(np.argmax(sc))
            name = CLASSES[cid] if cid < len(CLASSES) else "UNKNOWN"
            detections.append(Detection(
                x1=int(max(0, box[0])), y1=int(max(0, box[1])),
                x2=int(min(w, box[2])), y2=int(min(h, box[3])),
                score=float(sc[cid]), class_id=cid, class_name=name))
        return self._apply_region_mask(detections, h, w)

    def _multiscale_forward(self, x: np.ndarray, h: int, w: int):
        """多尺度切块推理（与 MetDetPy YOLOModel.forward 相同的切图策略）。

        尺度 0：整图直接缩放推理；
        尺度 k：把图切成 partition^k 块（相邻块 20% 重叠）分别推理，
        使小目标在切块后相对更大，提高召回。最后跨块 NMS 合并。
        """
        input_hw_ratio = h / w
        transpose_flag = False
        h_rep, w_rep = 1, 1
        # 图像长宽比与模型差异过大时，先转置再切块
        if abs(self.hw_ratio - input_hw_ratio) > self.hw_tolerance:
            if ((input_hw_ratio - 1) * (self.hw_ratio - 1)) > 0 or \
               abs(self.hw_ratio - 1 / input_hw_ratio) < self.hw_tolerance:
                transpose_flag = True
                x = np.ascontiguousarray(x.transpose(1, 0, 2))
                input_hw_ratio = 1 / input_hw_ratio
                h, w = w, h
            if h > w:
                h_rep = int(np.ceil(h * self.w / (self.h * w)))
            else:
                w_rep = int(np.ceil(w * self.h / (h * self.w)))

        n = self.multiscale_partition ** 2
        result_pos, result_joint = [], []
        for scale in range(self.multiscale_pred):
            if scale > 0:
                h_rep *= self.multiscale_partition
                w_rep *= self.multiscale_partition
            tot_h = (h_rep - 1) * PARTITION_MIN_OVERLAP
            tot_w = (w_rep - 1) * PARTITION_MIN_OVERLAP
            h_size = int(h // (h_rep - tot_h))
            w_size = int(w // (w_rep - tot_w))
            h_stride = int(h // (h_rep + tot_h))
            w_stride = int(w // (w_rep + tot_w))
            for i in range(h_rep):
                for j in range(w_rep):
                    tile = x[i * h_stride:i * h_stride + h_size,
                             j * w_stride:j * w_stride + w_size]
                    if tile.shape[0] < 8 or tile.shape[1] < 8:
                        continue
                    clip_pos, clip_joint = self._forward_tile(tile)
                    clip_pos = clip_pos.astype(np.float64).copy()
                    clip_pos[:, 1] += i * h_stride
                    clip_pos[:, 3] += i * h_stride
                    clip_pos[:, 0] += j * w_stride
                    clip_pos[:, 2] += j * w_stride
                    result_pos.append(clip_pos)
                    result_joint.append(clip_joint)

        if not result_pos:
            return (np.zeros((0, 4), dtype=np.int64),
                    np.zeros((0, len(CLASSES)), dtype=np.float64))
        all_pos = np.concatenate(result_pos, axis=0)
        all_joint = np.concatenate(result_joint, axis=0)

        # 跨块合并（把 xyxy 转回 tlwh 做 NMS）
        tlwh = all_pos.copy()
        tlwh[:, 2] -= tlwh[:, 0]
        tlwh[:, 3] -= tlwh[:, 1]
        selected = _class_aware_nms(tlwh, all_joint,
                                    self.pos_thre ** 2, self.nms_thre)
        all_pos, all_joint = all_pos[selected], all_joint[selected]

        if transpose_flag:  # 转置回原始方向
            all_pos = all_pos[:, [1, 0, 3, 2]]
        return np.array(all_pos, dtype=np.int64), all_joint

    # ------------------------------------------------------------------
    # 文件级接口
    # ------------------------------------------------------------------
    def detect_file(self, path: str | Path) -> list[Detection]:
        img = imread_unicode(path)
        if img is None:
            raise ValueError(f"无法读取图片: {path}")
        return self.detect(img)

    def filter_image(self, image_bgr: np.ndarray, threshold: float = 0.5,
                     ) -> dict:
        """完整的筛选判定：图片是否带有流星。

        返回:
            {
              "has_meteor": bool,     # 存在正类目标且分数 ≥ threshold
              "meteor_score": float,  # 最高正类置信度（无正类目标时为 0；
                                      # 与 threshold 无关，供复核带等使用）
              "detections": [...],    # 全部检测目标（含飞机/飞虫等负类）
            }
        """
        detections = self.detect(image_bgr)
        positives = sorted((d for d in detections if d.is_positive),
                           key=lambda d: d.score, reverse=True)
        return {
            "has_meteor": bool(positives) and positives[0].score >= threshold,
            "meteor_score": positives[0].score if positives else 0.0,
            "detections": [d.to_dict() for d in detections],
        }

    def filter_file(self, path: str | Path, threshold: float = 0.5) -> dict:
        img = imread_unicode(path)
        if img is None:
            raise ValueError(f"无法读取图片: {path}")
        return self.filter_image(img, threshold)


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """把检测结果画在图上（正类绿框、负类红框），用于人工核对。"""
    vis = image_bgr.copy()
    for d in detections:
        color = (0, 255, 0) if d.is_positive else (0, 0, 255)
        cv2.rectangle(vis, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(vis, f"{d.class_name} {d.score:.2f}",
                    (d.x1, max(12, d.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis

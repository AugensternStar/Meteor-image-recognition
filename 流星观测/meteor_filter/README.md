# 流星图片自动筛选服务

从摄像头自动捕捉 → 合成图片之后，用本程序**自动判断每张图片是否含有流星**，
并把流星图归档（剔除飞机/卫星/飞虫/闪电等误检），供后续轨迹计算与多站点落点解算使用。

检测模型采用 [MetDetPy](https://github.com/LilacMeteorObservatory/MetDetPy)（MPL-2.0）
提供的 YOLOv5s 预训练权重（`weights/yolov5s_v2.onnx`，输入 960×544，ONNX 格式，
CPU 即可运行），推理流程（多尺度切块 + 类别感知 NMS + sqrt(objectness×class) 置信度）
与 MetDetPy 保持一致。方案另参考了 RMS 的 MLFilter 与
[satmonkey/MeteorDL](https://github.com/satmonkey/MeteorDL)。

## 目录结构

```
meteor_filter/
├── meteor_detector.py    # 核心检测模块（可独立 import）
├── batch_filter.py       # 批量筛选共享逻辑
├── service.py            # HTTP API 服务（FastAPI，含网页测试界面）
├── filter_cli.py         # 命令行批量筛选 / 目录监听模式
├── start_server.bat      # 双击启动 API 服务
├── weights/yolov5s_v2.onnx
├── requirements.txt
├── demo_input/           # 示例图片（含真实流星帧，来自 MetDetPy 测试视频）
└── ref/                  # MetDetPy 参考源码与测试素材
```

## 安装

Python ≥ 3.10（本机 Anaconda 已验证 3.14）：

```bash
pip install -r requirements.txt
```

## 快速验证

```bash
python filter_cli.py --input demo_input --mode copy --save-annotated
# 流星图 → demo_input_meteor/，其余 → demo_input_noise/，画框图 → annotated/
```

## 用法一：HTTP API 服务（推荐接入方式）

```bash
python service.py            # 或双击 start_server.bat
# 服务地址 http://127.0.0.1:8734 ，浏览器打开有网页测试界面
```

| 接口 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/predict` | POST | 上传单张图片（multipart），返回判定 |
| `/api/predict_base64` | POST | JSON 提交 base64 图片，同上 |
| `/api/filter_folder` | POST | 批量筛选服务器本地目录并归档 |
| `/` | GET | 网页测试界面 |

### 单张判定

```bash
curl -F "file=@meteor.jpg" -F "threshold=0.5" http://127.0.0.1:8734/api/predict
```

返回：

```json
{
  "has_meteor": true,
  "meteor_score": 0.9043,
  "detections": [{"box": [1305,503,1416,634], "score": 0.9043,
                   "class_name": "METEOR", "is_positive": true}]
}
```

Python 调用（适合接入现有捕捉/合成流水线）：

```python
import requests
r = requests.post("http://127.0.0.1:8734/api/predict",
                  files={"file": open("合成图.jpg", "rb")},
                  data={"threshold": 0.5}).json()
if r["has_meteor"]:
    ...  # 送去算轨迹 / 多站点匹配
```

`draw=true` 时额外返回画好检测框的 `annotated_image_base64`，便于人工抽查。

### 目录批量筛选（一次调用自动完成）

```python
import requests
r = requests.post("http://127.0.0.1:8734/api/filter_folder", json={
    "input_dir": r"D:\captures\20261024_night",  # 服务器上的合成图目录
    "mode": "copy",        # copy=复制归档 / move=移动归档 / report=仅报告
    "threshold": 0.5,
    "save_annotated": False,   # True 时额外保存画框图
}).json()
print(r["summary"])   # {total, meteor, noise, error, meteor_dir, csv_report, ...}
```

流星图归档到 `<input>_meteor/`，其余到 `<input>_noise/`，并生成 CSV 明细报告。
`workers` 可指定并行推理 worker 数（0=按 CPU 核数自动，每个 worker 独占一个
ONNX session）。注意：该接口为同步调用，图片很多时耗时较长，持续自动化
建议使用 filter_cli.py --watch。

## 用法二：命令行 / 持续监听

```bash
# 一次性批量筛选
python filter_cli.py --input D:\captures\20261024_night --mode copy

# 持续监听目录：合成图一落盘就自动筛选归档（全自动，推荐）
python filter_cli.py --input D:\captures\live --mode move --watch --interval 15
#   流星图 → live_meteor/，非流星图 → live_noise/，Ctrl+C 停止

# 常用参数
#   --threshold 0.5     正类阈值（误检多调高，漏检多调低）
#   --multiscale 0      0=最快（约0.1s/张） / 2=默认最准（小流星召回更好）
#   --workers N         并行推理 worker 数（0=自动按 CPU 核数，1=单线程）
#   --save-annotated    额外输出画框图
```

监听模式对写入中的图片有容错：文件大小稳定但尚解码失败（写入中途被扫描到）
时会自动重试最多 3 次，重试期间不归档；重试耗尽仍失败的文件保留在原目录并
打印 `[错误]`，不会丢图也不会死循环。

## 判定规则与调优

- 模型识别 8 类目标：`METEOR`、`PLANE/SATELLITE`、`RED_SPRITE`、`LIGHTNING`、
  `JET`、`RARE_SPRITE`、`SPACECRAFT`、`BUGS`（飞虫）。
- **判定为"有流星"的条件**：存在 METEOR 或 RED_SPRITE 类目标，且置信度
  （sqrt(objectness×类别分)）≥ `threshold`（默认 0.5）。
- 所有检测结果（含飞机、飞虫等负类）都会返回，可据此统计误检来源。
- 若白天/强月光场景误检偏多，优先调高 threshold 至 0.6~0.7；
  暗弱流星被漏检时降至 0.35~0.4，并用 `--save-annotated` 人工抽查。

## 性能（本机 CPU 实测，20 核，multiscale=2，1080p）

- 预处理已优化为 uint8 切块（整图不再转 float32 再缩放），单线程约
  **0.5 秒/张**；`--workers 5` 并行后约 **0.32 秒/张**（此前版本约 4.5 秒/张）。
- 批量 `--multiscale 0`（整图一次前向）可再提速约 5 倍，适合初筛。
- CSV 报告为增量追加写入，上万张图也不会因反复重写整个文件而变慢。
- **GPU 加速（可选）**：启动时自动探测推理后端（CUDA > DirectML > CPU）。
  Windows 上任意显卡（含核显）装 DirectML 版即可提速：
  `pip uninstall onnxruntime && pip install onnxruntime-directml`，
  代码无需任何修改。

## 已知局限与后续

- 模型在夜间监控视频域训练；大幅面星野深空照片、极端曝光未验证。
- 若累积了自有标注数据，可用同结构 YOLOv5 重新训练后直接替换
  `weights/yolov5s_v2.onnx`（输入 960×544，输出 [1,32130,14]）。
- 可进一步提速的方向：INT8 量化模型（CPU 约 2 倍，需校准数据）、
  对明显空白的夜空帧做快速预筛（统计亮像素跳过 YOLO，需谨慎以免漏检）。

## 变更记录

- 2026-10-25
  - 修复检测框 cxcywh→xyxy 转换错误（x2/y2 偏大半个宽高，画框与 NMS 受影响）。
  - 修复 watch 模式漏文件：写入中的文件曾被永久跳过，现会自动重试。
  - 预处理改为 uint8 切块，单线程约提速 9 倍；新增 `--workers` 并行推理。
  - CSV 改为增量写入；`move + --save-annotated` 不再因源文件已移走而漏存画框图。
  - 自动探测 CUDA / DirectML 推理后端（安装对应 onnxruntime 包即生效）。

## 致谢

- [MetDetPy](https://github.com/LilacMeteorObservatory/MetDetPy)（MPL-2.0）：
  预训练模型与推理算法来源；`ref/` 内保留了其参考源码与测试视频。
- [RMS / GMN](https://github.com/CroatianMeteorNetwork/RMS)：观测流程与
  MLFilter 误检过滤思路。
- [satmonkey/MeteorDL](https://github.com/satmonkey/MeteorDL)：实时前端 +
  maxpixel 图检测思路参考。

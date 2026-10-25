# -*- coding: utf-8 -*-
"""流星图片筛选 HTTP 服务。

启动：
    uvicorn service:app --host 0.0.0.0 --port 8734
或直接：
    python service.py

接口一览：
    GET  /health                 健康检查
    POST /api/predict            上传单张图片（multipart），返回是否含流星
    POST /api/predict_base64     JSON 提交 base64 图片，同上（方便脚本调用）
    POST /api/filter_folder      批量筛选服务器本地目录（自动归档 + CSV 报告）
    GET  /                       网页测试界面
"""
from __future__ import annotations

import base64
import io
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from meteor_detector import (MeteorDetector, POSITIVE_CLASSES, CLASSES,
                             Detection, draw_detections)
from batch_filter import run_batch, result_to_json, auto_workers

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("meteor_service")

MODEL_PATH = Path(__file__).parent / "weights" / "yolov5s_v2.onnx"

detector: MeteorDetector | None = None

# 批量任务互斥锁：防止并发 filter_folder 各建一组 session 导致 CPU 超额订阅
_batch_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = MeteorDetector(MODEL_PATH)
    logger.info("流星筛选服务就绪: http://127.0.0.1:8734")
    yield
    detector = None


app = FastAPI(title="流星图片筛选服务", version="1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def decode_upload(data: bytes) -> np.ndarray:
    """字节数据解码为 BGR 图像（自动处理灰度/RGBA）。"""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "不是有效的图片文件（支持 jpg/png/bmp/tif/webp）")
    return img


def filter_bytes(data: bytes, threshold: float, draw: bool) -> dict:
    img = decode_upload(data)
    res = detector.filter_image(img, threshold)
    out = {
        "has_meteor": res["has_meteor"],
        "meteor_score": round(res["meteor_score"], 4),
        "positive_classes": list(POSITIVE_CLASSES),
        "threshold": threshold,
        "detections": res["detections"],
    }
    if draw and res["detections"]:
        dets = [Detection(d["box"][0], d["box"][1], d["box"][2], d["box"][3],
                          d["score"], d["class_id"], d["class_name"])
                for d in res["detections"]]
        ok, buf = cv2.imencode(".jpg", draw_detections(img, dets))
        if ok:
            out["annotated_image_base64"] = base64.b64encode(buf).decode()
    return out


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "input_shape": f"1x{detector.channels}x{detector.h}x{detector.w}"
                       if detector else "loading",
        "classes": CLASSES[:-1],
        "positive_classes": list(POSITIVE_CLASSES),
    }


@app.post("/api/predict")
def predict(file: UploadFile = File(...),
            threshold: float = Form(0.5),
            draw: bool = Form(False)):
    """上传单张图片，判定是否含流星。

    - threshold: 正类判定阈值（0~1），默认 0.5
    - draw: 为 True 时额外返回画好框的图片（base64 JPEG）
    """
    data = file.file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(400, "图片超过 100MB 限制")
    return filter_bytes(data, threshold, draw)


class Base64Request(BaseModel):
    image_base64: str = Field(..., description="图片的 base64 字符串")
    threshold: float = 0.5
    draw: bool = False


@app.post("/api/predict_base64")
def predict_base64(req: Base64Request):
    """与 /api/predict 相同，但以 JSON base64 方式提交，方便脚本集成。"""
    try:
        data = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(400, "image_base64 不是合法的 base64 字符串")
    return filter_bytes(data, req.threshold, req.draw)


class FolderRequest(BaseModel):
    input_dir: str = Field(..., description="服务器上的图片目录（绝对路径）")
    mode: str = Field("copy", description="report=只报告 / copy=复制归档 / move=移动归档")
    meteor_dir: str | None = Field(None, description="流星图归档目录（默认 自动）")
    noise_dir: str | None = Field(None, description="非流星图归档目录（默认 自动）")
    threshold: float = 0.5
    recursive: bool = True
    save_annotated: bool = Field(False, description="额外保存画框图，便于人工抽查")
    workers: int = Field(0, description="并行推理 worker 数（0=按 CPU 自动，1=单线程）")


@app.post("/api/filter_folder")
def filter_folder(req: FolderRequest):
    """批量筛选服务器本地目录并自动归档，返回汇总与逐图明细。

    默认 mode=copy：流星图复制到 <input>_meteor/，其余复制到 <input>_noise/，
    并在归档目录旁生成 CSV 报告。workers>1 时多 ONNX session 并行推理。
    持续自动化建议使用 filter_cli.py --watch。
    """
    if req.mode not in ("report", "copy", "move"):
        raise HTTPException(400, "mode 只能是 report / copy / move")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(req.input_dir).resolve().parent / f"filter_report_{stamp}.csv"
    workers = req.workers if req.workers > 0 else auto_workers()
    try:
        with _batch_lock:
            out = run_batch(detector, req.input_dir, mode=req.mode,
                            meteor_dir=req.meteor_dir, noise_dir=req.noise_dir,
                            threshold=req.threshold, recursive=req.recursive,
                            save_annotated=req.save_annotated,
                            csv_path=csv_path,
                            workers=workers,
                            detector_factory=lambda: MeteorDetector(MODEL_PATH))
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    return {
        "summary": out["summary"],
        "results": [result_to_json(r) for r in out["results"]],
    }


# ---------------------------------------------------------------------------
# 网页测试界面
# ---------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>流星图片筛选</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:960px;margin:24px auto;padding:0 12px;background:#0d1117;color:#e6edf3}
h1{font-size:22px} h2{font-size:16px;margin-top:28px;border-left:3px solid #58a6ff;padding-left:8px}
button{background:#238636;color:#fff;border:none;border-radius:6px;padding:8px 18px;font-size:14px;cursor:pointer;margin-right:8px}
button:disabled{background:#30363d;cursor:wait}
input,select{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:6px 8px}
#drop{border:2px dashed #30363d;border-radius:10px;padding:28px;text-align:center;color:#8b949e;margin:10px 0}
#drop.hover{border-color:#58a6ff;color:#58a6ff}
img#preview{max-width:100%;border-radius:8px;margin-top:10px}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px}
td,th{border:1px solid #30363d;padding:5px 8px;text-align:left}
.ok{color:#3fb950}.bad{color:#f85149}.dim{color:#8b949e}
#summary{margin-top:8px;line-height:1.8}
</style></head><body>
<h1>🌠 流星图片筛选（MetDetPy YOLOv5s）</h1>
<div class="dim">上传图片判定是否含流星；或填写目录路径批量筛选归档。</div>

<h2>单张图片测试</h2>
<div id="drop">拖拽图片到此处，或 <input type="file" id="file" accept="image/*"></div>
<div>
  <label>阈值 <input id="thr" type="number" value="0.5" step="0.05" min="0" max="1" style="width:70px"></label>
  <label><input id="draw" type="checkbox" checked> 返回标注图</label>
  <button id="btn1" onclick="doPredict()">检测</button>
</div>
<div id="out1"></div>
<img id="preview" style="display:none">

<h2>目录批量筛选（服务器本地路径）</h2>
<div>
  <label>图片目录 <input id="dir" style="width:340px" placeholder="D:\\captures\\2026-10-24"></label>
  <label>方式
    <select id="mode"><option value="copy">复制归档</option><option value="move">移动归档</option><option value="report">仅报告</option></select>
  </label>
  <label>阈值 <input id="thr2" type="number" value="0.5" step="0.05" min="0" max="1" style="width:70px"></label>
  <button id="btn2" onclick="doFolder()">开始筛选</button>
</div>
<div id="summary"></div>
<div id="out2"></div>

<script>
let img=null;
const drop=document.getElementById('drop');
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hover')};
drop.ondragleave=()=>drop.classList.remove('hover');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hover');
  if(e.dataTransfer.files.length){document.getElementById('file').files=e.dataTransfer.files;loadFile()}};
document.getElementById('file').onchange=loadFile;
function loadFile(){const f=document.getElementById('file').files[0];
  if(!f)return;img=f;const url=URL.createObjectURL(f);
  const p=document.getElementById('preview');p.src=url;p.style.display='block'}

async function doPredict(){
  if(!img){alert('请先选择图片');return}
  const b=document.getElementById('btn1');b.disabled=true;b.textContent='检测中…';
  const fd=new FormData();fd.append('file',img);
  fd.append('threshold',document.getElementById('thr').value);
  fd.append('draw',document.getElementById('draw').checked);
  try{
    const r=await(await fetch('/api/predict',{method:'POST',body:fd})).json();
    if(r.detail){document.getElementById('out1').innerHTML='<span class="bad">'+r.detail+'</span>';return}
    let html=r.has_meteor?'<p class="ok">✔ 检测到流星，置信度 '+r.meteor_score+'</p>':'<p class="bad">✘ 未检测到流星</p>';
    html+='<table><tr><th>类别</th><th>置信度</th><th>边界框 x1,y1,x2,y2</th></tr>';
    for(const d of r.detections)html+='<tr><td>'+d.class_name+(d.is_positive?' ✓':'')+'</td><td>'+d.score+'</td><td>'+d.box.join(', ')+'</td></tr>';
    document.getElementById('out1').innerHTML=html+'</table>';
    if(r.annotated_image_base64){document.getElementById('preview').src='data:image/jpeg;base64,'+r.annotated_image_base64;document.getElementById('preview').style.display='block'}
  }catch(e){document.getElementById('out1').innerHTML='<span class="bad">请求失败: '+e+'</span>'}
  b.disabled=false;b.textContent='检测'}

async function doFolder(){
  const dir=document.getElementById('dir').value.trim();
  if(!dir){alert('请填写目录');return}
  const b=document.getElementById('btn2');b.disabled=true;b.textContent='筛选中…';
  document.getElementById('out2').innerHTML='';document.getElementById('summary').innerHTML='<span class="dim">处理中，图片多时可能需要几分钟…</span>';
  try{
    const r=await(await fetch('/api/filter_folder',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input_dir:dir,mode:document.getElementById('mode').value,
        threshold:parseFloat(document.getElementById('thr2').value)})})).json();
    if(r.detail){document.getElementById('summary').innerHTML='<span class="bad">'+r.detail+'</span>';return}
    const s=r.summary;
    document.getElementById('summary').innerHTML=
      '共 '+s.total+' 张，流星 <b class="ok">'+s.meteor+'</b> 张，非流星 <b class="bad">'+s.noise+'</b> 张，错误 '+s.error+' 张，耗时 '+s.elapsed_sec+'s<br>'+
      '流星目录: '+s.meteor_dir+' &nbsp;|&nbsp; 报告: '+(s.csv_report||'无');
    let html='<table><tr><th>文件</th><th>有流星</th><th>置信度</th><th>最高类别</th><th>动作</th></tr>';
    for(const x of r.results.slice(0,200))html+='<tr><td>'+x.name+'</td><td>'+(x.has_meteor?'<span class=ok>是</span>':'<span class=bad>否</span>')+'</td><td>'+x.meteor_score+'</td><td>'+x.top_class+'</td><td>'+x.action+'</td></tr>';
    document.getElementById('out2').innerHTML=html+'</table>';
  }catch(e){document.getElementById('summary').innerHTML='<span class="bad">请求失败: '+e+'</span>'}
  b.disabled=false;b.textContent='开始筛选'}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8734)

# -*- coding: utf-8 -*-
"""
Lora网格调参可视化报告 v8（固定补偿，补偿值由全量均值决定）
- 第一遍：统计所有图的 hair_ratio，计算均值
- 固定补偿 FIXED_COMPENSATION = min(0.25, hair_ratio_mean × 0.5)
- 第二遍：所有图统一用 FIXED_COMPENSATION 计算头顶线
- 头顶线 = 人脸框顶 - FIXED_COMPENSATION × 脸高
- 下巴线 = Anime Face Detector 点2
- 身高底部 = max(脚踝点y, 人体框底部y)
- Gram矩阵 + 11×11网格 + 灯箱(↑↓纵向/←→横向) + 热力图 + 数据表 + CSV + HTML
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, r"E:\comfyui\pylibs")
sys.path.insert(0, r"E:\comfyui")
sys.path.insert(0, r"E:\comfyui\anime-face-detector\src")

import torch
import cv2
from ultralytics import YOLO
from anime_face_detector import create_detector

# ==================== 配置 ====================
IMG_DIR = r"D:\张有容\张有容\comfyui\lora_grid_images"
OUTPUT_DIR = r"D:\张有容\张有容\comfyui\lora_grid_report"
YOLO_MODEL = r"E:\comfyui\yolov8n-pose.pt"
GRID_SIZE = 11
THUMB_SIZE = 100
LIGHTBOX_SIZE = 600
MAX_COMPENSATION = 0.25
HAIR_FACTOR = 0.5
FALLBACK_HEAD_COEFF = 5.4

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[设备] {DEVICE}")

# ==================== 模型加载 ====================
print("[1/6] 加载模型...")
pose_model = YOLO(YOLO_MODEL)
anime_face_detector = create_detector(device='cuda:0')

import torchvision.models as models
import torchvision.transforms as transforms
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(DEVICE).eval()
vgg_layers = {'3': 'relu1_1', '8': 'relu2_1', '15': 'relu3_1', '22': 'relu4_1', '29': 'relu5_1'}
vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("  所有模型加载完成")

# ==================== 工具函数 ====================
def read_image_cv(img_path):
    return cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

def get_gram_matrices(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = vgg_transform(img).unsqueeze(0).to(DEVICE)
    features = {}
    x = tensor
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in vgg_layers:
            features[vgg_layers[name]] = x
    grams = {}
    for layer_name, feat in features.items():
        b, c, h, w = feat.shape
        feat_reshaped = feat.view(c, h * w)
        gram = torch.mm(feat_reshaped, feat_reshaped.t()) / (c * h * w)
        grams[layer_name] = gram
    return grams

def gram_distance(grams1, grams2):
    dist = 0.0
    for key in grams1:
        if key in grams2:
            dist += torch.sum((grams1[key] - grams2[key]) ** 2).item()
    return dist

def detect_all(img_path):
    """一次性检测：返回人体框、人脸框、下巴点、身体关键点"""
    img = read_image_cv(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # YOLO
    pose_results = pose_model(img, verbose=False)
    pose_result = pose_results[0]
    if pose_result.keypoints is None or len(pose_result.keypoints) == 0:
        return {"img": img, "status": "no_pose"}

    # 取可见点最多的人
    max_visible = 0
    best_kps = None
    best_conf = None
    best_box = None
    for idx, kps in enumerate(pose_result.keypoints):
        conf = kps.conf.cpu().numpy().flatten()
        xy = kps.xy.cpu().numpy()[0]
        visible = int(np.sum(conf > 0.5))
        if visible > max_visible:
            max_visible = visible
            best_kps = xy
            best_conf = conf
            if pose_result.boxes is not None and idx < len(pose_result.boxes):
                best_box = pose_result.boxes[idx].xyxy.cpu().numpy()[0]

    # Anime Face
    face_box_top = None
    face_height = None
    chin_y = None
    try:
        face_results = anime_face_detector(img)
        if face_results:
            fb = face_results[0]['bbox']
            fk = face_results[0]['keypoints']
            face_box_top = fb[1]
            face_height = fb[3] - fb[1]
            if fk[2][2] > 0.3:
                chin_y = fk[2][1]
    except Exception:
        pass

    return {
        "img": img, "status": "ok",
        "kps": best_kps, "conf": best_conf, "box": best_box,
        "keypoint_count": max_visible,
        "face_box_top": face_box_top, "face_height": face_height,
        "chin_y": chin_y,
    }

# ==================== 第一遍：统计 hair_ratio 均值 ====================
print("[2/6] 第一遍：统计头发比均值...")
img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"  共 {len(img_files)} 张图片")

hair_ratios = []
detections = []

for i, fname in enumerate(img_files):
    img_path = os.path.join(IMG_DIR, fname)
    det = detect_all(img_path)
    detections.append(det)

    if det and det["status"] == "ok" and det["box"] is not None and det["face_box_top"] is not None and det["face_height"] and det["face_height"] > 0:
        person_top = det["box"][1]
        hr = (det["face_box_top"] - person_top) / det["face_height"]
        hair_ratios.append(hr)

    if (i + 1) % 30 == 0:
        print(f"  已统计 {i+1}/{len(img_files)}")

hair_ratio_mean = np.mean(hair_ratios)
hair_ratio_median = np.median(hair_ratios)
FIXED_COMPENSATION = min(MAX_COMPENSATION, hair_ratio_mean * HAIR_FACTOR)

print(f"\n  有效统计: {len(hair_ratios)}/{len(img_files)} 张")
print(f"  hair_ratio 均值: {hair_ratio_mean:.4f}")
print(f"  hair_ratio 中位数: {hair_ratio_median:.4f}")
print(f"  固定补偿 FIXED_COMPENSATION = min(0.25, {hair_ratio_mean:.4f} × 0.5) = {FIXED_COMPENSATION:.4f}")

# ==================== 第二遍：用固定补偿计算头身比 ====================
print(f"\n[3/6] 第二遍：固定补偿={FIXED_COMPENSATION:.4f} 计算头身比...")

rows = []
all_grams = []
thumb_b64_list = []
full_b64_list = []
chin_sources = {}
bottom_sources = {}

for i, fname in enumerate(img_files):
    img_path = os.path.join(IMG_DIR, fname)
    det = detections[i]
    idx = i
    pw = round((idx // GRID_SIZE) * 0.1, 1)
    cg = round((idx % GRID_SIZE) * 0.1, 1)

    if det is None or det["status"] != "ok" or det["kps"] is None:
        rows.append({"index": idx, "filename": fname, "pw_strength": pw, "cg_strength": cg,
                      "keypoint_count": 0, "integrity": 0.0, "head_body_ratio": None,
                      "upper_lower_ratio": None, "hair_ratio": None,
                      "head_source": "detect_fail", "chin_source": "none", "bottom_source": "none"})
        grams = get_gram_matrices(img_path)
        all_grams.append(grams)
        thumb_b64_list.append("")
        full_b64_list.append("")
        continue

    kps = det["kps"]
    conf = det["conf"]
    box = det["box"]
    keypoint_count = det["keypoint_count"]
    integrity = keypoint_count / 17.0

    required = [0, 1, 2, 5, 6, 11, 12, 13, 14, 15, 16]
    if any(conf[j] < 0.3 for j in required):
        rows.append({"index": idx, "filename": fname, "pw_strength": pw, "cg_strength": cg,
                      "keypoint_count": keypoint_count, "integrity": round(integrity, 4),
                      "head_body_ratio": None, "upper_lower_ratio": None, "hair_ratio": None,
                      "head_source": "low_conf", "chin_source": "low_conf", "bottom_source": "low_conf"})
        grams = get_gram_matrices(img_path)
        all_grams.append(grams)
        thumb_img = Image.open(img_path).convert("RGB")
        thumb_img.thumbnail((THUMB_SIZE, THUMB_SIZE * 1.5))
        buf = BytesIO(); thumb_img.save(buf, format="JPEG", quality=75)
        thumb_b64_list.append(base64.b64encode(buf.getvalue()).decode())
        full_img = Image.open(img_path).convert("RGB")
        full_img.thumbnail((LIGHTBOX_SIZE, LIGHTBOX_SIZE * 1.8))
        buf2 = BytesIO(); full_img.save(buf2, format="JPEG", quality=85)
        full_b64_list.append(base64.b64encode(buf2.getvalue()).decode())
        continue

    nose = kps[0]
    eye_center = (kps[1] + kps[2]) / 2
    shoulder_center = (kps[5] + kps[6]) / 2
    hip_center = (kps[11] + kps[12]) / 2
    ankle_center = (kps[15] + kps[16]) / 2

    # 头发比（记录用）
    hair_ratio = None
    if box is not None and det["face_box_top"] is not None and det["face_height"]:
        hair_ratio = (det["face_box_top"] - box[1]) / det["face_height"]

    # 头顶线 = 人脸框顶 - 固定补偿 × 脸高
    chin_source = "anime_face"
    head_source = "fixed_compensation"
    if det["face_box_top"] is not None and det["face_height"] and det["face_height"] > 0:
        head_top_y = det["face_box_top"] - FIXED_COMPENSATION * det["face_height"]
    else:
        eye_nose_dist = abs(nose[1] - eye_center[1])
        head_top_y = eye_center[1] - FALLBACK_HEAD_COEFF * eye_nose_dist / 2
        head_source = "estimated_fallback"

    # 下巴
    chin_y = det["chin_y"]
    if chin_y is None:
        chin_y = nose[1] + 2.0 * abs(nose[1] - eye_center[1])
        chin_source = "estimated"

    # 身高底部 = max(脚踝, 框底)
    bottom_y = ankle_center[1]
    bottom_source = "ankle"
    if box is not None:
        if box[3] > bottom_y:
            bottom_y = box[3]
            bottom_source = "box_bottom"

    # 计算
    head_length = chin_y - head_top_y
    body_height = bottom_y - head_top_y
    head_body_ratio = body_height / head_length if head_length > 10 and body_height > 100 else None

    waist_y = hip_center[1] - 0.4 * (hip_center[1] - shoulder_center[1])
    upper = waist_y - head_top_y
    lower = bottom_y - waist_y
    upper_lower_ratio = lower / upper if upper > 1 else None

    chin_sources[chin_source] = chin_sources.get(chin_source, 0) + 1
    bottom_sources[bottom_source] = bottom_sources.get(bottom_source, 0) + 1

    # Gram + 缩略图
    grams = get_gram_matrices(img_path)
    all_grams.append(grams)

    thumb_img = Image.open(img_path).convert("RGB")
    thumb_img.thumbnail((THUMB_SIZE, THUMB_SIZE * 1.5))
    buf = BytesIO(); thumb_img.save(buf, format="JPEG", quality=75)
    thumb_b64_list.append(base64.b64encode(buf.getvalue()).decode())

    full_img = Image.open(img_path).convert("RGB")
    full_img.thumbnail((LIGHTBOX_SIZE, LIGHTBOX_SIZE * 1.8))
    buf2 = BytesIO(); full_img.save(buf2, format="JPEG", quality=85)
    full_b64_list.append(base64.b64encode(buf2.getvalue()).decode())

    rows.append({
        "index": idx, "filename": fname,
        "pw_strength": pw, "cg_strength": cg,
        "keypoint_count": keypoint_count,
        "integrity": round(integrity, 4),
        "head_body_ratio": round(head_body_ratio, 2) if head_body_ratio else None,
        "upper_lower_ratio": round(upper_lower_ratio, 2) if upper_lower_ratio else None,
        "hair_ratio": round(hair_ratio, 3) if hair_ratio is not None else None,
        "head_source": head_source,
        "chin_source": chin_source,
        "bottom_source": bottom_source,
    })

    if (i + 1) % 30 == 0:
        print(f"  已计算 {i+1}/{len(img_files)}")

df = pd.DataFrame(rows)
print(f"\n  下巴来源: {chin_sources}")
print(f"  底部来源: {bottom_sources}")

# ==================== Gram风格距离 ====================
print("[4/6] 计算Gram矩阵风格距离...")
mean_grams = {}
for key in all_grams[0]:
    mean_grams[key] = torch.stack([g[key] for g in all_grams]).mean(dim=0)

style_distances = [gram_distance(g, mean_grams) for g in all_grams]
df["gram_style_distance"] = np.round(style_distances, 6)
max_dist = max(style_distances) if max(style_distances) > 0 else 1
df["style_similarity"] = np.round(1 - np.array(style_distances) / max_dist, 4)

df["head_score"] = df["head_body_ratio"].apply(
    lambda x: max(0, 1.0 - abs(x - 8.5) / 8.5) if pd.notna(x) else 0)
df["ratio_score"] = df["upper_lower_ratio"].apply(
    lambda x: max(0, 1.0 - abs(x - 1.4) / 1.4) if pd.notna(x) else 0)
df["composite_score"] = (df["integrity"] * 0.3 + df["style_similarity"] * 0.25 +
                          df["head_score"] * 0.25 + df["ratio_score"] * 0.2)

csv_path = os.path.join(OUTPUT_DIR, "grid_analysis.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"  CSV已保存: {csv_path}")

# ==================== 统计摘要 ====================
summary = {
    "total_images": len(df),
    "integrity_mean": round(df["integrity"].mean(), 4),
    "integrity_full_count": int((df["keypoint_count"] == 17).sum()),
    "head_body_mean": round(df["head_body_ratio"].dropna().mean(), 2),
    "head_body_max": round(df["head_body_ratio"].dropna().max(), 2),
    "head_body_min": round(df["head_body_ratio"].dropna().min(), 2),
    "upper_lower_mean": round(df["upper_lower_ratio"].dropna().mean(), 2),
    "gram_dist_mean": round(df["gram_style_distance"].mean(), 6),
    "fixed_compensation": round(FIXED_COMPENSATION, 4),
    "hair_ratio_mean": round(hair_ratio_mean, 4),
    "hair_ratio_median": round(hair_ratio_median, 4),
    "chin_anime_count": chin_sources.get("anime_face", 0),
    "bottom_box_count": bottom_sources.get("box_bottom", 0),
}
best = df.loc[df["composite_score"].idxmax()]
summary["best_pw"] = float(best["pw_strength"])
summary["best_cg"] = float(best["cg_strength"])
summary["best_score"] = round(float(best["composite_score"]), 4)

# ==================== 生成网格 + 数据表 + 热力图 ====================
print("[5/6] 生成图片网格、数据表、热力图...")

images_data = []
for i in range(len(img_files)):
    row_data = df.iloc[i]
    images_data.append({
        "index": i, "pw": float(row_data["pw_strength"]), "cg": float(row_data["cg_strength"]),
        "thumb": thumb_b64_list[i], "full": full_b64_list[i],
        "integrity": float(row_data["integrity"]),
        "head_body": row_data["head_body_ratio"] if pd.notna(row_data["head_body_ratio"]) else None,
        "upper_lower": row_data["upper_lower_ratio"] if pd.notna(row_data["upper_lower_ratio"]) else None,
        "gram_dist": float(row_data["gram_style_distance"]),
        "composite": float(row_data["composite_score"]),
        "hair_ratio": float(row_data["hair_ratio"]) if pd.notna(row_data["hair_ratio"]) else None,
        "head_src": row_data["head_source"],
        "chin_src": row_data["chin_source"],
        "bottom_src": row_data["bottom_source"],
    })
images_json = json.dumps(images_data, ensure_ascii=False)

# 网格
grid_html = '<div class="grid-container" id="imageGrid">\n'
grid_html += '<div class="grid-cell grid-header corner"></div>\n'
for cg_col in range(GRID_SIZE):
    grid_html += f'<div class="grid-cell grid-header">cg={round(cg_col*0.1,1)}</div>\n'
for pw_row in range(GRID_SIZE):
    pw_val = round(pw_row * 0.1, 1)
    grid_html += f'<div class="grid-cell grid-header">pw={pw_val}</div>\n'
    for cg_col in range(GRID_SIZE):
        idx = pw_row * GRID_SIZE + cg_col
        if idx < len(img_files) and thumb_b64_list[idx]:
            grid_html += f'<div class="grid-cell img-cell" data-idx="{idx}"><img src="data:image/jpeg;base64,{thumb_b64_list[idx]}"/></div>\n'
        else:
            grid_html += '<div class="grid-cell"></div>\n'
grid_html += '</div>\n'

# 数据表
table_html = """
<h2>📋 完整数据表格（121张）</h2>
<div style="overflow-x:auto; max-height:600px; overflow-y:auto; background:white; border-radius:8px; padding:10px;">
<table style="border-collapse:collapse; width:100%; font-size:13px;">
<thead style="position:sticky; top:0; background:#4a90d9; color:white;">
<tr>
<th style="border:1px solid #ddd; padding:6px;">序号</th>
<th style="border:1px solid #ddd; padding:6px;">pw</th>
<th style="border:1px solid #ddd; padding:6px;">cg</th>
<th style="border:1px solid #ddd; padding:6px;">关键点</th>
<th style="border:1px solid #ddd; padding:6px;">完整性</th>
<th style="border:1px solid #ddd; padding:6px;">头发比</th>
<th style="border:1px solid #ddd; padding:6px;">头身比</th>
<th style="border:1px solid #ddd; padding:6px;">上下比</th>
<th style="border:1px solid #ddd; padding:6px;">Gram距离</th>
<th style="border:1px solid #ddd; padding:6px;">综合评分</th>
<th style="border:1px solid #ddd; padding:6px;">头顶</th>
<th style="border:1px solid #ddd; padding:6px;">下巴</th>
<th style="border:1px solid #ddd; padding:6px;">底部</th>
</tr>
</thead>
<tbody>
"""
for _, row in df.iterrows():
    hbr = f"{row['head_body_ratio']:.2f}" if pd.notna(row['head_body_ratio']) else "N/A"
    ulr = f"{row['upper_lower_ratio']:.2f}" if pd.notna(row['upper_lower_ratio']) else "N/A"
    hr = f"{row['hair_ratio']:.3f}" if pd.notna(row['hair_ratio']) else "N/A"
    hbr_color = ""
    if pd.notna(row['head_body_ratio']):
        if row['head_body_ratio'] >= 8.0: hbr_color = "color:green; font-weight:bold;"
        elif row['head_body_ratio'] < 6.0: hbr_color = "color:red;"
    table_html += f"""<tr style="background:{'#f9f9f9' if row['index']%2==0 else 'white'};">
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{int(row['index'])+1}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{row['pw_strength']:.1f}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{row['cg_strength']:.1f}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{int(row['keypoint_count'])}/17</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{row['integrity']:.4f}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{hr}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center; {hbr_color}">{hbr}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{ulr}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center;">{row['gram_style_distance']:.4f}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center; font-weight:bold;">{row['composite_score']:.4f}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center; font-size:11px;">{row['head_source']}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center; font-size:11px;">{row['chin_source']}</td>
<td style="border:1px solid #ddd; padding:4px; text-align:center; font-size:11px;">{row['bottom_source']}</td>
</tr>"""
table_html += """</tbody></table></div>
<p style="color:#888; font-size:12px; margin-top:8px;">头身比绿色=≥8.0，红色=<6.0。头发比=(人脸框顶-人体框顶)/脸高，仅用于统计，头顶线使用全局固定补偿。</p>
"""

# 热力图
def make_heatmap_data(df, value_col):
    data = []
    for pw_row in range(GRID_SIZE):
        row = []
        for cg_col in range(GRID_SIZE):
            idx = pw_row * GRID_SIZE + cg_col
            val = df[df["index"] == idx][value_col].values
            row.append(float(val[0]) if len(val) > 0 and val[0] is not None and not pd.isna(val[0]) else None)
        data.append(row)
    return data

heatmaps = {
    "integrity": make_heatmap_data(df, "integrity"),
    "head_body_ratio": make_heatmap_data(df, "head_body_ratio"),
    "upper_lower_ratio": make_heatmap_data(df, "upper_lower_ratio"),
    "gram_style_distance": make_heatmap_data(df, "gram_style_distance"),
    "composite": make_heatmap_data(df, "composite_score"),
}
x_labels = json.dumps([round(i*0.1,1) for i in range(11)])
y_labels = json.dumps([round(i*0.1,1) for i in range(11)])

# ==================== 生成HTML ====================
print("[6/6] 生成HTML报告...")

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Lora网格调参可视化报告 v8（固定补偿，均值决定）</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; border-bottom: 3px solid #4a90d9; padding-bottom: 10px; }}
h2 {{ color: #4a90d9; margin-top: 30px; }}
.summary {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }}
.summary-card {{ background: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); min-width: 130px; }}
.summary-card .label {{ color: #888; font-size: 13px; }}
.summary-card .value {{ font-size: 22px; font-weight: bold; color: #4a90d9; }}
.best {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
.best .label {{ color: rgba(255,255,255,0.8); }}
.best .value {{ color: white; font-size: 18px; }}
.grid-container {{ display: grid; grid-template-columns: repeat(12, auto); gap: 2px; background: #ddd; padding: 2px; border-radius: 8px; overflow-x: auto; width: fit-content; }}
.grid-cell {{ background: white; display: flex; align-items: center; justify-content: center; min-width: {THUMB_SIZE}px; min-height: {THUMB_SIZE}px; }}
.grid-cell.img-cell {{ cursor: pointer; transition: transform 0.15s; }}
.grid-cell.img-cell:hover {{ transform: scale(1.08); z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
.grid-cell img {{ max-width: {THUMB_SIZE}px; max-height: {THUMB_SIZE * 1.5}px; }}
.grid-header {{ background: #4a90d9; color: white; font-weight: bold; font-size: 11px; padding: 4px; text-align: center; min-height: 25px; }}
.grid-header.corner {{ background: #357abd; }}
.chart {{ width: 100%; height: 480px; margin: 15px 0; background: white; border-radius: 8px; padding: 10px; }}
.lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }}
.lightbox.active {{ display: flex; }}
.lightbox-img {{ max-height: 70vh; max-width: 90vw; object-fit: contain; border-radius: 4px; }}
.lightbox-info {{ color: white; margin-top: 15px; text-align: center; font-size: 16px; line-height: 1.8; }}
.lightbox-info .param {{ font-size: 20px; font-weight: bold; color: #4a90d9; }}
.lightbox-close {{ position: absolute; top: 20px; right: 30px; color: white; font-size: 40px; cursor: pointer; user-select: none; }}
.lightbox-nav {{ position: absolute; top: 50%; transform: translateY(-50%); color: white; font-size: 50px; cursor: pointer; user-select: none; padding: 20px; opacity: 0.7; }}
.lightbox-nav:hover {{ opacity: 1; }}
.lightbox-prev {{ left: 20px; }}
.lightbox-next {{ right: 20px; }}
.lightbox-hint {{ color: rgba(255,255,255,0.5); font-size: 13px; margin-top: 10px; }}
.method-box {{ background: #e8f4fd; border-left: 4px solid #4a90d9; padding: 15px; margin: 15px 0; border-radius: 4px; }}
</style>
</head>
<body>

<h1>Lora网格调参可视化报告 v8（固定补偿，由全量均值决定）</h1>
<p>完美世界国漫风 Lora × 国漫CG Lora 强度网格调参（11×11 = 121张图）</p>

<div class="method-box">
<strong>计算方法（v8）：</strong><br>
• 第一遍统计全量 hair_ratio 均值 = {summary['hair_ratio_mean']}（中位数 {summary['hair_ratio_median']}）<br>
• 固定补偿 = min(0.25, 均值 × 0.5) = <strong>{summary['fixed_compensation']}</strong>（所有图统一使用）<br>
• 头顶线 = 人脸框顶 - {summary['fixed_compensation']} × 脸高<br>
• 下巴线 = Anime Face Detector 点2（真实检测）— {summary['chin_anime_count']}/121张<br>
• 身高底部 = max(脚踝点y, 人体框底部y) — {summary['bottom_box_count']}/121张用框底兜底<br>
• 头身比 = (底部 - 头顶) / (下巴 - 头顶)
</div>

<h2>📊 总览统计</h2>
<div class="summary">
  <div class="summary-card"><div class="label">总图片数</div><div class="value">{summary['total_images']}</div></div>
  <div class="summary-card"><div class="label">平均完整性</div><div class="value">{summary['integrity_mean']}</div></div>
  <div class="summary-card"><div class="label">完整人体</div><div class="value">{summary['integrity_full_count']}/121</div></div>
  <div class="summary-card"><div class="label">头身比(均/最/最)</div><div class="value">{summary['head_body_mean']}<br><span style="font-size:14px">{summary['head_body_min']}~{summary['head_body_max']}</span></div></div>
  <div class="summary-card"><div class="label">平均上下比</div><div class="value">{summary['upper_lower_mean']}</div></div>
  <div class="summary-card"><div class="label">固定补偿值</div><div class="value">{summary['fixed_compensation']}</div></div>
  <div class="summary-card best"><div class="label">推荐最佳参数</div><div class="value">pw={summary['best_pw']}<br>cg={summary['best_cg']}</div></div>
</div>

<h2>🖼️ 11×11 图片网格（点击查看大图，↑↓纵向/←→横向）</h2>
<p>X轴：国漫CG Lora强度（cg），Y轴：完美世界国漫风 Lora强度（pw）。↑↓键纵向切行，←→键横向切列，ESC关闭。</p>
{grid_html}

{table_html}

<h2>🔥 热力图分析</h2>
<div id="chart-integrity" class="chart"></div>
<div id="chart-headbody" class="chart"></div>
<div id="chart-upperlower" class="chart"></div>
<div id="chart-gram" class="chart"></div>
<div id="chart-composite" class="chart"></div>

<div class="lightbox" id="lightbox">
  <span class="lightbox-close" onclick="closeLightbox()">&times;</span>
  <span class="lightbox-nav lightbox-prev" onclick="navigateDir(-1, 0)">&#10094;</span>
  <img class="lightbox-img" id="lightboxImg" src="" alt=""/>
  <div class="lightbox-info" id="lightboxInfo"></div>
  <div class="lightbox-hint">↑↓ 纵向切行 | ←→ 横向切列 | ESC 关闭 | 点击图片切下一张</div>
  <span class="lightbox-nav lightbox-next" onclick="navigateDir(1, 0)">&#10095;</span>
</div>

<script>
var images = {images_json};
var currentIdx = 0;
var GRID_W = 11;

document.querySelectorAll('.img-cell').forEach(function(cell) {{
    cell.addEventListener('click', function() {{
        openLightbox(parseInt(this.getAttribute('data-idx')));
    }});
}});

function openLightbox(idx) {{
    currentIdx = idx;
    updateLightbox();
    document.getElementById('lightbox').classList.add('active');
}}
function closeLightbox() {{ document.getElementById('lightbox').classList.remove('active'); }}
function navigateDir(dx, dy) {{
    var row = Math.floor(currentIdx / GRID_W);
    var col = currentIdx % GRID_W;
    row = (row + dy + GRID_W) % GRID_W;
    col = (col + dx + GRID_W) % GRID_W;
    currentIdx = row * GRID_W + col;
    updateLightbox();
}}
function navigate(dir) {{ navigateDir(dir > 0 ? 1 : -1, 0); }}
function updateLightbox() {{
    var img = images[currentIdx];
    document.getElementById('lightboxImg').src = 'data:image/jpeg;base64,' + img.full;
    var hb = img.head_body !== null ? img.head_body.toFixed(2) : 'N/A';
    var ul = img.upper_lower !== null ? img.upper_lower.toFixed(2) : 'N/A';
    var hr = img.hair_ratio !== null ? img.hair_ratio.toFixed(3) : 'N/A';
    document.getElementById('lightboxInfo').innerHTML =
        '<span class="param">pw=' + img.pw + ' | cg=' + img.cg + '</span><br>' +
        '位置: 第' + (Math.floor(currentIdx/GRID_W)+1) + '行 第' + ((currentIdx%GRID_W)+1) + '列 (' + (currentIdx+1) + '/' + images.length + ')<br>' +
        '完整性: ' + img.integrity.toFixed(2) + ' | 头身比: ' + hb + ' | 上下比: ' + ul + '<br>' +
        '头发比: ' + hr + ' | 固定补偿: {summary['fixed_compensation']}<br>' +
        'Gram距离: ' + img.gram_dist.toFixed(4) + ' | 综合评分: ' + img.composite.toFixed(4) + '<br>' +
        '头顶: ' + img.head_src + ' | 下巴: ' + img.chin_src + ' | 底部: ' + img.bottom_src;
}}

document.addEventListener('keydown', function(e) {{
    if (!document.getElementById('lightbox').classList.contains('active')) return;
    if (e.key === 'Escape') closeLightbox();
    else if (e.key === 'ArrowUp') {{ e.preventDefault(); navigateDir(0, -1); }}
    else if (e.key === 'ArrowDown') {{ e.preventDefault(); navigateDir(0, 1); }}
    else if (e.key === 'ArrowLeft') {{ e.preventDefault(); navigateDir(-1, 0); }}
    else if (e.key === 'ArrowRight' || e.key === ' ') {{ e.preventDefault(); navigateDir(1, 0); }}
}});
document.getElementById('lightboxImg').addEventListener('click', function() {{ navigate(1); }});

var xLabels = {x_labels};
var yLabels = {y_labels};
function makeHeatmap(domId, title, data, min, max) {{
    var chart = echarts.init(document.getElementById(domId));
    var heatData = [];
    for (var i = 0; i < data.length; i++) {{
        for (var j = 0; j < data[i].length; j++) {{
            if (data[i][j] !== null) heatData.push([j, i, data[i][j]]);
        }}
    }}
    chart.setOption({{
        title: {{ text: title, left: 'center' }},
        tooltip: {{ position: 'top', formatter: function(p) {{
            return 'pw=' + yLabels[p.value[1]] + ', cg=' + xLabels[p.value[0]] + '<br/>值: ' + p.value[2].toFixed(4);
        }}}},
        grid: {{ height: '70%', top: '12%' }},
        xAxis: {{ type: 'category', data: xLabels, name: 'cg强度', splitArea: {{ show: true }} }},
        yAxis: {{ type: 'category', data: yLabels, name: 'pw强度', splitArea: {{ show: true }} }},
        visualMap: {{ min: min, max: max, calculable: true, orient: 'horizontal', left: 'center', bottom: '3%',
            inRange: {{ color: ['#313695','#4575b4','#74add1','#abd9e9','#e0f3f8','#ffffbf','#fee090','#fdae61','#f46d43','#d73027','#a50026'] }} }},
        series: [{{ name: title, type: 'heatmap', data: heatData,
            label: {{ show: true, fontSize: 9, formatter: function(p) {{ return p.value[2] !== null ? p.value[2].toFixed(2) : ''; }} }},
            emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' }} }} }}]
    }});
}}

makeHeatmap('chart-integrity', '人体完整性（关键点/17，越高越好）', {json.dumps(heatmaps['integrity'])}, 0, 1);
makeHeatmap('chart-headbody', '头身比（固定补偿{summary['fixed_compensation']} + AnimeFace下巴 + 框底兜底，目标8.5）', {json.dumps(heatmaps['head_body_ratio'])}, 5, 11);
makeHeatmap('chart-upperlower', '上下半身比例（下/上，目标1.4）', {json.dumps(heatmaps['upper_lower_ratio'])}, 0.8, 2.5);
makeHeatmap('chart-gram', 'Gram风格距离（与平均风格，越小越接近）', {json.dumps(heatmaps['gram_style_distance'])}, 0, {round(max(style_distances),6)});
makeHeatmap('chart-composite', '综合评分（完整性0.3+风格0.25+头身比0.25+上下比0.2）', {json.dumps(heatmaps['composite'])}, 0, 1);
</script>

</body>
</html>"""

html_path = os.path.join(OUTPUT_DIR, "lora_grid_report.html")
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n{'='*60}")
print(f"✅ 报告v8生成完成!")
print(f"  HTML报告: {html_path}")
print(f"  CSV数据: {csv_path}")
print(f"{'='*60}")
print(f"\n统计摘要:")
for k, v in summary.items():
    print(f"  {k}: {v}")

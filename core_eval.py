"""
core_eval.py - 质检核心模块
整合：quality_eval.py（三层质检）+ brisque_worker.py（BRISQUE多进程worker）+ tt.py（质检入口）
"""
"""
轻量级 BRISQUE 多进程 worker 模块
独立于 quality_eval.py，避免 Windows spawn 模式下子进程重复导入 torch/CUDA 等重依赖
"""
import cv2
import numpy as np


def brisque_worker(img_path_str):
    """子进程：单独计算一张图的 BRISQUE 分数"""
    try:
        from brisque import BRISQUE
        img = cv2.imdecode(np.fromfile(img_path_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 999.0
        b = BRISQUE()
        score = b.score(img)
        return round(float(score), 2)
    except Exception:
        return 999.0


# -*- coding: utf-8 -*-
"""
短剧资产三层质检评估模块
架构：第一层 传统IQA画质 → 第二层 CLIP语义对齐 → 第三层 专项硬约束检测
内置统计学能力：批次分布统计、分位数分析、AB显著性检验、SPC统计过程控制
"""
import os
# 修复CLIP网络重试卡顿：使用镜像站、禁用符号链接警告、限制重试次数
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_MAX_RETRIES"] = "1"
# 模型已本地缓存，强制离线避免网络等待
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 将 NVIDIA CUDA 运行时 DLL 加入 PATH（onnxruntime-gpu 需要）
import sys
_sp = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
for _dll_dir in ["nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin", "nvidia/cuda_nvrtc/bin"]:
    _p = os.path.join(_sp, _dll_dir)
    if os.path.isdir(_p) and _p not in os.environ["PATH"]:
        os.environ["PATH"] = _p + os.pathsep + os.environ["PATH"]

import cv2
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from PIL import Image
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================== 全局懒加载模型容器 ====================
_MODELS = {}

# CUDA 设备自动检测（有 GPU 用 GPU，无 GPU 回退 CPU）
try:
    import torch as _torch
    DEVICE = 'cuda' if _torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'
print(f"  [设备] 使用计算设备: {DEVICE}")

# ==================== 道具OCR校验配置 ====================
# 无文字预期的道具，是否开启反向校验（检测到多余文字则判不合格）
ENABLE_PROP_REVERSE_OCR = False


def expand_character_prompt_for_eval(prompt):
    """
    人物评估专用：将单视图提示词扩展为多视图角色设计图描述
    解决"提示词描述特写单视图，但实际是特写+三视图拼接图"导致的CLIP匹配度低问题
    仅在评估阶段使用，不影响生成流程，eval_detail.csv中prompt列保持原值
    """
    prefix = "character design sheet with multiple views (close-up, front, side, back) of the same character: "
    return prefix + prompt


# CLIP提示词精简：去掉通用模板词，保留核心描述，避免77token截断丢失关键信息
_CLIP_GENERIC_PATTERNS = [
    r'\bof\s+Asian\s+style\b', r'\bAsian\s+style\b', r'\bin\s+the\s+New\s+Chinese\s+Style\b',
    r'\bChinese\s+style\b', r'\bWestern\s+style\b', r'\bEuropean\s+style\b',
    r'\b45-degree\s+oblique\s+side\s+view\b', r'\boblique\s+side\s+view\b',
    r'\bside\s+view\b', r'\bfront\s+view\b', r'\bback\s+view\b', r'\bclose-up\b',
    r'\bwhite\s+background\b', r'\bpure\s+white\s+background\b', r'\bon\s+a\s+white\s+background\b',
    r'\b8K\b', r'\b4K\b', r'\bhighly\s+detailed\b', r'\bultra\s+detailed\b',
    r'\bhyper\s+detailed\b', r'\bmasterpiece\b', r'\bbest\s+quality\b',
    r'\bproduct\s+photography\b', r'\bstudio\s+lighting\b', r'\bprofessional\s+lighting\b',
    r'\bdramatic\s+lighting\b', r'\bcinematic\s+lighting\b',
    r'\bno\s+people\b', r'\bno\s+human\b', r'\bno\s+characters\b',
    r'\bif\s+there\s+is\s+text,\s+it\s+must\s+be\s+Chinese\b',
    r'\bthe\s+font\s+must\s+be\s+Chinese\b',
]
_CLIP_GENERIC_RE = re.compile('|'.join(_CLIP_GENERIC_PATTERNS), re.IGNORECASE)


def simplify_prompt_for_clip(prompt):
    """
    道具/场景评估专用：精简提示词，去掉通用模板词，保留核心描述
    解决"提示词含大量通用模板词(Asian style/45-degree view/white background/8K等)，
    CLIP 77token截断后丢失核心描述"导致的匹配度低问题
    仅在评估阶段使用，不影响生成流程
    """
    # 去掉通用模板词
    simplified = _CLIP_GENERIC_RE.sub('', prompt)
    # 清理多余的逗号、空格
    simplified = re.sub(r',\s*,+', ',', simplified)
    simplified = re.sub(r'\s+', ' ', simplified)
    simplified = re.sub(r'^[,\s]+|[,\s]+$', '', simplified)
    return simplified if simplified else prompt


def auto_extract_expected_text(prompt_list, img_filename_list):
    """
    从道具提示词中自动提取预期语种和核心关键词，生成校验字典
    支持中英文提示词，根据风格判断预期语种，从文件名提取核心关键词
    :param prompt_list: 道具提示词列表
    :param img_filename_list: 对应图片文件名列表
    :return: dict {文件名: {"expected_lang": "zh"/"en", "expected_keyword": str/None}}
             无文字预期的道具不加入字典（跳过OCR）
    """
    # 文字存在性关键词（中英文），匹配到则认为该道具预期有文字
    # 收紧：只保留明确提到文字内容的触发词，去掉"with text/displays/显示"等模糊词
    text_trigger_pattern = re.compile(
        r'(?:Chinese text|English text|printed text|handwritten text|text says|text reads|\breads\b|\bsays\b|写着|印有|刻着|内容为|文字为|写有|印着|刻有|inscribed with|engraved with)',
        re.IGNORECASE
    )
    # 风格判断 → 预期语种
    zh_style_pattern = re.compile(r'(?:Asian style|Chinese style|中式|亚洲风格)', re.IGNORECASE)
    en_style_pattern = re.compile(r'(?:Western style|European style|American style|欧式|欧美风格)', re.IGNORECASE)

    expected_map = {}
    for prompt, filename in zip(prompt_list, img_filename_list):
        # 1. 判断是否有文字预期（提示词提到文字 → 触发OCR）
        if not text_trigger_pattern.search(prompt):
            continue  # 无文字预期，跳过OCR

        # 2. 判断预期语种（从风格判断）
        if zh_style_pattern.search(prompt):
            expected_lang = "zh"
        elif en_style_pattern.search(prompt):
            expected_lang = "en"
        else:
            expected_lang = "zh"  # 默认中文

        # 3. 提取核心关键词（从文件名去扩展名）
        expected_keyword = os.path.splitext(filename)[0]
        # 纯英文且过短的文件名不作为有效关键词
        if expected_keyword and re.match(r'^[a-zA-Z0-9_\s]+$', expected_keyword) and len(expected_keyword) < 3:
            expected_keyword = None

        expected_map[filename] = {
            "expected_lang": expected_lang,
            "expected_keyword": expected_keyword
        }
    return expected_map


def _get_brisque():
    """第一层：BRISQUE 无参考画质模型（使用brisque独立库，兼容Python3.12）"""
    if 'brisque' not in _MODELS:
        try:
            from brisque import BRISQUE
            _MODELS['brisque'] = BRISQUE()
        except Exception:
            _MODELS['brisque'] = None
    return _MODELS['brisque']


def _get_clip():
    """第二层：CLIP 图文匹配 + 特征提取"""
    if 'clip' not in _MODELS:
        from transformers import CLIPProcessor, CLIPModel
        model_name = "openai/clip-vit-base-patch32"
        _MODELS['clip_model'] = CLIPModel.from_pretrained(model_name, local_files_only=True)
        _MODELS['clip_model'].to(DEVICE)
        _MODELS['clip_model'].eval()
        _MODELS['clip_processor'] = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    return _MODELS['clip_model'], _MODELS['clip_processor']


# ==================== 性能优化：批量并行函数 ====================
def batch_eval_brisque(img_paths, max_workers=None):
    """
    批量并行计算 BRISQUE 分数（多进程，CPU密集型）
    使用独立的 brisque_worker 模块，避免 Windows spawn 模式下子进程重复导入 torch/CUDA
    :param img_paths: 图片路径列表
    :param max_workers: 进程数，默认=min(4, CPU核心数)
    :return: {路径字符串: brisque_score} 字典
    """
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 2)
    path_strs = [str(p) for p in img_paths]
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(brisque_worker, p): p for p in path_strs}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                results[p] = future.result()
            except Exception:
                results[p] = 999.0
    # 失败重试：多进程返回999.0的图片（可能大图内存不足），在主进程重新计算
    failed = [p for p, s in results.items() if s >= 999.0]
    if failed:
        print(f"  ⚠️ BRISQUE多进程有{len(failed)}张失败，主进程重试中...")
        brisque = _get_brisque()
        for p in failed:
            try:
                import cv2 as _cv2
                img = _cv2.imdecode(np.fromfile(p, dtype=np.uint8), _cv2.IMREAD_COLOR)
                if img is not None and brisque is not None:
                    results[p] = round(float(brisque.score(img)), 2)
            except Exception:
                pass  # 保持999.0
    return results


def batch_eval_clip_alignment(img_paths, prompts, batch_size=8):
    """
    批量 CLIP 图文匹配（GPU batch 推理，替代逐张推理）
    :param img_paths: 图片路径列表
    :param prompts: 对应提示词列表
    :param batch_size: 每批图片数
    :return: [{clip_score, clip_pass, clip_fail_reason}, ...] 列表，顺序与输入一致
    """
    model, processor = _get_clip()
    results = [None] * len(img_paths)

    for start in range(0, len(img_paths), batch_size):
        end = min(start + batch_size, len(img_paths))
        batch_paths = img_paths[start:end]
        batch_prompts = prompts[start:end]

        # 批量加载图片
        images = []
        valid_indices = []
        for idx, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception:
                results[start + idx] = {
                    "clip_score": 0.0, "clip_pass": False,
                    "clip_fail_reason": "图片读取失败"
                }

        if not images:
            continue

        # 批量推理：每张图对应自己的prompt
        # CLIP processor 支持 text 列表和 images 列表，一一对应
        valid_prompts = [batch_prompts[i] for i in valid_indices]
        inputs = processor(
            text=valid_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with _torch.no_grad():
            outputs = model(**inputs)
        # logits_per_image 形状 [batch, batch]，对角线是图-文匹配度
        logits = outputs.logits_per_image.diagonal().detach().cpu().numpy()

        for i, idx in enumerate(valid_indices):
            score = max(0.0, min(1.0, float(logits[i]) / 100.0))
            results[start + idx] = {
                "clip_score": round(score, 4),
                "clip_pass": score >= 0.26,
                "clip_fail_reason": "" if score >= 0.26 else "图文语义匹配度不足"
            }

    return results


def _get_insightface():
    """第三层：人脸检测 + 特征提取"""
    if 'insightface' not in _MODELS:
        try:
            # 先检查模型是否已本地存在，避免触发下载超时
            import os
            model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
            if not os.path.isdir(model_dir):
                print("  ⚠️  未检测到本地 buffalo_l 模型，跳过 InsightFace 相关检测（需手动下载模型）")
                _MODELS['insightface'] = None
                return _MODELS['insightface']
            from insightface.app import FaceAnalysis
            # 驱动升级后支持 CUDA 12.8+，onnxruntime-gpu 可正常使用 GPU
            _providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
            app = FaceAnalysis(name='buffalo_l', providers=_providers)
            app.prepare(ctx_id=0 if DEVICE == 'cuda' else -1, det_size=(640, 640))
            _MODELS['insightface'] = app
        except Exception as e:
            print(f"  ⚠️  InsightFace 模型加载失败: {e}")
            _MODELS['insightface'] = None
    return _MODELS['insightface']


def _get_pose():
    """第三层：人体关键点检测（完整性校验），使用YOLOv8-pose（17关键点）"""
    if 'pose' not in _MODELS:
        try:
            from ultralytics import YOLO
            _base_dir = os.path.dirname(os.path.abspath(__file__))
            _model_path = os.path.join(_base_dir, 'yolov8n-pose.pt')
            if not os.path.exists(_model_path):
                _model_path = 'yolov8n-pose.pt'  # 让ultralytics自动下载到当前目录
            _MODELS['pose'] = YOLO(_model_path)
        except Exception:
            _MODELS['pose'] = None
    return _MODELS['pose']


def _get_ocr():
    """第三层：OCR 文字检测识别（使用rapidocr-onnxruntime，兼容Windows/Python3.12）"""
    if 'ocr' not in _MODELS:
        from rapidocr_onnxruntime import RapidOCR
        _MODELS['ocr'] = RapidOCR()
    return _MODELS['ocr']


# ==================== 通用工具函数 ====================
def _read_image_cv(img_path):
    """兼容中文路径的OpenCV读图"""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def _cosine_sim(a, b):
    """余弦相似度计算"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ==================== 第一层：传统IQA（纯画质粗筛） ====================
def eval_iqa(img_path, precomputed_brisque=None):
    """
    第一层：无参考画质评估
    输出：BRISQUE画质分（越低越好）、拉普拉斯模糊分（越高越清晰）、合格判定
    :param precomputed_brisque: 预计算的BRISQUE分数（批量并行时传入，跳过重复计算）
    """
    result = {
        "brisque_score": 0.0,
        "blur_score": 0.0,
        "iqa_pass": False,
        "iqa_fail_reason": ""
    }
    img = _read_image_cv(img_path)
    if img is None:
        result["iqa_fail_reason"] = "图片读取失败"
        return result

    # 1. BRISQUE 通用画质评分（优先用预计算结果，避免重复计算）
    if precomputed_brisque is not None:
        result["brisque_score"] = precomputed_brisque
    else:
        try:
            brisque = _get_brisque()
            if brisque is not None:
                score = brisque.score(img)
                result["brisque_score"] = round(float(score), 2)
            else:
                result["brisque_score"] = 999.0
        except Exception:
            result["brisque_score"] = 999.0

    # 2. 拉普拉斯方差 模糊检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result["blur_score"] = round(laplacian_var, 2)

    # 合格判定（阈值可根据业务调整）
    if result["brisque_score"] < 45 and result["blur_score"] > 50:
        result["iqa_pass"] = True
    else:
        reasons = []
        if result["brisque_score"] >= 45:
            reasons.append("整体画质偏低")
        if result["blur_score"] <= 50:
            reasons.append("画面模糊失焦")
        result["iqa_fail_reason"] = ";".join(reasons)

    return result


# ==================== 第二层：CLIP 语义对齐层 ====================
def eval_clip_alignment(img_path, prompt):
    """
    第二层：图文语义匹配度
    输出：CLIP得分 0~1，越高匹配度越好
    """
    model, processor = _get_clip()
    image = Image.open(img_path).convert("RGB")
    inputs = processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,  # 开启自动截断
        max_length=77  # 对齐模型原生最大长度
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits_per_image.item()
    # 归一化到 0~1 区间
    score = max(0.0, min(1.0, logits / 100.0))
    return {
        "clip_score": round(score, 4),
        "clip_pass": score >= 0.26,
        "clip_fail_reason": "" if score >= 0.26 else "图文语义匹配度不足"
    }


def _load_image_for_clip(img_input):
    """统一加载图片为PIL RGB格式，支持路径字符串或PIL Image"""
    if isinstance(img_input, Image.Image):
        return img_input.convert("RGB")
    else:
        return Image.open(img_input).convert("RGB")


def eval_group_consistency(img_path_list, threshold=0.82):
    """
    第二层：同组图片一致性（人物三视图/场景四视图通用）
    计算两两CLIP特征余弦相似度的均值，衡量组内风格/内容统一性
    :param img_path_list: 图片路径列表，或PIL Image列表
    :param threshold: 合格阈值，人物/道具默认0.82，场景四视图建议0.65
    """
    model, processor = _get_clip()
    features = []
    for p in img_path_list:
        img = _load_image_for_clip(p)
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        output = model.get_image_features(**inputs)
        # 兼容 transformers >=5.x：get_image_features 返回 BaseModelOutputWithPooling
        if hasattr(output, 'pooler_output'):
            feat = output.pooler_output.detach().cpu().numpy()[0]
        elif hasattr(output, 'last_hidden_state'):
            feat = output.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
        else:
            feat = output.detach().cpu().numpy()[0]
        features.append(feat)

    sims = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sims.append(_cosine_sim(features[i], features[j]))

    mean_sim = np.mean(sims) if sims else 0.0
    return {
        "group_consistency": round(float(mean_sim), 4),
        "group_pass": mean_sim >= threshold,
        "group_fail_reason": "" if mean_sim >= threshold else "组内风格内容一致性不足"
    }


# ==================== 第三层：专项硬约束精检层 ====================
# ---------- 人物专项 ----------
def eval_face_consistency(ref_img_path, gen_img_path):
    """人物：人脸一致性校验（余弦相似度）"""
    app = _get_insightface()

    def get_emb(path):
        img = _read_image_cv(path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # 取面积最大的人脸作为主体
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        return faces[0].embedding

    ref_emb = get_emb(ref_img_path)
    gen_emb = get_emb(gen_img_path)

    if ref_emb is None:
        return {"face_sim": 0.0, "face_pass": False, "face_fail_reason": "基准图未检测到人脸"}
    if gen_emb is None:
        return {"face_sim": 0.0, "face_pass": False, "face_fail_reason": "生成图未检测到人脸"}

    sim = _cosine_sim(ref_emb, gen_emb)
    return {
        "face_sim": round(sim, 4),
        "face_pass": sim >= 0.72,
        "face_fail_reason": "" if sim >= 0.72 else "人脸相似度不足，角色漂移"
    }


def eval_intra_face_consistency(img_path):
    """
    人物：图内人脸一致性（1×4横排拼接图，只比较第1格特写图和第2格正视图的人脸）
    第3、4格为侧视图/后视图，人脸不完整，不参与比对
    分别在前两格检测人脸，取每格面积最大的人脸，计算余弦相似度
    任一格未检测到人脸 → 不合格
    阈值 ≥ 0.72 合格
    """
    app = _get_insightface()
    if app is None:
        return {
            "face_count": None,
            "intra_face_consistency": None,
            "intra_face_pass": None,
            "intra_face_fail_reason": "InsightFace模型不可用，跳过图内人脸一致性检测"
        }
    img = _read_image_cv(img_path)
    if img is None:
        return {
            "face_count": 0,
            "intra_face_consistency": 0.0,
            "intra_face_pass": False,
            "intra_face_fail_reason": "图片读取失败"
        }

    h, w = img.shape[:2]
    # 1×4横排，每格宽度 = w/4，只取前两格
    col_w = w // 4
    region1 = img[:, 0:col_w]           # 第1格：特写图
    region2 = img[:, col_w:2*col_w]     # 第2格：正视图

    def get_largest_face(region):
        faces = app.get(region)
        if not faces:
            return None
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        return faces[0]

    face1 = get_largest_face(region1)
    face2 = get_largest_face(region2)

    if face1 is None and face2 is None:
        return {
            "face_count": 0,
            "intra_face_consistency": 0.0,
            "intra_face_pass": False,
            "intra_face_fail_reason": "特写图和正视图均未检测到人脸"
        }
    if face1 is None:
        return {
            "face_count": 1,
            "intra_face_consistency": 0.0,
            "intra_face_pass": False,
            "intra_face_fail_reason": "第1格特写图未检测到人脸"
        }
    if face2 is None:
        return {
            "face_count": 1,
            "intra_face_consistency": 0.0,
            "intra_face_pass": False,
            "intra_face_fail_reason": "第2格正视图未检测到人脸"
        }

    sim = _cosine_sim(face1.embedding, face2.embedding)
    is_pass = sim >= 0.72

    return {
        "face_count": 2,
        "intra_face_consistency": round(sim, 4),
        "intra_face_pass": is_pass,
        "intra_face_fail_reason": "" if is_pass else "特写图与正视图人脸一致性不足，角色漂移"
    }


def eval_intra_clip_consistency(img_path):
    """
    人物：分区域CLIP风格一致性（单张拼接图裁成4区域，计算CLIP特征两两相似度）
    支持1×4横排布局，自动检测布局
    阈值 ≥ 0.70 合格
    """
    model, processor = _get_clip()
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # 自动判断布局：宽高比>1.5 认为是1×4横排，否则2×2网格
    aspect = w / h
    if aspect > 1.5:
        # 1×4 横排
        region_w = w // 4
        regions = [
            img.crop((i * region_w, 0, (i + 1) * region_w, h))
            for i in range(4)
        ]
    else:
        # 2×2 网格
        half_w, half_h = w // 2, h // 2
        regions = [
            img.crop((0, 0, half_w, half_h)),
            img.crop((half_w, 0, w, half_h)),
            img.crop((0, half_h, half_w, h)),
            img.crop((half_w, half_h, w, h)),
        ]

    # 提取每个区域的CLIP图像特征
    features = []
    for region in regions:
        inputs = processor(images=region, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        output = model.get_image_features(**inputs)
        # 兼容 transformers >=5.x：get_image_features 返回 BaseModelOutputWithPooling
        if hasattr(output, 'pooler_output'):
            feat = output.pooler_output.detach().cpu().numpy()[0]
        elif hasattr(output, 'last_hidden_state'):
            feat = output.last_hidden_state[:, 0, :].detach().cpu().numpy()[0]
        else:
            feat = output.detach().cpu().numpy()[0]
        features.append(feat)

    # 计算4区域两两余弦相似度均值
    sims = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            sims.append(_cosine_sim(features[i], features[j]))

    mean_sim = float(np.mean(sims)) if sims else 0.0
    is_pass = mean_sim >= 0.70

    return {
        "intra_clip_consistency": round(mean_sim, 4),
        "intra_clip_pass": is_pass,
        "intra_clip_fail_reason": "" if is_pass else "分区域风格一致性不足"
    }


def eval_person_integrity(img_path):
    """人物：人体完整性检测（缺肢、边缘截断），使用YOLOv8-pose 17关键点"""
    pose = _get_pose()
    if pose is None:
        return {"integrity_pass": True, "keypoint_count": None,
                "integrity_fail_reason": "人体检测模型不可用，跳过人体完整性检测"}
    img = _read_image_cv(img_path)
    if img is None:
        return {"integrity_pass": False, "keypoint_count": 0, "integrity_fail_reason": "图片读取失败"}

    results = pose(img, verbose=False)
    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        return {"integrity_pass": False, "keypoint_count": 0, "integrity_fail_reason": "未检测到人体"}

    # 遍历所有检测到的人，取可见关键点最多的那个人（兼容多人同框/三视图图片）
    max_visible = 0
    for kps in result.keypoints:
        conf = kps.conf.cpu().numpy().flatten()
        visible = int(np.sum(conf > 0.5))
        if visible > max_visible:
            max_visible = visible

    keypoint_count = max_visible
    # 必须检测到全部17个关键点才算完整，否则不完整
    is_pass = keypoint_count == 17
    return {
        "integrity_pass": is_pass,
        "keypoint_count": keypoint_count,
        "integrity_fail_reason": "" if is_pass else "人体不完整/被边缘截断"
    }


# ---------- 道具专项 ----------
def eval_prop_text(img_path, expected_lang="zh", expected_keyword=None):
    """
    道具：OCR文字校验（语种风格校验 + 核心关键词包含校验）
    :param expected_lang: 预期语种 zh/en
    :param expected_keyword: 核心关键词，None表示不做关键词校验
    """
    ocr = _get_ocr()
    result, _ = ocr(str(img_path))

    if not result:
        detected_text = ""
    else:
        # rapidocr格式: [[box, text, confidence], ...]
        detected_text = "".join([line[1] for line in result])

    # 文字存在性
    has_text = len(detected_text.strip()) > 0

    # ===== 指标1：语种风格校验 =====
    zh_count = sum('\u4e00' <= c <= '\u9fff' for c in detected_text)
    en_count = sum(c.isalpha() and c.isascii() for c in detected_text)
    # 分母只算中英文字符，排除数字和符号（数字不影响语种占比）
    letter_chars = zh_count + en_count

    # 判断识别到的主要语种
    if zh_count > 0 and (en_count == 0 or zh_count >= en_count):
        detected_lang = "zh"
    elif en_count > 0:
        detected_lang = "en"
    else:
        detected_lang = "unknown"

    text_lang_pass = False
    lang_fail_reason = ""
    if expected_lang == "zh":
        zh_ratio = zh_count / letter_chars if letter_chars > 0 else 0
        if zh_count > 0 and zh_ratio >= 0.3:
            text_lang_pass = True
        else:
            lang_fail_reason = "预期中文但中文字符不足"
    elif expected_lang == "en":
        en_ratio = en_count / letter_chars if letter_chars > 0 else 0
        if en_count > 0 and en_ratio >= 0.5:
            text_lang_pass = True
        else:
            lang_fail_reason = "预期英文但英文字符不足"

    # ===== 指标2：核心关键词包含校验 =====
    text_keyword_pass = None
    keyword_fail_reason = ""
    if expected_keyword:
        if expected_keyword.lower() in detected_text.lower():
            text_keyword_pass = True
        else:
            text_keyword_pass = False
            keyword_fail_reason = f"未包含核心关键词'{expected_keyword}'"

    # ===== 综合判断 =====
    # OCR未检测到文字时，所有校验设为None（跳过，不计入合格率分母）
    if not has_text:
        text_lang_pass = None
        text_keyword_pass = None
        text_pass = None
        text_fail_reason = "OCR未检测到文字（可能文字过小/艺术字体，建议人工复核）"
    else:
        keyword_ok = text_keyword_pass if text_keyword_pass is not None else True
        text_pass = text_lang_pass and keyword_ok
        # 综合失败原因
        fail_reasons = []
        if not text_lang_pass:
            fail_reasons.append(lang_fail_reason)
        if text_keyword_pass is False:
            fail_reasons.append(keyword_fail_reason)
        text_fail_reason = ";".join(fail_reasons) if fail_reasons else ""

    return {
        "has_text": has_text,
        "detected_text": detected_text,
        "detected_lang": detected_lang,
        "text_lang_pass": text_lang_pass,
        "lang_fail_reason": lang_fail_reason,
        "expected_keyword": expected_keyword,
        "text_keyword_pass": text_keyword_pass,
        "keyword_fail_reason": keyword_fail_reason,
        "text_pass": text_pass,
        "text_fail_reason": text_fail_reason
    }


# ---------- 场景专项 ----------
def eval_scene_phash_consistency(img_path_list):
    """场景：感知哈希粗一致性校验（同一场景四视图）"""
    import imagehash
    hashes = []
    for p in img_path_list:
        h = imagehash.phash(Image.open(p))
        hashes.append(h)

    diffs = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            diffs.append(hashes[i] - hashes[j])

    avg_diff = np.mean(diffs) if diffs else 999.0
    return {
        "phash_avg_diff": round(float(avg_diff), 2),
        "phash_pass": avg_diff <= 15,
        "phash_fail_reason": "" if avg_diff <= 15 else "场景视图间差异过大"
    }


def _load_image_cv(img_input):
    """统一加载图片为cv2 BGR格式，支持路径字符串或numpy array"""
    if isinstance(img_input, np.ndarray):
        return img_input
    else:
        return _read_image_cv(img_input)


def eval_scene_color_consistency(img_path_list):
    """
    场景：色调一致性（HSV颜色直方图两两相关系数均值）
    对每张场景图转 HSV 色彩空间，计算颜色直方图，归一化
    用 cv2.compareHist(HISTCMP_CORREL) 计算两两直方图相关系数
    多张场景图取均值，阈值 ≥ 0.40 合格（四宫格不同视角内容差异大，0.70过严）
    只有1张图时跳过（单图无组一致性）
    :param img_path_list: 图片路径列表，或numpy array列表（cv2 BGR格式）
    """
    if len(img_path_list) < 2:
        return {
            "scene_color_consistency": None,
            "scene_color_pass": None,
            "scene_color_fail_reason": "单图无组一致性，跳过"
        }

    histograms = []
    for p in img_path_list:
        img = _load_image_cv(p)
        if img is None:
            continue
        # 转 HSV 色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 计算 H-S 二维直方图（粗粒度bins，避免对内容差异过度敏感）
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 15], [0, 180, 0, 256])
        # 归一化
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        histograms.append(hist)

    if len(histograms) < 2:
        return {
            "scene_color_consistency": None,
            "scene_color_pass": None,
            "scene_color_fail_reason": "有效图片不足，跳过"
        }

    # 计算两两直方图相关系数
    corrs = []
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            corr = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CORREL)
            corrs.append(float(corr))

    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    is_pass = mean_corr >= 0.40

    return {
        "scene_color_consistency": round(mean_corr, 4),
        "scene_color_pass": is_pass,
        "scene_color_fail_reason": "" if is_pass else "场景色调一致性不足"
    }


# ==================== 统计学模块 ====================
def calc_stats(score_list, pass_list=None):
    """
    【统计方法1：批次分布统计】
    计算单维度完整描述性统计量：均值、中位数、标准差、P10/P90分位数、极值、合格率
    自动过滤 None 值
    """
    # 过滤空值
    valid_scores = [s for s in score_list if s is not None]
    if not valid_scores:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "p10": None, "p90": None, "min": None, "max": None}

    arr = np.array(valid_scores)
    stats_result = {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }
    if pass_list is not None:
        valid_pass = [p for p in pass_list if p is not None]
        if valid_pass:
            stats_result["pass_rate"] = round(sum(valid_pass) / len(valid_pass) * 100, 2)
        else:
            stats_result["pass_rate"] = None
    return stats_result


def ab_test(group_a_scores, group_b_scores, metric_name="指标"):
    """
    AB测试：曼-惠特尼U检验（非参数检验，稳健）
    用于验证两组方案是否存在统计学显著差异
    """
    a, b = np.array(group_a_scores), np.array(group_b_scores)
    u_stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
    diff = b.mean() - a.mean()
    diff_pct = round((b.mean() - a.mean()) / a.mean() * 100, 2)

    alpha = 0.05
    if p_value < alpha:
        conclusion = "显著差异，B组显著优于A组" if diff > 0 else "显著差异，A组显著优于B组"
    else:
        conclusion = "无统计学显著差异"

    return {
        "A组均值": round(float(a.mean()), 4),
        "B组均值": round(float(b.mean()), 4),
        "A组P10": round(float(np.percentile(a, 10)), 4),
        "B组P10": round(float(np.percentile(b, 10)), 4),
        "绝对提升量": round(float(diff), 4),
        "相对提升": f"{diff_pct}%",
        "p值": round(float(p_value), 4),
        "检验方法": "曼-惠特尼U检验",
        "结论": conclusion
    }


def spc_process_control(history_baseline, current_value, metric_name="指标"):
    """
    【统计方法2：SPC统计过程控制】
    基于3σ原则计算控制限，监控批次质量波动，识别异常批次
    :param history_baseline: list，历史正常批次的该指标值（建议≥5批数据作为基线）
    :param current_value: float，当前批次的该指标值
    :param metric_name: 指标名称
    :return: 控制限、异常判定、异常原因
    """
    arr = np.array(history_baseline)
    cl = float(np.mean(arr))  # 中心线 Center Line
    sigma = float(np.std(arr))  # 过程标准差
    ucl = cl + 3 * sigma  # 上控制限
    lcl = cl - 3 * sigma  # 下控制限

    # 异常判定
    is_abnormal = False
    abnormal_type = []

    if current_value > ucl:
        is_abnormal = True
        abnormal_type.append("超出上控制限，质量异常变好/变差")
    if current_value < lcl:
        is_abnormal = True
        abnormal_type.append("超出下控制限，质量异常变差")

    # 补充：接近控制限预警（±2σ）
    if abs(current_value - cl) > 2 * sigma:
        abnormal_type.append("接近控制限，需关注波动趋势")

    return {
        "metric": metric_name,
        "cl": round(cl, 4),
        "ucl": round(ucl, 4),
        "lcl": round(lcl, 4),
        "current_value": round(current_value, 4),
        "is_abnormal": is_abnormal,
        "warning_info": ";".join(abnormal_type) if abnormal_type else "过程稳定，无异常"
    }


# ==================== 统一批量评估入口 ====================
def batch_evaluate(
        prompt_list,
        image_dir,
        asset_type="character",
        ref_img_path=None,
        expected_text_map=None,
        output_report=True,
        output_dir="./eval_results"
):
    """
    批量评估主入口，直接对接pipe.py输出
    :param prompt_list: 提示词列表，与图片按顺序一一对应
    :param image_dir: 生成图片目录
    :param asset_type: character / prop / scene
    :param ref_img_path: 人物基准图路径（人物一致性必传）
    :param expected_text_map: 道具预期文字字典 {文件名: 预期文字}，留空则自动从提示词提取
    :param output_report: 是否输出CSV报告
    :return: 明细DataFrame + 批次统计汇总 + 组一致性结果
    """
    img_dir = Path(image_dir)
    img_paths = sorted([p for p in img_dir.glob("*")
                        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    n = min(len(prompt_list), len(img_paths))

    # ========== 提示词与图片文件名匹配重排 ==========
    # 提示词列表按剧本顺序，图片按文件名排序，直接zip会错位
    # 解析提示词中的道具名（格式"• 道具名: 提示词"），按文件名重排
    import re as _re
    _name_prompt_map = {}
    for _p in prompt_list:
        if isinstance(_p, str) and ':' in _p:
            _m = _re.match(r'^\s*[•\-\*\d\.\s]*([^:：]+)[:：]', _p)
            if _m:
                _name = _m.group(1).strip()
                _name_prompt_map[_name] = _p
    if _name_prompt_map:
        _reordered = []
        _missed = 0
        for _ip in img_paths[:n]:
            _stem = _ip.stem  # 去扩展名的文件名
            if _stem in _name_prompt_map:
                _reordered.append(_name_prompt_map[_stem])
            else:
                # 模糊匹配：文件名包含道具名或道具名包含文件名
                _found = None
                for _pn, _pp in _name_prompt_map.items():
                    if _stem in _pn or _pn in _stem:
                        _found = _pp
                        break
                if _found:
                    _reordered.append(_found)
                else:
                    _reordered.append(prompt_list[len(_reordered)] if len(_reordered) < len(prompt_list) else "")
                    _missed += 1
        if _missed > 0:
            print(f"  ⚠️  提示词重排：{_missed} 张图片未找到匹配提示词，使用原顺序")
        prompt_list = _reordered

    # ========== 道具类：提前一次性生成预期文字字典 ==========
    if asset_type == "prop" and expected_text_map is None:
        img_filenames = [p.name for p in img_paths]
        expected_text_map = auto_extract_expected_text(prompt_list, img_filenames)

    # ========== 性能优化：批量并行计算 BRISQUE（多进程）和 CLIP（GPU batch） ==========
    eval_img_paths = img_paths[:n]
    print(f"  ⚡ 批量预计算：BRISQUE多进程并行 + CLIP GPU batch推理（{n}张）...")
    # 1. BRISQUE 多进程并行（CPU密集型，最大瓶颈）
    brisque_map = batch_eval_brisque(eval_img_paths)
    # 2. CLIP 图文匹配 GPU batch 推理
    # 人物：扩展多视图前缀；道具/场景：精简通用模板词，避免77token截断丢失核心描述
    clip_prompts = []
    for i in range(n):
        if asset_type == "character":
            clip_prompts.append(expand_character_prompt_for_eval(prompt_list[i]))
        else:
            clip_prompts.append(simplify_prompt_for_clip(prompt_list[i]))
    clip_results = batch_eval_clip_alignment(eval_img_paths, clip_prompts, batch_size=8)
    print(f"  ✅ 批量预计算完成")

    rows = []
    for i in range(n):
        img_path = img_paths[i]
        prompt = prompt_list[i]
        row = {"filename": img_path.name, "prompt": prompt}

        # 第一层：画质检测（BRISQUE用预计算结果，只算拉普拉斯模糊）
        iqa_res = eval_iqa(img_path, precomputed_brisque=brisque_map.get(str(img_path)))
        row.update(iqa_res)

        # 第二层：CLIP语义匹配（用预计算的batch推理结果）
        row.update(clip_results[i])

        # 第三层：专项检测
        if asset_type == "character":
            if ref_img_path:
                face_res = eval_face_consistency(ref_img_path, img_path)
                row.update(face_res)
            integ_res = eval_person_integrity(img_path)
            row.update(integ_res)
            # 新增：图内人脸一致性（拼接图中多个人脸）
            intra_face_res = eval_intra_face_consistency(img_path)
            row.update(intra_face_res)
            # 新增：分区域CLIP风格一致性
            intra_clip_res = eval_intra_clip_consistency(img_path)
            row.update(intra_clip_res)

        elif asset_type == "prop":
            # ========== OCR分流逻辑 ==========
            if expected_text_map and img_path.name in expected_text_map:
                # A类：预期有文字 → 语种校验 + 关键词包含校验
                cfg = expected_text_map[img_path.name]
                text_res = eval_prop_text(
                    img_path,
                    expected_lang=cfg["expected_lang"],
                    expected_keyword=cfg["expected_keyword"]
                )
            else:
                # B类：预期无文字 → 按配置决定是否反向校验
                if ENABLE_PROP_REVERSE_OCR:
                    text_res = eval_prop_text(img_path)
                    # 反向判定：检测到文字反而不合格
                    text_res["text_pass"] = not text_res["has_text"]
                    text_res["text_fail_reason"] = "生成了多余的乱码文字" if text_res["has_text"] else ""
                else:
                    # 默认：直接跳过OCR，标记为不适用
                    text_res = {
                        "has_text": None,
                        "detected_text": "",
                        "detected_lang": None,
                        "text_lang_pass": None,
                        "lang_fail_reason": "",
                        "expected_keyword": None,
                        "text_keyword_pass": None,
                        "keyword_fail_reason": "",
                        "text_pass": True,
                        "text_fail_reason": "无文字校验要求，跳过OCR"
                    }
            row.update(text_res)

        elif asset_type == "scene":
            pass  # 场景组一致性在循环外统一计算

        rows.append(row)

    df = pd.DataFrame(rows)

    # 组一致性计算（多图一组）
    group_res = None
    if asset_type == "scene":
        # 场景：每张拼接图按2×2四宫格裁剪成4张独立视图，再计算一致性
        scene_clip_imgs = []  # PIL Image，用于CLIP
        scene_cv_imgs = []    # numpy array，用于色调直方图
        for p in img_paths[:n]:
            pil_img = Image.open(p).convert("RGB")
            cv_img = _read_image_cv(p)
            w, h = pil_img.size
            half_w, half_h = w // 2, h // 2
            # 2×2四宫格：左上、右上、左下、右下
            crops_pil = [
                pil_img.crop((0, 0, half_w, half_h)),
                pil_img.crop((half_w, 0, w, half_h)),
                pil_img.crop((0, half_h, half_w, h)),
                pil_img.crop((half_w, half_h, w, h)),
            ]
            crops_cv = [
                cv_img[0:half_h, 0:half_w],
                cv_img[0:half_h, half_w:w],
                cv_img[half_h:h, 0:half_w],
                cv_img[half_h:h, half_w:w],
            ]
            scene_clip_imgs.extend(crops_pil)
            scene_cv_imgs.extend(crops_cv)

        # 用裁剪后的四视图计算一致性（即使原始只有1张拼接图，裁剪后也有4张）
        if len(scene_clip_imgs) >= 2:
            group_res = eval_group_consistency(scene_clip_imgs, threshold=0.65)
    elif len(img_paths) > 1:
        # 人物/道具：直接用原图计算组一致性
        group_res = eval_group_consistency([str(p) for p in img_paths])

    # ========== 批次分布统计 ==========
    summary = {}
    if not df.empty:
        summary["画质BRISQUE"] = calc_stats(df["brisque_score"].values)
        summary["CLIP匹配度"] = calc_stats(df["clip_score"].values, df["clip_pass"].values)

        if asset_type == "character" and "face_sim" in df.columns:
            summary["人脸相似度"] = calc_stats(df["face_sim"].values, df["face_pass"].values)
        if asset_type == "character" and "intra_face_consistency" in df.columns:
            summary["图内人脸一致性"] = calc_stats(df["intra_face_consistency"].values, df["intra_face_pass"].values)
        if asset_type == "character" and "intra_clip_consistency" in df.columns:
            summary["分区域CLIP一致性"] = calc_stats(df["intra_clip_consistency"].values, df["intra_clip_pass"].values)
        if asset_type == "character" and "keypoint_count" in df.columns:
            summary["人体关键点"] = calc_stats(df["keypoint_count"].values, df["integrity_pass"].values)
        if asset_type == "prop" and "text_lang_pass" in df.columns:
            # 只统计实际执行了OCR的道具（has_text不为None表示执行了OCR）
            ocr_mask = df["has_text"].notna()
            if ocr_mask.any():
                ocr_df = df[ocr_mask]
                # 语种校验合格率
                lang_pass_rate = round(float(ocr_df["text_lang_pass"].mean()) * 100, 2)
                # 关键词校验合格率（仅对有关键词校验的样本）
                kw_mask = ocr_df["text_keyword_pass"].notna()
                kw_pass_rate = round(float(ocr_df.loc[kw_mask, "text_keyword_pass"].mean()) * 100, 2) if kw_mask.any() else None
                # 综合合格率
                text_pass_rate = round(float(ocr_df["text_pass"].mean()) * 100, 2)

                summary["道具文字-语种校验"] = {
                    "count": len(ocr_df), "pass_rate": lang_pass_rate
                }
                summary["道具文字-关键词校验"] = {
                    "count": int(kw_mask.sum()), "pass_rate": kw_pass_rate
                }
                summary["道具文字-综合"] = {
                    "count": len(ocr_df), "pass_rate": text_pass_rate
                }
            else:
                summary["道具文字-语种校验"] = {"count": 0, "pass_rate": None}
                summary["道具文字-关键词校验"] = {"count": 0, "pass_rate": None}
                summary["道具文字-综合"] = {"count": 0, "pass_rate": None}

    # 输出报告
    if output_report:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(f"{output_dir}/eval_detail.csv", index=False, encoding="utf-8-sig")
        # 汇总统计转成表格保存
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(f"{output_dir}/batch_stats_summary.csv", encoding="utf-8-sig")
        if group_res:
            pd.DataFrame([group_res]).to_csv(
                f"{output_dir}/group_consistency.csv",
                index=False, encoding="utf-8-sig"
            )

    return df, summary, group_res


# ==================== tt.py 质检入口函数 ====================
# -*- coding: utf-8 -*-
"""
单独测试三层质量评估模块
无需运行生成流程，直接读取已有的提示词JSON + 重命名后的图片
"""
import os
import json
import sys
import shutil

# ==================== 配置项（按需修改） ====================
# 生成结果根目录（和你 pipe.py 的输出目录一致）
# 支持环境变量 EVAL_OUTPUT_DIR 覆盖
BASE_OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", "./RunningHub_Outputs")

# 人物基准图路径（做人脸一致性校验，没有则留空或填None）
CHAR_REF_IMG_PATH = "./ref/主角基准图.png"

# 评估报告输出根目录（和主流程保持一致）
EVAL_OUTPUT_DIR = BASE_OUTPUT_DIR

# 快速测试模式：每类资产只取前N张图片（设为None或0则全量评估）
QUICK_TEST = None

# ==================== 读取已有数据 ====================
def load_json_safe(path):
    """安全读取JSON列表"""
    if not os.path.exists(path):
        print(f"⚠️  文件不存在，跳过: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"✅ 读取成功: {path}，共 {len(data)} 条")
            return data
        else:
            print(f"⚠️  JSON 内容不是列表: {path}")
            return []
    except Exception as e:
        print(f"❌ 读取失败 {path}: {e}")
        return []


def main():
    print("=" * 60)
    print(" 三层质量评估 独立测试工具")
    print("=" * 60)
    print(f"数据根目录: {BASE_OUTPUT_DIR}")

    # 1. 读取提示词列表
    print("\n>>> 读取提示词文件...")
    person_prompts = load_json_safe(os.path.join(BASE_OUTPUT_DIR, "人物提示词_.json"))
    prop_prompts = load_json_safe(os.path.join(BASE_OUTPUT_DIR, "道具提示词_.json"))
    scene_prompts = load_json_safe(os.path.join(BASE_OUTPUT_DIR, "场景提示词_.json"))

    # 快速测试模式：截断每类提示词数量
    if QUICK_TEST and QUICK_TEST > 0:
        person_prompts = person_prompts[:QUICK_TEST]
        prop_prompts = prop_prompts[:QUICK_TEST]
        scene_prompts = scene_prompts[:QUICK_TEST]
        print(f"⚡ 快速测试模式：每类只取前 {QUICK_TEST} 张图片")

    # 清除旧的评估报告目录（仅删除报告，不碰图片和提示词）
    import time
    for report_dir_name in ["评估报告_人物", "评估报告_道具", "评估报告_场景"]:
        report_dir = os.path.join(EVAL_OUTPUT_DIR, report_dir_name)
        if os.path.exists(report_dir):
            # 重试3次，应对文件临时被占用的情况
            for attempt in range(3):
                try:
                    shutil.rmtree(report_dir)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        print(f"  ⚠️  无法删除 {report_dir_name}（文件可能被其他程序占用），跳过")
    print("🗑️  已清除旧的评估报告目录")

    # 2. 导入质检模块
    print("\n>>> 加载质检模块...")
    try:
        print("✅ quality_eval 模块加载成功")
    except Exception as e:
        print(f"❌ 质检模块加载失败: {e}")
        print("💡 请确保 quality_eval.py 与本脚本在同一目录下")
        sys.exit(1)

    eval_results = {}

    # ==================== 3. 人物资产评估 ====================
    char_img_dir = os.path.join(BASE_OUTPUT_DIR, '重命名后的图片_人物')
    char_eval_dir = os.path.join(EVAL_OUTPUT_DIR, '评估报告_人物')

    if os.path.exists(char_img_dir) and person_prompts and len(os.listdir(char_img_dir)) > 0:
        print("\n👤 正在评估人物资产...")
        try:
            ref_img = CHAR_REF_IMG_PATH if (CHAR_REF_IMG_PATH and os.path.exists(CHAR_REF_IMG_PATH)) else None
            if not ref_img:
                print("  ⚠️  未找到人物基准图，跳过人脸一致性校验，仅做画质与语义评估")

            df_char, stats_char, group_char = batch_evaluate(
                prompt_list=person_prompts,
                image_dir=char_img_dir,
                asset_type="character",
                ref_img_path=ref_img,
                output_report=True,
                output_dir=char_eval_dir
            )

            eval_results["character"] = {"detail": df_char, "stats": stats_char, "group": group_char}

            print(f"  ✅ 人物评估完成，报告已保存至: {char_eval_dir}")
            if "画质BRISQUE" in stats_char:
                s = stats_char["画质BRISQUE"]
                print(f"     · 画质BRISQUE 均值: {s['mean']}")
            if "CLIP匹配度" in stats_char:
                s = stats_char["CLIP匹配度"]
                print(f"     · CLIP匹配度 均值: {s['mean']}，合格率: {s['pass_rate']}%")
            if "人脸相似度" in stats_char:
                s = stats_char["人脸相似度"]
                print(f"     · 人脸相似度 均值: {s['mean']}，P10: {s['p10']}，合格率: {s['pass_rate']}%")
            if "图内人脸一致性" in stats_char:
                s = stats_char["图内人脸一致性"]
                print(f"     · 图内人脸一致性 均值: {s['mean']}，合格率: {s.get('pass_rate', 'N/A')}%")
            if "分区域CLIP一致性" in stats_char:
                s = stats_char["分区域CLIP一致性"]
                print(f"     · 分区域CLIP一致性 均值: {s['mean']}，合格率: {s.get('pass_rate', 'N/A')}%")
            if "人体关键点" in stats_char:
                s = stats_char["人体关键点"]
                print(f"     · 人体完整性 合格率: {s.get('pass_rate', 'N/A')}%")

        except Exception as e:
            print(f"  ❌ 人物质检执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n👤 人物资产为空或图片目录不存在，跳过评估")

    # ==================== 4. 道具资产评估 ====================
    prop_img_dir = os.path.join(BASE_OUTPUT_DIR, '重命名后的图片_道具')
    prop_eval_dir = os.path.join(EVAL_OUTPUT_DIR, '评估报告_道具')

    if os.path.exists(prop_img_dir) and prop_prompts and len(os.listdir(prop_img_dir)) > 0:
        print("\n📦 正在评估道具资产...")
        try:
            # 留空则自动从提示词提取带文字的道具
            expected_text_map = None

            df_prop, stats_prop, group_prop = batch_evaluate(
                prompt_list=prop_prompts,
                image_dir=prop_img_dir,
                asset_type="prop",
                expected_text_map=expected_text_map,
                output_report=True,
                output_dir=prop_eval_dir
            )

            eval_results["prop"] = {"detail": df_prop, "stats": stats_prop, "group": group_prop}

            print(f"  ✅ 道具评估完成，报告已保存至: {prop_eval_dir}")
            if "画质BRISQUE" in stats_prop:
                s = stats_prop["画质BRISQUE"]
                print(f"     · 画质BRISQUE 均值: {s['mean']}")
            if "CLIP匹配度" in stats_prop:
                s = stats_prop["CLIP匹配度"]
                print(f"     · CLIP匹配度 均值: {s['mean']}，合格率: {s['pass_rate']}%")
            if "道具文字-语种校验" in stats_prop:
                s = stats_prop["道具文字-语种校验"]
                print(f"     · 道具文字-语种校验 合格率: {s.get('pass_rate', 'N/A')}%")
            if "道具文字-关键词校验" in stats_prop:
                s = stats_prop["道具文字-关键词校验"]
                print(f"     · 道具文字-关键词校验 合格率: {s.get('pass_rate', 'N/A')}%")
            if "道具文字-综合" in stats_prop:
                s = stats_prop["道具文字-综合"]
                print(f"     · 道具文字-综合 合格率: {s.get('pass_rate', 'N/A')}%")

        except Exception as e:
            print(f"  ❌ 道具质检执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n📦 道具资产为空或图片目录不存在，跳过评估")

    # ==================== 5. 场景资产评估 ====================
    scene_img_dir = os.path.join(BASE_OUTPUT_DIR, '重命名后的图片_场景')
    scene_eval_dir = os.path.join(EVAL_OUTPUT_DIR, '评估报告_场景')

    if os.path.exists(scene_img_dir) and scene_prompts and len(os.listdir(scene_img_dir)) > 0:
        print("\n🏞️  正在评估场景资产...")
        try:
            df_scene, stats_scene, group_scene = batch_evaluate(
                prompt_list=scene_prompts,
                image_dir=scene_img_dir,
                asset_type="scene",
                output_report=True,
                output_dir=scene_eval_dir
            )

            eval_results["scene"] = {"detail": df_scene, "stats": stats_scene, "group": group_scene}

            print(f"  ✅ 场景评估完成，报告已保存至: {scene_eval_dir}")
            if "画质BRISQUE" in stats_scene:
                s = stats_scene["画质BRISQUE"]
                print(f"     · 画质BRISQUE 均值: {s['mean']}")
            if "CLIP匹配度" in stats_scene:
                s = stats_scene["CLIP匹配度"]
                print(f"     · CLIP匹配度 均值: {s['mean']}，合格率: {s['pass_rate']}%")
            if group_scene and "group_consistency" in group_scene:
                print(f"     · 场景组内一致性: {group_scene['group_consistency']}")

        except Exception as e:
            print(f"  ❌ 场景质检执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n🏞️  场景资产为空或图片目录不存在，跳过评估")

    print("\n" + "=" * 60)
    print("🎉 质量评估测试完成")
    print(f"📊 所有报告已保存至: {BASE_OUTPUT_DIR} 下的「评估报告_*」文件夹")
    print("=" * 60)

    return eval_results



# ==================== tt.py 主入口 ====================
if __name__ == "__main__":
    main()


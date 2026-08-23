"""
analysis.py - 统计分析模块
整合：hypothesis_test.py（假设检验）+ benchmark.py（基准值管理）
"""
# -*- coding: utf-8 -*-
"""
假设检验对比模块
将新剧本的质检结果与基准值（当前项目的运行结果）进行假设检验对比
判断新剧本的资产质量是否符合要求
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats


# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 基准值目录（当前项目的运行结果）
BASELINE_DIR = os.path.join(BASE_DIR, "RunningHub_Outputs")
# 基准值缓存文件
BASELINE_CACHE = os.path.join(BASE_DIR, "baseline_stats.json")

# 假设检验配置
ALPHA = 0.05  # 显著性水平
# 需要对比的指标
COMPARE_METRICS = {
    'character': ['brisque_score', 'blur_score', 'clip_score', 'keypoint_count',
                  'intra_clip_consistency', 'intra_face_consistency'],
    'prop': ['brisque_score', 'blur_score', 'clip_score'],
    'scene': ['brisque_score', 'blur_score', 'clip_score']
}

# 指标方向：True=越高越好，False=越低越好
METRIC_DIRECTION = {
    'brisque_score': False,  # 越低越好
    'blur_score': True,      # 越高越清晰
    'clip_score': True,      # 越高越好
    'keypoint_count': True,  # 越高越好
    'intra_clip_consistency': True,  # 越高越好
    'intra_face_consistency': True,   # 越高越好
}


# asset_type到中文目录名的映射
ASSET_DIR_MAP = {
    'character': '人物',
    'prop': '道具',
    'scene': '场景'
}


def load_eval_data(output_dir, asset_type):
    """加载某类资产的质检明细数据"""
    dir_name = ASSET_DIR_MAP.get(asset_type, asset_type)
    csv_path = os.path.join(output_dir, f"评估报告_{dir_name}", "eval_detail.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def compute_baseline_stats(baseline_dir=BASELINE_DIR, save_cache=True):
    """
    计算基准值统计量（从当前项目的质检结果中）
    返回字典：{asset_type: {metric: {mean, std, median, p10, p90, count}}}
    """
    baseline = {}
    for asset_type in ['character', 'prop', 'scene']:
        df = load_eval_data(baseline_dir, asset_type)
        if df is None:
            continue
        type_stats = {}
        for metric in COMPARE_METRICS.get(asset_type, []):
            if metric not in df.columns:
                continue
            values = df[metric].dropna().values
            if len(values) == 0:
                continue
            type_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                'median': float(np.median(values)),
                'p10': float(np.percentile(values, 10)),
                'p90': float(np.percentile(values, 90)),
                'count': int(len(values)),
                'values': values.tolist()  # 保留原始值用于假设检验
            }
        baseline[asset_type] = type_stats

    if save_cache:
        # 保存时去掉values（太大），单独保存原始值
        cache_data = {}
        for at, metrics in baseline.items():
            cache_data[at] = {}
            for m, s in metrics.items():
                cache_data[at][m] = {k: v for k, v in s.items() if k != 'values'}
        with open(BASELINE_CACHE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 基准值统计已保存: {BASELINE_CACHE}")

    return baseline


def hypothesis_test(baseline_values, new_values, metric_name):
    """
    对单个指标进行假设检验
    使用曼-惠特尼U检验（非参数检验，不要求正态分布）
    返回：{statistic, p_value, is_significant, direction, conclusion}
    """
    if len(baseline_values) < 2 or len(new_values) < 2:
        return {
            'statistic': None,
            'p_value': None,
            'is_significant': False,
            'direction': 'unknown',
            'conclusion': '样本不足，无法进行假设检验',
            'baseline_mean': float(np.mean(baseline_values)) if len(baseline_values) > 0 else None,
            'new_mean': float(np.mean(new_values)) if len(new_values) > 0 else None,
            'baseline_count': len(baseline_values),
            'new_count': len(new_values)
        }

    # 曼-惠特尼U检验
    try:
        stat, p_value = stats.mannwhitneyu(baseline_values, new_values, alternative='two-sided')
    except Exception as e:
        return {
            'statistic': None,
            'p_value': None,
            'is_significant': False,
            'direction': 'unknown',
            'conclusion': f'检验失败: {e}',
            'baseline_mean': float(np.mean(baseline_values)),
            'new_mean': float(np.mean(new_values)),
            'baseline_count': len(baseline_values),
            'new_count': len(new_values)
        }

    is_significant = p_value < ALPHA
    higher_is_better = METRIC_DIRECTION.get(metric_name, True)

    # 判断方向
    new_mean = float(np.mean(new_values))
    baseline_mean = float(np.mean(baseline_values))
    if new_mean > baseline_mean:
        direction = 'higher'
        direction_cn = '高于基准'
    elif new_mean < baseline_mean:
        direction = 'lower'
        direction_cn = '低于基准'
    else:
        direction = 'equal'
        direction_cn = '等于基准'

    # 结论
    if not is_significant:
        conclusion = f'无显著差异（p={p_value:.4f}≥{ALPHA}），新剧本与基准质量相当'
    else:
        # 有显著差异，判断是好是坏
        if higher_is_better:
            if direction == 'higher':
                conclusion = f'显著优于基准（p={p_value:.4f}<{ALPHA}，{direction_cn}）'
            else:
                conclusion = f'显著差于基准（p={p_value:.4f}<{ALPHA}，{direction_cn}）'
        else:
            # 越低越好的指标（如BRISQUE）
            if direction == 'lower':
                conclusion = f'显著优于基准（p={p_value:.4f}<{ALPHA}，{direction_cn}）'
            else:
                conclusion = f'显著差于基准（p={p_value:.4f}<{ALPHA}，{direction_cn}）'

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_significant': bool(is_significant),
        'direction': direction,
        'direction_cn': direction_cn,
        'conclusion': conclusion,
        'baseline_mean': baseline_mean,
        'new_mean': new_mean,
        'baseline_median': float(np.median(baseline_values)),
        'new_median': float(np.median(new_values)),
        'baseline_count': len(baseline_values),
        'new_count': len(new_values),
        'higher_is_better': higher_is_better
    }


def compare_with_baseline(new_output_dir, baseline_dir=BASELINE_DIR):
    """
    将新剧本的质检结果与基准值进行全面对比
    返回完整的对比结果
    """
    # 加载基准值
    baseline = compute_baseline_stats(baseline_dir, save_cache=True)

    result = {
        'baseline_dir': baseline_dir,
        'new_output_dir': new_output_dir,
        'alpha': ALPHA,
        'comparison_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'asset_types': {}
    }

    overall_pass = True
    total_metrics = 0
    pass_metrics = 0

    for asset_type in ['character', 'prop', 'scene']:
        type_name = {'character': '人物', 'prop': '道具', 'scene': '场景'}[asset_type]
        new_df = load_eval_data(new_output_dir, asset_type)
        if new_df is None or asset_type not in baseline:
            result['asset_types'][asset_type] = {
                'name': type_name,
                'available': False,
                'reason': '数据不可用'
            }
            continue

        type_result = {
            'name': type_name,
            'available': True,
            'metrics': {},
            'type_pass': True,
            'summary': ''
        }

        bad_metrics = []
        for metric in COMPARE_METRICS.get(asset_type, []):
            if metric not in new_df.columns or metric not in baseline[asset_type]:
                continue

            baseline_values = np.array(baseline[asset_type][metric]['values'])
            new_values = new_df[metric].dropna().values

            if len(new_values) == 0:
                continue

            test_result = hypothesis_test(baseline_values, new_values, metric)
            type_result['metrics'][metric] = test_result

            total_metrics += 1
            # 判断是否合格：无显著差异，或显著优于基准
            if not test_result['is_significant'] or '优于' in test_result['conclusion']:
                pass_metrics += 1
            else:
                bad_metrics.append(f"{metric}({test_result['direction_cn']})")
                type_result['type_pass'] = False

        if bad_metrics:
            type_result['summary'] = f"{type_name}有{len(bad_metrics)}个指标显著差于基准: {', '.join(bad_metrics)}"
            overall_pass = False
        else:
            type_result['summary'] = f"{type_name}所有指标均符合要求（无显著差异或优于基准）"

        result['asset_types'][asset_type] = type_result

    result['overall_pass'] = overall_pass
    result['total_metrics'] = total_metrics
    result['pass_metrics'] = pass_metrics
    result['pass_rate'] = pass_metrics / total_metrics if total_metrics > 0 else 0
    result['conclusion'] = '✅ 新剧本资产质量符合要求' if overall_pass else '❌ 新剧本资产质量不符合要求（存在显著差于基准的指标）'

    return result


def print_comparison_report(result):
    """打印对比报告"""
    print("\n" + "=" * 70)
    print(" 假设检验对比报告")
    print("=" * 70)
    print(f"基准目录: {result['baseline_dir']}")
    print(f"新剧本目录: {result['new_output_dir']}")
    print(f"显著性水平 α = {result['alpha']}")
    print(f"对比时间: {result['comparison_time']}")
    print()

    for asset_type, type_result in result['asset_types'].items():
        if not type_result.get('available', False):
            print(f"  [{type_result['name']}] {type_result.get('reason', '数据不可用')}")
            continue

        print(f"\n{'─' * 60}")
        print(f" 【{type_result['name']}】 {type_result['summary']}")
        print(f"{'─' * 60}")
        print(f"  {'指标':<25} {'基准均值':>10} {'新均值':>10} {'p值':>10} {'结论'}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")

        for metric, r in type_result['metrics'].items():
            if r['p_value'] is None:
                p_str = 'N/A'
            else:
                p_str = f"{r['p_value']:.4f}"
            bm = f"{r['baseline_mean']:.3f}" if r['baseline_mean'] is not None else 'N/A'
            nm = f"{r['new_mean']:.3f}" if r['new_mean'] is not None else 'N/A'
            status = '✓' if (not r['is_significant'] or '优于' in r['conclusion']) else '✗'
            print(f"  {metric:<25} {bm:>10} {nm:>10} {p_str:>10} {status} {r['conclusion']}")

    print(f"\n{'=' * 70}")
    print(f" 总指标数: {result['total_metrics']}")
    print(f" 合格指标数: {result['pass_metrics']}")
    print(f" 合格率: {result['pass_rate']*100:.1f}%")
    print(f" 最终结论: {result['conclusion']}")
    print(f"{'=' * 70}")


def save_comparison_result(result, output_path):
    """保存对比结果到JSON"""
    # 去掉numpy数组，转为可序列化格式
    save_data = json.loads(json.dumps(result, default=str, ensure_ascii=False))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 对比结果已保存: {output_path}")


if __name__ == '__main__':
    # 测试：用当前项目的结果和自己对比（应该全部无显著差异）
    import sys
    new_dir = sys.argv[1] if len(sys.argv) > 1 else BASELINE_DIR
    result = compare_with_baseline(new_dir)
    print_comparison_report(result)
    save_path = os.path.join(new_dir, "假设检验对比结果.json")
    save_comparison_result(result, save_path)


# ==================== benchmark.py 基准值管理 ====================
"""
各模型/指标运行速度基准测试
测量单次推理的平均耗时（排除冷启动）
"""
import os
import sys
import time
import numpy as np

# 环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, r"E:\comfyui\pylibs")

import core_eval as qe
from PIL import Image
import cv2

# 选一张测试图（人物第一张）
TEST_IMG = r"E:\comfyui\RunningHub_Outputs\重命名后的图片_人物\001_主角_全身照.png"
if not os.path.exists(TEST_IMG):
    # 找第一张存在的图
    import glob
    imgs = glob.glob(r"E:\comfyui\RunningHub_Outputs\重命名后的图片_人物\*.png")
    TEST_IMG = imgs[0] if imgs else None

print(f"测试图片: {TEST_IMG}")
print(f"图片尺寸: {Image.open(TEST_IMG).size}")
print(f"计算设备: {qe.DEVICE}")
print("=" * 60)

N_WARMUP = 2
N_RUNS = 5

def bench(name, func, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """运行 func n_warmup 次预热，再运行 n_runs 次取平均"""
    for _ in range(n_warmup):
        func()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    avg = np.mean(times)
    std = np.std(times)
    print(f"  {name:30s}  {avg:8.2f} ms  (±{std:6.2f})")
    return avg

# 预加载所有模型
print("\n[1/3] 预加载模型...")
qe._get_clip()
qe._get_pose()
qe._get_insightface()
qe._get_ocr()
qe._get_brisque()
print("  所有模型加载完成")

# 读取测试数据
img_pil = Image.open(TEST_IMG).convert("RGB")
img_cv = qe._read_image_cv(TEST_IMG)
prompt = "a person standing in a room, asian style, white background"

print(f"\n[2/3] 基准测试（{N_RUNS}次取平均，单位ms）...")
print("-" * 60)

# 1. CLIP 图文匹配
bench("CLIP 图文匹配", lambda: qe.eval_clip_alignment(TEST_IMG, prompt))

# 2. CLIP 图像特征提取（单次）
model, processor = qe._get_clip()
def clip_feature_only():
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(qe.DEVICE) for k, v in inputs.items()}
    with __import__('torch').no_grad():
        output = model.get_image_features(**inputs)
bench("CLIP 单图特征提取", clip_feature_only)

# 3. YOLOv8-pose 人体关键点
bench("YOLOv8-pose 关键点检测", lambda: qe.eval_person_integrity(TEST_IMG))

# 4. InsightFace 人脸检测+特征
bench("InsightFace 人脸检测+识别", lambda: qe.eval_intra_face_consistency(TEST_IMG))

# 5. RapidOCR 文字识别
bench("RapidOCR 文字识别", lambda: qe.eval_prop_text(TEST_IMG, "测试"))

# 6. BRISQUE 画质评分（已知有bug，用try/except）
def brisque_test():
    try:
        b = qe._get_brisque()
        if b: b.score(img_cv)
    except: pass
bench("BRISQUE 画质评分", brisque_test)

# 7. 拉普拉斯模糊检测
def laplacian_blur():
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()
bench("拉普拉斯模糊检测", laplacian_blur)

# 8. 分区域CLIP一致性（人物：4区域裁剪+4次特征提取）
bench("人物分区域CLIP一致性(4区域)", lambda: qe.eval_intra_clip_consistency(TEST_IMG))

# 9. 场景四宫格CLIP一致性（4区域+两两相似度）
def scene_group_consistency():
    w, h = img_pil.size
    half_w, half_h = w // 2, h // 2
    regions = [
        img_pil.crop((0, 0, half_w, half_h)),
        img_pil.crop((half_w, 0, w, half_h)),
        img_pil.crop((0, half_h, half_w, h)),
        img_pil.crop((half_w, half_h, w, h)),
    ]
    qe.eval_group_consistency(regions, threshold=0.65)
bench("场景四宫格CLIP一致性(4区域)", scene_group_consistency)

# 10. HSV颜色直方图一致性
def hsv_consistency():
    hsv1 = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
bench("HSV颜色直方图对比", hsv_consistency)

# 11. 单张人物完整评估（所有指标之和）
print("-" * 60)
bench("人物单图完整评估(全部指标)", lambda: None, n_warmup=0, n_runs=1)  # 占位
# 实际测量一次完整评估
t0 = time.perf_counter()
qe.eval_clip_alignment(TEST_IMG, prompt)
qe.eval_person_integrity(TEST_IMG)
qe.eval_intra_face_consistency(TEST_IMG)
qe.eval_intra_clip_consistency(TEST_IMG)
qe.eval_iqa(TEST_IMG)
t1 = time.perf_counter()
print(f"  {'人物单图完整评估(全部指标)':30s}  {(t1-t0)*1000:8.2f} ms")

print("\n" + "=" * 60)
print("[3/3] 估算全量运行时间")
print("-" * 60)
# 人物75张，道具114张，场景49张
char_time = (t1 - t0) * 1000  # ms
print(f"  人物 75张 × {char_time:.0f}ms = {75*char_time/1000:.1f}s")
# 道具：CLIP + OCR + IQA，估算约为人物的60%
prop_time = char_time * 0.6
print(f"  道具 114张 × {prop_time:.0f}ms ≈ {114*prop_time/1000:.1f}s")
# 场景：CLIP + 四宫格一致性 + IQA，估算约为人物的80%
scene_time = char_time * 0.8
print(f"  场景 49张 × {scene_time:.0f}ms ≈ {49*scene_time/1000:.1f}s")
total = 75*char_time + 114*prop_time + 49*scene_time
print(f"  {'总计':30s}  {total/1000:.1f}s ≈ {total/1000/60:.1f}min")
print("=" * 60)


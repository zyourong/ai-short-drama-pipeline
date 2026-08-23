# -*- coding: utf-8 -*-
"""
短剧资产质检返工模块
针对ComfyUI生成本身的小概率随机问题（如人体关键点缺失），自动重新生成并质检。
目前实现：人体完整性返工（keypoint_count < 17）
"""
import os
import sys
import time
import json
import re
import zipfile
import shutil
import requests
import pandas as pd

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("EVAL_OUTPUT_DIR", os.path.join(BASE_DIR, "RunningHub_Outputs"))
CHAR_IMG_DIR = os.path.join(OUTPUT_DIR, "重命名后的图片_人物")
CHAR_EVAL_DIR = os.path.join(OUTPUT_DIR, "评估报告_人物")
CHAR_PROMPT_JSON = os.path.join(OUTPUT_DIR, "人物提示词_.json")
EVAL_DETAIL_CSV = os.path.join(CHAR_EVAL_DIR, "eval_detail.csv")

# RunningHub API配置（与pipe.py一致）
WORKFLOW_CHAR = "2086686775652737025"
CHAR_NODE_CONFIG = {"nodeId": "103", "fieldName": "String"}
BASE_URL = "https://www.runninghub.cn"
CREATE_URL = f"{BASE_URL}/task/openapi/create"
QUERY_URL = f"{BASE_URL}/openapi/v2/query"
POLL_INTERVAL = 15      # 轮询间隔（秒）
TIMEOUT_SECONDS = 1800  # 单张返工超时30分钟
MAX_RETRY = 3            # 最大返工重试次数

# 人体完整性阈值
INTEGRITY_THRESHOLD = 17  # 必须检测到全部17个关键点

# 临时目录
TEMP_DIR = os.path.join(BASE_DIR, "rework_temp")
# 返工日志文件
REWORK_LOG_FILE = os.path.join(OUTPUT_DIR, "返工日志.json")


def load_api_key():
    """从本地文件读取API密钥"""
    key_file = os.path.join(BASE_DIR, "API密钥.txt")
    if not os.path.exists(key_file):
        print(f"❌ 未找到API密钥文件: {key_file}")
        sys.exit(1)
    with open(key_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # 提取密钥（支持"API密钥：xxx"格式）
    match = re.search(r'([a-f0-9]{32})', content, re.IGNORECASE)
    if match:
        return match.group(1)
    # 尝试按冒号分割
    if '：' in content:
        return content.split('：')[-1].strip()
    if ':' in content:
        return content.split(':')[-1].strip()
    return content


def load_character_prompts():
    """加载人物提示词，返回 {角色名: 提示词} 字典"""
    if not os.path.exists(CHAR_PROMPT_JSON):
        print(f"❌ 未找到人物提示词文件: {CHAR_PROMPT_JSON}")
        return {}
    with open(CHAR_PROMPT_JSON, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    prompt_map = {}
    for item in prompts:
        if isinstance(item, str) and ':' in item:
            name = item.split(':')[0].strip().lstrip('•').strip()
            text = item[item.index(':')+1:].strip()
            prompt_map[name] = text
    return prompt_map


def find_incomplete_characters():
    """从质检结果中筛选人体不完整的人物图片"""
    if not os.path.exists(EVAL_DETAIL_CSV):
        print(f"❌ 未找到质检结果文件: {EVAL_DETAIL_CSV}")
        return []
    df = pd.read_csv(EVAL_DETAIL_CSV)
    # 筛选人体不完整的（integrity_pass == False 或 keypoint_count < 17）
    incomplete = df[
        (df['integrity_pass'] == False) |
        (df['keypoint_count'] < INTEGRITY_THRESHOLD)
    ]
    result = []
    for _, row in incomplete.iterrows():
        result.append({
            'filename': row['filename'],
            'keypoint_count': int(row['keypoint_count']) if pd.notna(row['keypoint_count']) else 0,
            'prompt': row.get('prompt', '')
        })
    return result


def submit_single_character(api_key, prompt_text):
    """提交单个人物提示词到RunningHub，返回task_id"""
    headers = {
        "Host": "www.runninghub.cn",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "apiKey": api_key,
        "workflowId": WORKFLOW_CHAR,
        "addMetadata": False,
        "nodeInfoList": [{
            "nodeId": CHAR_NODE_CONFIG["nodeId"],
            "fieldName": CHAR_NODE_CONFIG["fieldName"],
            "fieldValue": prompt_text
        }]
    }
    resp = requests.post(CREATE_URL, headers=headers, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    resp_json = resp.json()
    if not resp_json.get("data") or not resp_json["data"].get("taskId"):
        raise Exception(f"提交任务失败: {resp_json}")
    return str(resp_json["data"]["taskId"])


def query_task(api_key, task_id):
    """查询任务状态，返回 (status, file_list)"""
    headers = {
        "Host": "www.runninghub.cn",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"taskId": task_id}
    resp = requests.post(QUERY_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    if 'code' in raw:
        if raw.get('code') != 0:
            raise Exception(f"查询失败: {raw.get('msg')}")
        data = raw.get('data')
        if isinstance(data, dict):
            status = data.get('taskStatus') or data.get('status')
            file_list = data.get('data') or data.get('results')
        elif isinstance(data, list):
            status = 'SUCCESS'
            file_list = data
        else:
            status = raw.get('status')
            file_list = raw.get('results') or raw.get('data')
    else:
        status = raw.get('status') or raw.get('taskStatus')
        file_list = raw.get('results') or raw.get('data')
    # 标准化file_list
    if isinstance(file_list, list):
        for item in file_list:
            if 'fileUrl' in item and 'url' not in item:
                item['url'] = item['fileUrl']
        file_list = [item for item in file_list if item.get('url')]
    else:
        file_list = [] if status == 'SUCCESS' else None
    return status, file_list


def wait_and_download(api_key, task_id, save_dir):
    """等待任务完成并下载结果，返回下载的图片路径列表"""
    os.makedirs(save_dir, exist_ok=True)
    start_time = time.time()
    while True:
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise TimeoutError(f"任务 {task_id} 超时")
        time.sleep(POLL_INTERVAL)
        status, file_list = query_task(api_key, task_id)
        elapsed = int(time.time() - start_time)
        print(f"    ⏳ 状态: {status}，已等待 {elapsed} 秒")
        if status == "SUCCESS":
            break
        elif status == "FAILED":
            raise Exception(f"任务 {task_id} 失败")
    # 下载结果
    if not file_list:
        raise Exception(f"任务 {task_id} 成功但无文件列表")
    # 查找ZIP或图片文件
    dl_url = None
    for item in file_list:
        out_type = item.get('outputType') or item.get('fileType', '')
        url = item.get('url')
        if out_type == 'zip' or (url and url.endswith('.zip')):
            dl_url = url
            break
    if not dl_url:
        # 直接下载图片
        img_paths = []
        for i, item in enumerate(file_list):
            url = item.get('url')
            if url and (url.endswith('.png') or url.endswith('.jpg')):
                ext = '.png' if url.endswith('.png') else '.jpg'
                img_path = os.path.join(save_dir, f"result_{i}{ext}")
                with requests.get(url, stream=True, timeout=120) as r:
                    with open(img_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                img_paths.append(img_path)
        return img_paths
    # 下载ZIP并解压
    zip_path = os.path.join(save_dir, "result.zip")
    with requests.get(dl_url, stream=True, timeout=120) as r:
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    # 解压
    extract_dir = os.path.join(save_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    # 收集图片
    img_paths = []
    for root, dirs, files in os.walk(extract_dir):
        for fn in files:
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, fn))
    return img_paths


def load_rework_log():
    """加载返工日志"""
    if os.path.exists(REWORK_LOG_FILE):
        with open(REWORK_LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"records": [], "total_reworks": 0, "success_count": 0, "fail_count": 0}


def save_rework_log(log_data):
    """保存返工日志"""
    os.makedirs(os.path.dirname(REWORK_LOG_FILE), exist_ok=True)
    with open(REWORK_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def check_integrity(img_path):
    """检查单张图片的人体完整性，返回 (keypoint_count, is_pass)"""
    # 动态导入quality_eval（避免循环导入）
    sys.path.insert(0, BASE_DIR)
    # 设置PYTHONPATH以支持GPU torch
    pylibs = os.path.join(BASE_DIR, "pylibs")
    if os.path.exists(pylibs):
        sys.path.insert(0, pylibs)
    from core_eval import eval_person_integrity
    result = eval_person_integrity(img_path)
    kp = result.get('keypoint_count', 0)
    is_pass = result.get('integrity_pass', False)
    return kp, is_pass


def rework_character(api_key, char_name, prompt_text, original_path, original_kp=0):
    """返工单个人物图片，返回 (是否成功, 最终关键点数量, 返工记录)"""
    print(f"\n{'='*60}")
    print(f"🔄 返工: {char_name}")
    print(f"   原图片: {original_path}")
    print(f"   原始关键点: {original_kp}/{INTEGRITY_THRESHOLD}")
    print(f"{'='*60}")

    rework_record = {
        "char_name": char_name,
        "filename": os.path.basename(original_path),
        "original_path": original_path,
        "original_keypoint_count": original_kp,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attempts": [],
        "final_success": False,
        "final_keypoint_count": None,
        "end_time": None
    }

    for attempt in range(1, MAX_RETRY + 1):
        attempt_info = {
            "attempt": attempt,
            "task_id": None,
            "submit_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keypoint_count": None,
            "success": False,
            "error": None
        }
        print(f"\n  📤 第 {attempt}/{MAX_RETRY} 次尝试提交生成...")
        try:
            task_id = submit_single_character(api_key, prompt_text)
            attempt_info["task_id"] = task_id
            print(f"  ✅ 任务已提交: {task_id}")

            # 等待并下载
            save_dir = os.path.join(TEMP_DIR, char_name, f"attempt_{attempt}")
            img_paths = wait_and_download(api_key, task_id, save_dir)

            if not img_paths:
                print(f"  ⚠️  未下载到图片，重试")
                attempt_info["error"] = "未下载到图片"
                rework_record["attempts"].append(attempt_info)
                continue

            # 取第一张图片进行质检（单提示词应该只有一张）
            new_img = img_paths[0]
            print(f"  📷 新图片: {new_img}")

            # 质检
            print(f"  🔍 正在质检人体完整性...")
            kp, is_pass = check_integrity(new_img)
            attempt_info["keypoint_count"] = kp
            attempt_info["success"] = is_pass
            print(f"  📊 关键点数量: {kp}/{INTEGRITY_THRESHOLD}，{'✅ 合格' if is_pass else '❌ 不合格'}")

            if is_pass:
                # 覆盖原图片
                shutil.copy2(new_img, original_path)
                print(f"  ✅ 返工成功！已覆盖原图片: {original_path}")
                rework_record["attempts"].append(attempt_info)
                rework_record["final_success"] = True
                rework_record["final_keypoint_count"] = kp
                rework_record["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                return True, kp, rework_record
            else:
                print(f"  ❌ 仍不完整（{kp}个关键点），准备重试...")

        except Exception as e:
            print(f"  ❌ 第 {attempt} 次尝试失败: {e}")
            attempt_info["error"] = str(e)
            import traceback
            traceback.print_exc()

        rework_record["attempts"].append(attempt_info)

    print(f"\n  ❌ {char_name} 经过 {MAX_RETRY} 次返工仍未通过，保留原图片")
    rework_record["final_success"] = False
    rework_record["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return False, None, rework_record


def main():
    print("=" * 60)
    print(" 短剧资产质检返工模块 - 人体完整性")
    print("=" * 60)

    # 1. 加载API密钥
    api_key = load_api_key()
    print(f"✅ API密钥已加载")

    # 2. 加载人物提示词
    prompt_map = load_character_prompts()
    print(f"✅ 已加载 {len(prompt_map)} 条人物提示词")

    # 3. 查找人体不完整的图片
    incomplete = find_incomplete_characters()
    if not incomplete:
        print("\n✅ 所有人像人体完整性均合格，无需返工")
        return 0  # 返回成功返工数量

    print(f"\n⚠️  发现 {len(incomplete)} 张人体不完整的图片:")
    for item in incomplete:
        print(f"  - {item['filename']} (关键点: {item['keypoint_count']}/{INTEGRITY_THRESHOLD})")

    # 4. 加载返工日志
    log_data = load_rework_log()

    # 5. 逐个返工
    os.makedirs(TEMP_DIR, exist_ok=True)
    success_count = 0
    fail_list = []

    for item in incomplete:
        filename = item['filename']
        original_path = os.path.join(CHAR_IMG_DIR, filename)
        if not os.path.exists(original_path):
            print(f"\n❌ 原图片不存在: {original_path}")
            fail_list.append(filename)
            continue

        # 从文件名提取角色名（去扩展名）
        char_name = os.path.splitext(filename)[0]

        # 查找提示词
        prompt_text = prompt_map.get(char_name, '')
        if not prompt_text:
            # 尝试模糊匹配
            for pn, pt in prompt_map.items():
                if char_name in pn or pn in char_name:
                    prompt_text = pt
                    char_name = pn
                    break
        if not prompt_text:
            print(f"\n❌ 未找到 {char_name} 的提示词，跳过")
            fail_list.append(filename)
            continue

        # 执行返工
        success, kp, record = rework_character(
            api_key, char_name, prompt_text, original_path,
            original_kp=item['keypoint_count']
        )
        # 记录日志
        log_data["records"].append(record)
        log_data["total_reworks"] += 1
        if success:
            success_count += 1
            log_data["success_count"] += 1
        else:
            fail_list.append(filename)
            log_data["fail_count"] += 1

    # 6. 保存返工日志
    save_rework_log(log_data)
    print(f"\n📝 返工日志已保存: {REWORK_LOG_FILE}")

    # 7. 汇总
    print(f"\n{'='*60}")
    print(f"📊 返工汇总")
    print(f"{'='*60}")
    print(f"  发现不完整: {len(incomplete)} 张")
    print(f"  返工成功: {success_count} 张")
    print(f"  返工失败: {len(fail_list)} 张")
    if fail_list:
        print(f"  失败列表: {', '.join(fail_list)}")
    print(f"\n  临时文件目录: {TEMP_DIR}（可手动删除）")

    return success_count


if __name__ == '__main__':
    main()

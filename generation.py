"""
generation.py - 资产生成模块
整合：pipe.py（主生成管线）+ run_new_script.py（非交互式入口）
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RunningHub 剧本图片自动化生成流程（交互式命令行版）
使用官方 V2 查询接口 /openapi/v2/query
支持并发提交（最多5个任务同时运行）
支持欧美/亚洲风格选择（节点28控制）
【新增】三层质检体系：IQA画质 + CLIP语义对齐 + 专项硬约束检测
【新增】统计学能力：批次分布统计、分位数分析、合格率计算
"""
import os
import sys
import time
import json
import math
import zipfile

# 强制设置stdout/stderr为UTF-8编码（解决Windows cmd GBK编码问题）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import shutil
import re
import requests

# ==================== 配置常量 ====================
WORKFLOW_PROMPT = "2086440329527521282"
WORKFLOW_CHAR = "2086686775652737025"
WORKFLOW_PROP = "2086624313771388929"
WORKFLOW_SCENE = "2086687375505321986"

NODE_CONFIG = {
    WORKFLOW_CHAR: {"nodeId": "103", "fieldName": "String"},
    WORKFLOW_PROP: {"nodeId": "58", "fieldName": "String"},
    WORKFLOW_SCENE: {"nodeId": "242", "fieldName": "String"},
}

BASE_URL = "https://www.runninghub.cn"
QUERY_URL = f"{BASE_URL}/openapi/v2/query"   # V2 查询接口
TIMEOUT_SECONDS = 7200  # 2小时
POLL_INTERVAL = 30      # 轮询间隔（秒）
MAX_CONCURRENT = 5      # 最大并发任务数

# ==================== 质检配置（可按需修改） ====================
# 人物基准图路径（做人脸一致性校验，多个角色可后续扩展为字典）
CHAR_REF_IMG_PATH = "./ref/主角基准图.png"
# 是否启用自动质检
ENABLE_QUALITY_EVAL = True


# ==================== 辅助函数：V2 查询 ====================
def query_task_v2(api_key, task_id):
    """
    使用 V2 接口查询任务状态和结果。
    返回 (status, file_list)：
      - status: 字符串，如 'RUNNING', 'SUCCESS', 'FAILED'
      - file_list: 当 status 为 SUCCESS 时，返回文件列表（每个元素包含 url, nodeId, fileType 等），否则为 None
    """
    headers = {
        "Host": "www.runninghub.cn",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {"taskId": task_id}
    resp = requests.post(QUERY_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    # ----- 兼容两种响应结构 -----
    if 'code' in raw:
        if raw.get('code') != 0:
            raise Exception(f"查询失败: {raw.get('msg')} (完整响应: {raw})")
        data = raw.get('data')
        if data is None:
            raise Exception("响应中缺少 data 字段")
        if isinstance(data, dict):
            status = data.get('taskStatus') or data.get('status')
            file_list = data.get('data') or data.get('results')
        elif isinstance(data, list):
            status = 'SUCCESS'
            file_list = data
        else:
            raise Exception(f"未知的 data 类型: {type(data)}")
    else:
        status = raw.get('status') or raw.get('taskStatus')
        if not status:
            raise Exception(f"响应中没有 status 字段: {raw}")
        file_list = raw.get('results') or raw.get('data')

    # 标准化 file_list
    if isinstance(file_list, list):
        for item in file_list:
            if 'fileUrl' in item and 'url' not in item:
                item['url'] = item['fileUrl']
        file_list = [item for item in file_list if item.get('url')]
    else:
        if status == 'SUCCESS':
            file_list = []
        else:
            file_list = None

    return status, file_list


# ==================== 交互输入 ====================
def interactive_input():
    print("\n" + "=" * 60)
    print(" RunningHub 剧本图片自动化生成流程（V2 API + 自动质检）")
    print("=" * 60)

    api_key = input("\n请输入 RunningHub API 密钥: ").strip()
    if not api_key:
        print("❌ API密钥不能为空！")
        sys.exit(1)

    # ===== 风格选择 =====
    print("\n请选择风格（输入 0 为欧美，输入 1 为亚洲）：")
    style_input = input("请输入 0 或 1: ").strip()
    if style_input not in ['0', '1']:
        print("❌ 无效选项，请输入 0 或 1")
        sys.exit(1)
    style = int(style_input)

    # ===== 剧本输入 =====
    print("\n请选择剧本输入方式：")
    print("  1. 直接粘贴剧本内容（输入完成后，在单独一行输入 EOF 并回车结束）")
    print("  2. 从文件读取剧本路径")
    choice = input("请输入选项 (1 或 2): ").strip()

    script_content = ""
    if choice == "1":
        print("\n请粘贴剧本内容（输入完成后，在单独一行输入 EOF 并回车结束）：")
        lines = []
        while True:
            line = input()
            if line.strip() == "EOF":
                break
            lines.append(line)
        script_content = "\n".join(lines)
        if not script_content.strip():
            print("❌ 剧本内容为空！")
            sys.exit(1)
    elif choice == "2":
        file_path = input("请输入剧本文本文件路径: ").strip()
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            sys.exit(1)
        with open(file_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        if not script_content.strip():
            print("❌ 剧本内容为空！")
            sys.exit(1)
        print(f"✅ 已从文件读取剧本，共 {len(script_content)} 字符")
    else:
        print("❌ 无效选项，请输入 1 或 2")
        sys.exit(1)

    output_dir = input("\n请输入输出目录（直接回车使用默认: ./RunningHub_Outputs）: ").strip()
    if not output_dir:
        output_dir = "./RunningHub_Outputs"

    return api_key, script_content, output_dir, style


# ==================== 辅助函数：从 results 中提取数据 ====================
def _extract_from_status_data(file_list, api_key, task_id):
    """
    从 V2 返回的文件列表（每个元素有 url, nodeId 等）中下载 JSON 文件并提取 6 个列表。
    参数 file_list 是包含文件信息的列表。
    """
    prefix_map = {
        '人物提示词': 'character_prompts',
        '道具提示词': 'prop_prompts',
        '场景提示词': 'scene_prompts',
        '人物命名集': 'character_names',
        '道具命名集': 'prop_names',
        '场景命名集': 'scene_names'
    }
    result = {v: [] for v in prefix_map.values()}
    temp_files = {}

    if not file_list:
        raise Exception("文件列表为空")

    print(f"📥 从 {len(file_list)} 个结果文件中下载并解析 JSON...")
    for item in file_list:
        url = item.get('url')
        node_id = item.get('nodeId')
        if not url:
            print(f"⚠️ 节点 {node_id} 没有 url，跳过")
            continue

        filename = url.split('/')[-1]
        if not filename.endswith('.json'):
            print(f"⚠️ {filename} 不是 JSON 文件，跳过")
            continue

        matched_field = None
        for prefix, field in prefix_map.items():
            if filename.startswith(prefix):
                matched_field = field
                break

        if not matched_field:
            print(f"⚠️ 无法识别文件 {filename}，跳过")
            continue

        try:
            print(f"  📥 下载: {filename}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.text
            temp_files[matched_field] = content
            print(f"  ✅ 下载成功: {filename}")
        except Exception as e:
            print(f"  ❌ 下载失败 {filename}: {e}")
            continue

    expected = ['character_prompts', 'prop_prompts', 'scene_prompts',
                'character_names', 'prop_names', 'scene_names']
    missing = [f for f in expected if f not in temp_files]
    if missing:
        print(f"⚠️ 以下文件未下载到: {missing}")

    for field, content in temp_files.items():
        if field.endswith('_prompts'):
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if lines and lines[0].startswith('['):
                try:
                    arr = json.loads(content)
                    if isinstance(arr, list):
                        result[field] = arr
                        continue
                except:
                    pass
            result[field] = lines
        else:
            try:
                arr = json.loads(content)
                if isinstance(arr, list):
                    result[field] = arr
                else:
                    lines = [line.strip() for line in content.splitlines() if line.strip()]
                    result[field] = lines
            except:
                lines = [line.strip() for line in content.splitlines() if line.strip()]
                result[field] = lines

    return result


# ==================== 核心函数 ====================
def submit_and_wait_prompt_task(api_key, script_content, style):
    """
    提交提示词工作流，并等待完成。
    style: 0 表示欧美风格，1 表示亚洲风格（对应节点28的value）
    """
    headers = {
        "Host": "www.runninghub.cn",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 节点列表：修改节点27的text（剧本内容），修改节点28的value（风格选择）
    node_list = [
        {"nodeId": "27", "fieldName": "text", "fieldValue": script_content},
        {"nodeId": "28", "fieldName": "value", "fieldValue": style}
    ]

    payload = {
        "apiKey": api_key,
        "workflowId": WORKFLOW_PROMPT,
        "addMetadata": False,
        "nodeInfoList": node_list
    }

    print("📤 提交提示词工作流...")
    resp = requests.post(
        f"{BASE_URL}/task/openapi/create",
        headers=headers,
        data=json.dumps(payload),
        timeout=30
    )
    resp.raise_for_status()
    resp_json = resp.json()
    print("🔍 服务器原始响应:", json.dumps(resp_json, ensure_ascii=False, indent=2))

    data = resp_json.get("data")
    if data is None:
        error_msg = resp_json.get("msg", "未知错误")
        raise Exception(f"API 返回错误: {error_msg} (完整响应: {resp_json})")

    task_id = str(data.get("taskId"))
    if not task_id:
        raise Exception("响应中缺少 taskId")

    print(f"✅ 提示词任务已提交: {task_id}")
    start_time = time.time()

    while True:
        if time.time() - start_time > TIMEOUT_SECONDS:
            raise TimeoutError("提示词任务超时")

        time.sleep(POLL_INTERVAL)
        status, file_list = query_task_v2(api_key, task_id)
        print(f"⏳ 当前状态: {status}")

        if status == "SUCCESS":
            print("✅ 任务完成，正在解析结果...")
            if not file_list:
                raise Exception("任务成功但未返回文件列表")
            extracted = _extract_from_status_data(file_list, api_key, task_id)
            print("✅ 数据解析成功")
            return (
                extracted.get('character_prompts', []),
                extracted.get('prop_prompts', []),
                extracted.get('scene_prompts', []),
                extracted.get('character_names', []),
                extracted.get('prop_names', []),
                extracted.get('scene_names', [])
            )
        elif status == "FAILED":
            raise Exception(f"提示词任务失败，状态: {status}")
        else:
            print(f"⏳ 继续等待... (已等待 {int(time.time() - start_time)} 秒)")


def save_names_json(names, prefix, base_dir):
    if not names:
        print(f"⚠️ {prefix} 列表为空，跳过保存")
        return None
    path = os.path.join(base_dir, f"{prefix}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    print(f"✅ 保存 {prefix}.json，共 {len(names)} 条")
    return path


# ==================== 分批逻辑（优化后） ====================
def prepare_sub_tasks_list(char_prompts, prop_prompts, scene_prompts):
    sub_tasks = []

    # 人物：数量>45时分成两批，第一批45，第二批剩余
    if char_prompts:
        if len(char_prompts) > 45:
            sub_tasks.append({"wf": WORKFLOW_CHAR, "prompts": char_prompts[:45], "cat": "人物", "idx": 1})
            sub_tasks.append({"wf": WORKFLOW_CHAR, "prompts": char_prompts[45:], "cat": "人物", "idx": 2})
            print(f"  人物_1: 45 条提示词")
            print(f"  人物_2: {len(char_prompts)-45} 条提示词")
        else:
            sub_tasks.append({"wf": WORKFLOW_CHAR, "prompts": char_prompts, "cat": "人物", "idx": 1})
            print(f"  人物_1: {len(char_prompts)} 条提示词")

    # 道具：始终一批
    if prop_prompts:
        sub_tasks.append({"wf": WORKFLOW_PROP, "prompts": prop_prompts, "cat": "道具", "idx": 1})
        print(f"  道具_1: {len(prop_prompts)} 条提示词")

    # 场景：数量>15时每15一批，最大4批，最后一批可超过15
    if scene_prompts:
        total = len(scene_prompts)
        if total > 15:
            batches = []
            start = 0
            batch_num = 0
            while start < total and batch_num < 4:
                end = min(start + 15, total)
                batches.append(scene_prompts[start:end])
                start = end
                batch_num += 1
                if batch_num == 4 and start < total:
                    batches[-1].extend(scene_prompts[start:])
                    break
            for idx, batch in enumerate(batches, 1):
                sub_tasks.append({"wf": WORKFLOW_SCENE, "prompts": batch, "cat": "场景", "idx": idx})
                print(f"  场景_{idx}: {len(batch)} 条提示词")
        else:
            sub_tasks.append({"wf": WORKFLOW_SCENE, "prompts": scene_prompts, "cat": "场景", "idx": 1})
            print(f"  场景_1: {len(scene_prompts)} 条提示词")

    # 优先级排序：人物 → 场景 → 道具
    priority_order = {"人物": 0, "场景": 1, "道具": 2}
    sub_tasks.sort(key=lambda x: priority_order.get(x["cat"], 99))

    return sub_tasks


# ==================== 并发提交处理 ====================
def process_sub_tasks(api_key, sub_tasks, base_output_dir):
    headers = {
        "Host": "www.runninghub.cn",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    task_info_list = []
    total = len(sub_tasks)
    print(f"📋 共 {total} 个子任务，将按每批最多 {MAX_CONCURRENT} 个并发提交")

    # 分批提交（每批最多 MAX_CONCURRENT 个）
    for batch_start in range(0, total, MAX_CONCURRENT):
        batch_end = min(batch_start + MAX_CONCURRENT, total)
        batch_subtasks = sub_tasks[batch_start:batch_end]
        batch_num = batch_start // MAX_CONCURRENT + 1
        print(f"\n--- 提交第 {batch_num} 批任务 ({batch_start+1}-{batch_end}) ---")

        # 存储当前批次的任务信息
        batch_task_ids = []

        # 并发提交该批次所有任务
        for t in batch_subtasks:
            cfg = NODE_CONFIG[t["wf"]]
            prompt_text = "\n".join(t["prompts"])
            payload = {
                "apiKey": api_key,
                "workflowId": t["wf"],
                "addMetadata": False,
                "nodeInfoList": [{
                    "nodeId": cfg["nodeId"],
                    "fieldName": cfg["fieldName"],
                    "fieldValue": prompt_text
                }]
            }

            print(f"  提交 {t['cat']}_{t['idx']} ...")
            resp = requests.post(
                f"{BASE_URL}/task/openapi/create",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            resp.raise_for_status()
            resp_json = resp.json()

            if not resp_json.get("data") or not resp_json["data"].get("taskId"):
                raise Exception(f"提交任务失败，响应缺少 taskId: {resp_json}")

            tid = str(resp_json["data"]["taskId"])
            info = {"tid": tid, "cat": t["cat"], "idx": t["idx"]}
            batch_task_ids.append(info)
            print(f"    ✅ {t['cat']}_{t['idx']} 已提交: {tid}")

        # 等待该批次所有任务完成
        print(f"  等待第 {batch_num} 批任务完成...")
        start_time = time.time()
        pending = batch_task_ids.copy()

        while pending:
            time.sleep(POLL_INTERVAL)
            # 检查所有待处理任务
            for info in pending[:]:
                status, _ = query_task_v2(api_key, info['tid'])
                if status == "SUCCESS":
                    print(f"    ✅ {info['cat']}_{info['idx']} 完成")
                    pending.remove(info)
                elif status == "FAILED":
                    raise Exception(f"任务 {info['tid']} 失败")
                else:
                    elapsed = int(time.time() - start_time)
                    print(f"    ⏳ {info['cat']}_{info['idx']} 状态: {status}，已等待 {elapsed} 秒")

            # 超时检查
            if time.time() - start_time > TIMEOUT_SECONDS:
                raise TimeoutError("部分任务超时")

        # 将本批次完成的任务加入总列表
        task_info_list.extend(batch_task_ids)

    # 所有任务已完成，开始下载结果
    print("\n📥 开始下载所有任务结果...")
    results = {
        "人物": {"zips": [], "jsons": []},
        "道具": {"zips": [], "jsons": []},
        "场景": {"zips": [], "jsons": []}
    }

    for info in task_info_list:
        # 查询文件列表
        status, file_list = query_task_v2(api_key, info['tid'])
        if status != "SUCCESS":
            print(f"⚠️ 任务 {info['tid']} 状态不是 SUCCESS，跳过")
            continue
        if not file_list:
            print(f"⚠️ 任务 {info['tid']} 无文件列表，跳过")
            continue

        # 查找 ZIP 文件
        dl_url = None
        for item in file_list:
            out_type = item.get('outputType') or item.get('fileType', '')
            url = item.get('url')
            if out_type == 'zip' or (url and url.endswith('.zip')):
                dl_url = url
                break

        if not dl_url:
            dl_url = file_list[0].get('url')
            if not dl_url:
                print(f"⚠️ 任务 {info['tid']} 无下载链接，跳过")
                continue

        t_dir = os.path.join(base_output_dir, f"temp_{info['cat']}_{info['idx']}")
        os.makedirs(t_dir, exist_ok=True)
        z_path = os.path.join(t_dir, f"{info['cat']}_{info['idx']}.zip")

        with requests.get(dl_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(z_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

        results[info['cat']]["zips"].append(z_path)
        print(f"  ✅ 下载 {info['cat']}_{info['idx']}.zip")

        with zipfile.ZipFile(z_path, 'r') as zf:
            for fn in zf.namelist():
                if fn.endswith('.json'):
                    j_path = os.path.join(t_dir, f"{info['cat']}_{info['idx']}.json")
                    with open(j_path, 'wb') as f:
                        f.write(zf.read(fn))
                    results[info['cat']]["jsons"].append(j_path)
                    print(f"  ✅ 提取 {info['cat']}_{info['idx']}.json")
                    break

    return results


# ==================== 合并与重命名（保持不变） ====================
def merge_category_zips(results, final_output_dir):
    final_zips = {}

    for cat in ["人物", "道具", "场景"]:
        cat_data = results[cat]
        if not cat_data["zips"]:
            print(f"⚠️ {cat} 无ZIP文件，跳过合并")
            continue

        cat_dir = os.path.join(final_output_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        m_zip = os.path.join(cat_dir, f"{cat}.zip")
        merged_temp = os.path.join(cat_dir, "merged_temp")
        os.makedirs(merged_temp, exist_ok=True)

        global_idx = 0
        total_images = 0

        for batch_idx, zp in enumerate(cat_data["zips"], 1):
            temp_single = os.path.join(cat_dir, f"temp_single_{batch_idx}")
            os.makedirs(temp_single, exist_ok=True)

            with zipfile.ZipFile(zp, 'r') as zf:
                zf.extractall(temp_single)

            images = [f for f in os.listdir(temp_single)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            if not images:
                print(f"  ⚠️ {zp} 中无图片，跳过")
                shutil.rmtree(temp_single)
                continue

            images.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else -1)

            for img in images:
                ext = os.path.splitext(img)[1]
                new_name = f"{global_idx:05d}{ext}"
                shutil.move(
                    os.path.join(temp_single, img),
                    os.path.join(merged_temp, new_name)
                )
                global_idx += 1
                total_images += 1

            shutil.rmtree(temp_single)
            print(f"  ✅ 批次 {batch_idx} 处理完成，添加 {len(images)} 张图片")

        with zipfile.ZipFile(m_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(merged_temp):
                for f in files:
                    fp = os.path.join(root, f)
                    zf.write(fp, os.path.relpath(fp, merged_temp))

        shutil.rmtree(merged_temp)
        final_zips[cat] = m_zip
        print(f"✅ {cat} 合并完成: {m_zip}，共 {total_images} 张图片")

    return final_zips


# ==================== 清理旧文件（彻底） ====================
def clean_old_files(base_dir):
    categories = ['人物', '场景', '道具']
    print("🧹 开始清理旧文件...")

    # 1. 删除重命名后的图片文件夹
    for cat in categories:
        folder = os.path.join(base_dir, f'重命名后的图片_{cat}')
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"  ✅ 删除文件夹: {folder}")

    # 2. 删除所有 temp_* 临时目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('temp_'):
            shutil.rmtree(item_path)
            print(f"  ✅ 删除临时目录: {item_path}")

    # 3. 删除整个“最终结果”目录
    final_result_dir = os.path.join(base_dir, "最终结果")
    if os.path.exists(final_result_dir):
        shutil.rmtree(final_result_dir)
        print(f"  ✅ 删除最终结果目录: {final_result_dir}")

    # 4. 删除根目录下的 6 个 JSON 数据文件
    json_files = [
        "人物提示词_.json", "人物命名集_.json",
        "道具提示词_.json", "道具命名集_.json",
        "场景提示词_.json", "场景命名集_.json"
    ]
    for fname in json_files:
        fpath = os.path.join(base_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"  ✅ 删除 JSON: {fname}")

    # 5. 删除评估报告目录（旧的）
    for cat in categories:
        eval_folder = os.path.join(base_dir, f'评估报告_{cat}')
        if os.path.exists(eval_folder):
            shutil.rmtree(eval_folder)
            print(f"  ✅ 删除旧评估报告: {eval_folder}")

    # 6. 删除残留的旧版本 ZIP/JSON（以防遗漏）
    deleted_count = 0
    for f in os.listdir(base_dir):
        f_path = os.path.join(base_dir, f)
        if os.path.isfile(f_path):
            for cat in categories:
                if f.startswith(f'{cat}图像集_') and f.endswith('.zip'):
                    os.remove(f_path)
                    print(f"  ✅ 删除 ZIP: {f}")
                    deleted_count += 1
                    break
                if f.startswith(f'{cat}命名集_') and f.endswith('.json'):
                    os.remove(f_path)
                    print(f"  ✅ 删除 JSON: {f}")
                    deleted_count += 1
                    break

    print(f"🧹 清理完成，共删除 {deleted_count} 个额外旧文件")


def rename_images_in_zip(zip_path, json_path, output_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError("JSON 内容不是列表")

    print(f"📄 读取到 {len(names)} 个名称")

    temp_dir = os.path.join(os.path.dirname(zip_path), 'temp_extract')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)
    print(f"📦 解压 ZIP 到 {temp_dir}")

    images = [f for f in os.listdir(temp_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not images:
        raise FileNotFoundError("ZIP 中无图片")

    images.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else -1)
    print(f"🖼️ 找到 {len(images)} 张图片")

    count = min(len(images), len(names))
    if len(images) != len(names):
        print(f"⚠️ 图片数({len(images)})与名称数({len(names)})不一致，取前 {count} 张")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        old_name = images[i]
        new_name = names[i]
        # 清洗非法字符（将 / 替换为 -）
        new_name = new_name.replace('/', '-')
        ext = os.path.splitext(old_name)[1]
        new_full = f"{new_name}{ext}"
        shutil.move(os.path.join(temp_dir, old_name), os.path.join(output_dir, new_full))
        print(f"  ✅ {old_name} → {new_full}")

    shutil.rmtree(temp_dir)
    print(f"🎉 重命名完成，输出: {output_dir}")


# ==================== 新增：自动质量质检入口 ====================
def run_auto_quality_eval(
    person_prompts,
    prop_prompts,
    scene_prompts,
    base_output_dir
):
    """
    自动执行三类资产的三层质检 + 批次统计
    异常捕获：质检失败不中断主流程，仅打印警告
    """
    print("\n" + "=" * 60)
    print(">>> 第8步：自动执行三层质量质检")
    print("=" * 60)

    try:
        from core_eval import batch_evaluate
    except Exception as e:
        print(f"⚠️  质检模块加载失败，跳过质检: {e}")
        print("💡 请确保 quality_eval.py 与本脚本在同一目录下")
        return None

    eval_results = {}

    # ========== 1. 人物资产评估 ==========
    char_img_dir = os.path.join(base_output_dir, '重命名后的图片_人物')
    char_eval_dir = os.path.join(base_output_dir, '评估报告_人物')

    if os.path.exists(char_img_dir) and person_prompts and len(os.listdir(char_img_dir)) > 0:
        try:
            print("\n👤 正在评估人物资产...")
            ref_img = CHAR_REF_IMG_PATH if os.path.exists(CHAR_REF_IMG_PATH) else None
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
            # 打印核心指标摘要
            if "CLIP匹配度" in stats_char:
                s = stats_char["CLIP匹配度"]
                print(f"     · CLIP匹配度 均值: {s['mean']}，合格率: {s['pass_rate']}%")
            if "人脸相似度" in stats_char:
                s = stats_char["人脸相似度"]
                print(f"     · 人脸相似度 均值: {s['mean']}，P10: {s['p10']}，合格率: {s['pass_rate']}%")
            if "人体关键点" in stats_char:
                s = stats_char["人体关键点"]
                print(f"     · 人体完整性 合格率: {s['pass_rate']}%")

        except Exception as e:
            print(f"  ❌ 人物质检执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n👤 人物资产为空，跳过评估")

    # ========== 2. 道具资产评估 ==========
    prop_img_dir = os.path.join(base_output_dir, '重命名后的图片_道具')
    prop_eval_dir = os.path.join(base_output_dir, '评估报告_道具')

    if os.path.exists(prop_img_dir) and prop_prompts and len(os.listdir(prop_img_dir)) > 0:
        try:
            print("\n📦 正在评估道具资产...")
            # 如需校验指定道具文字，可在此构造 expected_text_map = {文件名: 预期文字}
            expected_text_map = {}

            df_prop, stats_prop, group_prop = batch_evaluate(
                prompt_list=prop_prompts,
                image_dir=prop_img_dir,
                asset_type="prop",
                expected_text_map=expected_text_map if expected_text_map else None,
                output_report=True,
                output_dir=prop_eval_dir
            )

            eval_results["prop"] = {"detail": df_prop, "stats": stats_prop, "group": group_prop}

            print(f"  ✅ 道具评估完成，报告已保存至: {prop_eval_dir}")
            if "CLIP匹配度" in stats_prop:
                s = stats_prop["CLIP匹配度"]
                print(f"     · CLIP匹配度 均值: {s['mean']}，合格率: {s['pass_rate']}%")

        except Exception as e:
            print(f"  ❌ 道具质检执行失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n📦 道具资产为空，跳过评估")

    # ========== 3. 场景资产评估 ==========
    scene_img_dir = os.path.join(base_output_dir, '重命名后的图片_场景')
    scene_eval_dir = os.path.join(base_output_dir, '评估报告_场景')

    if os.path.exists(scene_img_dir) and scene_prompts and len(os.listdir(scene_img_dir)) > 0:
        try:
            print("\n🏞️  正在评估场景资产...")

            df_scene, stats_scene, group_scene = batch_evaluate(
                prompt_list=scene_prompts,
                image_dir=scene_img_dir,
                asset_type="scene",
                output_report=True,
                output_dir=scene_eval_dir
            )

            eval_results["scene"] = {"detail": df_scene, "stats": stats_scene, "group": group_scene}

            print(f"  ✅ 场景评估完成，报告已保存至: {scene_eval_dir}")
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
        print("\n🏞️  场景资产为空，跳过评估")

    print("\n✅ 质量质检流程执行完毕")
    return eval_results


# ==================== 主程序 ====================
def main():
    api_key, script_content, base_output_dir, style = interactive_input()
    final_result_dir = os.path.join(base_output_dir, "最终结果")

    try:
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(final_result_dir, exist_ok=True)
        print(f"✅ 初始化目录完成: {base_output_dir}")

        clean_old_files(base_output_dir)

        print("\n" + "=" * 60)
        print(">>> 第2步：提交提示词工作流（传入剧本）")
        print("=" * 60)

        person_prompts, prop_prompts, scene_prompts, person_names, prop_names, scene_names = submit_and_wait_prompt_task(
            api_key=api_key,
            script_content=script_content,
            style=style
        )

        print(f"📊 人物提示词: {len(person_prompts)} 条，道具: {len(prop_prompts)} 条，场景: {len(scene_prompts)} 条")
        print(f"📊 人物名称: {len(person_names)} 个，道具名称: {len(prop_names)} 个，场景名称: {len(scene_names)} 个")

        print("\n" + "=" * 60)
        print(">>> 第3步：保存全部 6 个 JSON 文件")
        print("=" * 60)

        person_json = save_names_json(person_names, "人物命名集_", base_output_dir)
        prop_json = save_names_json(prop_names, "道具命名集_", base_output_dir)
        scene_json = save_names_json(scene_names, "场景命名集_", base_output_dir)

        save_names_json(person_prompts, "人物提示词_", base_output_dir)
        save_names_json(prop_prompts, "道具提示词_", base_output_dir)
        save_names_json(scene_prompts, "场景提示词_", base_output_dir)

        print("✅ 全部 6 个 JSON 文件已保存")

        print("\n" + "=" * 60)
        print(">>> 第4步：准备子任务列表")
        print("=" * 60)

        sub_tasks = prepare_sub_tasks_list(person_prompts, prop_prompts, scene_prompts)
        print(f"📋 共生成 {len(sub_tasks)} 个子任务")

        print("\n" + "=" * 60)
        print(">>> 第5步：执行子任务（并发提交，最多5个同时运行）")
        print("=" * 60)

        results = process_sub_tasks(api_key, sub_tasks, base_output_dir)

        print("\n" + "=" * 60)
        print(">>> 第6步：合并同类 ZIP（保证顺序）")
        print("=" * 60)

        merged_zips = merge_category_zips(results, final_result_dir)

        print("\n" + "=" * 60)
        print(">>> 第7步：重命名图片")
        print("=" * 60)

        person_zip = merged_zips.get('人物', '')
        prop_zip = merged_zips.get('道具', '')
        scene_zip = merged_zips.get('场景', '')

        if person_zip and person_json:
            rename_images_in_zip(person_zip, person_json, os.path.join(base_output_dir, '重命名后的图片_人物'))
        if prop_zip and prop_json:
            rename_images_in_zip(prop_zip, prop_json, os.path.join(base_output_dir, '重命名后的图片_道具'))
        if scene_zip and scene_json:
            rename_images_in_zip(scene_zip, scene_json, os.path.join(base_output_dir, '重命名后的图片_场景'))

        # ========== 新增：自动质检 ==========
        if ENABLE_QUALITY_EVAL:
            run_auto_quality_eval(
                person_prompts=person_prompts,
                prop_prompts=prop_prompts,
                scene_prompts=scene_prompts,
                base_output_dir=base_output_dir
            )

        print("\n" + "=" * 60)
        print("🎉 全部流程执行完毕！")
        print(f"📁 最终生成路径: {final_result_dir}")
        print(f"📊 质检报告路径: {base_output_dir} 下的「评估报告_人物/道具/场景」")
        print(f"👤 人物ZIP: {person_zip}")
        print(f"📦 道具ZIP: {prop_zip}")
        print(f"🏞️ 场景ZIP: {scene_zip}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 流程失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==================== run_new_script.py 非交互式入口 ====================
# -*- coding: utf-8 -*-
"""
非交互式剧本资产生成入口
用于测试新剧本（如"无法言说的秘密"）的资产生成
用法: python run_new_script.py <剧本文件路径> <输出目录名>
"""
import os
import sys
import json

# 强制设置stdout/stderr为UTF-8编码（解决Windows cmd GBK编码问题）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# pipe.py的核心函数已在同一文件中定义，无需额外import


def load_api_key():
    """从本地文件读取API密钥"""
    key_file = os.path.join(BASE_DIR, "API密钥.txt")
    with open(key_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    import re
    match = re.search(r'([a-f0-9]{32})', content, re.IGNORECASE)
    if match:
        return match.group(1)
    if '：' in content:
        return content.split('：')[-1].strip()
    return content


def run_pipeline(api_key, script_content, output_dir, style=1):
    """
    非交互式运行完整生成管线
    style: 1=亚洲风格, 0=欧美风格
    """
    final_result_dir = os.path.join(output_dir, "最终结果")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_result_dir, exist_ok=True)
    print(f"✅ 初始化目录完成: {output_dir}")

    clean_old_files(output_dir)

    # 第2步：提交提示词工作流
    print("\n" + "=" * 60)
    print(">>> 第2步：提交提示词工作流（传入剧本）")
    print("=" * 60)
    person_prompts, prop_prompts, scene_prompts, person_names, prop_names, scene_names = \
        submit_and_wait_prompt_task(api_key=api_key, script_content=script_content, style=style)

    print(f"📊 人物提示词: {len(person_prompts)} 条，道具: {len(prop_prompts)} 条，场景: {len(scene_prompts)} 条")

    # 第3步：保存JSON
    print("\n" + "=" * 60)
    print(">>> 第3步：保存全部 6 个 JSON 文件")
    print("=" * 60)
    person_json = save_names_json(person_names, "人物命名集_", output_dir)
    prop_json = save_names_json(prop_names, "道具命名集_", output_dir)
    scene_json = save_names_json(scene_names, "场景命名集_", output_dir)
    save_names_json(person_prompts, "人物提示词_", output_dir)
    save_names_json(prop_prompts, "道具提示词_", output_dir)
    save_names_json(scene_prompts, "场景提示词_", output_dir)
    print("✅ 全部 6 个 JSON 文件已保存")

    # 第4步：准备子任务
    print("\n" + "=" * 60)
    print(">>> 第4步：准备子任务列表")
    print("=" * 60)
    sub_tasks = prepare_sub_tasks_list(person_prompts, prop_prompts, scene_prompts)
    print(f"📋 共生成 {len(sub_tasks)} 个子任务")

    # 第5步：执行子任务
    print("\n" + "=" * 60)
    print(">>> 第5步：执行子任务（并发提交，最多5个同时运行）")
    print("=" * 60)
    results = process_sub_tasks(api_key, sub_tasks, output_dir)

    # 第6步：合并ZIP
    print("\n" + "=" * 60)
    print(">>> 第6步：合并同类 ZIP（保证顺序）")
    print("=" * 60)
    merged_zips = merge_category_zips(results, final_result_dir)

    # 第7步：重命名图片
    print("\n" + "=" * 60)
    print(">>> 第7步：重命名图片")
    print("=" * 60)
    person_zip = merged_zips.get('人物', '')
    prop_zip = merged_zips.get('道具', '')
    scene_zip = merged_zips.get('场景', '')

    if person_zip and person_json:
        rename_images_in_zip(person_zip, person_json, os.path.join(output_dir, '重命名后的图片_人物'))
    if prop_zip and prop_json:
        rename_images_in_zip(prop_zip, prop_json, os.path.join(output_dir, '重命名后的图片_道具'))
    if scene_zip and scene_json:
        rename_images_in_zip(scene_zip, scene_json, os.path.join(output_dir, '重命名后的图片_场景'))

    print("\n" + "=" * 60)
    print("🎉 资产生成完毕！")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)

    return {
        "person_prompts": person_prompts,
        "prop_prompts": prop_prompts,
        "scene_prompts": scene_prompts,
        "output_dir": output_dir
    }


# run_new_script.py 的非交互式main逻辑已整合到下方的 if __name__ == "__main__" 块中

if __name__ == "__main__":
    # 支持两种模式：
    # 1. 有命令行参数 → 非交互式模式: python generation.py <剧本> <输出目录> <风格>
    # 2. 无命令行参数 → 交互式模式（原pipe.py的interactive_input）
    if len(sys.argv) >= 2:
        # 非交互式模式
        script_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else r"E:\comfyui\RunningHub_Outputs"
        style = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

        print(f"剧本文件: {script_file}")
        print(f"输出目录: {output_dir}")
        print(f"风格: {'亚洲' if style==1 else '欧美'}")

        # 读取剧本
        with open(script_file, 'r', encoding='utf-8') as f:
            script_content = f.read()
        print(f"剧本字符数: {len(script_content)}")

        # 读取API密钥
        api_key = load_api_key()
        print(f"API密钥: {api_key[:8]}...{api_key[-4:]}")

        # 运行管线
        run_pipeline(api_key, script_content, output_dir, style)
    else:
        # 交互式模式
        main()


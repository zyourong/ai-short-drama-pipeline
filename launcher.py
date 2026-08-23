"""
launcher.py - 流程编排模块
整合：launcher.py（全流程启动器）+ auto_run.py（旧自动化入口）
"""
# -*- coding: utf-8 -*-
"""
短剧资产全流程启动器
流程：剧本输入 → pipe资产生成 → 质检 → 自动返工 → 重新质检 → 生成报告 → 假设检验对比
用法：
  python launcher.py                           # 使用默认配置
  python launcher.py <剧本文件路径>             # 指定剧本文件
  python launcher.py <剧本文件路径> <输出目录>   # 指定剧本和输出目录
"""
import os
import sys
import time
import json
import subprocess

# 强制设置stdout/stderr为UTF-8编码（解决Windows cmd GBK编码问题）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==================== 配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GPU torch路径（必须在import pandas/torch之前添加）
PYLIBS = os.path.join(BASE_DIR, "pylibs")
if os.path.exists(PYLIBS):
    sys.path.insert(0, PYLIBS)

import pandas as pd

# 默认剧本文件（可通过命令行参数覆盖）
DEFAULT_SCRIPT = os.path.join(BASE_DIR, "剧本", "无法言说的秘密_完整版.txt")
# 默认输出目录
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "RunningHub_Outputs_无法言说的秘密")
# 基准值目录（用于假设检验对比）
BASELINE_DIR = os.path.join(BASE_DIR, "RunningHub_Outputs")
# 人体完整性阈值
INTEGRITY_THRESHOLD = 17

def get_env(output_dir):
    """获取子进程环境变量"""
    env = os.environ.copy()
    env["EVAL_OUTPUT_DIR"] = output_dir
    env["PYTHONIOENCODING"] = "utf-8"  # 强制子进程使用UTF-8编码
    env["PYTHONUNBUFFERED"] = "1"      # 禁用输出缓冲，实时打印
    if os.path.exists(PYLIBS):
        env["PYTHONPATH"] = PYLIBS
    return env


def run_live(cmd_list, description, env=None, cwd=None):
    """实时输出子进程进度（通过临时文件重定向），返回是否成功"""
    import tempfile
    print(f"\n{'='*70}")
    print(f" {description}")
    print(f"{'='*70}")
    script_path = cmd_list[1] if len(cmd_list) > 1 else cmd_list[0]
    print(f"[启动] {os.path.basename(script_path)}")
    start_time = time.time()

    # 用临时文件重定向输出，避免Windows下PIPE导致的问题
    tmp_out = tempfile.mktemp(suffix='.log', prefix='launcher_')
    f_out = open(tmp_out, 'w', encoding='utf-8', buffering=1)
    try:
        proc = subprocess.Popen(
            cmd_list,
            cwd=cwd or BASE_DIR,
            env=env or os.environ,
            stdin=subprocess.DEVNULL,
            stdout=f_out,
            stderr=subprocess.STDOUT
        )
        print(f"[子进程PID] {proc.pid}", flush=True)

        # 实时读取临时文件
        line_count = 0
        output_lines = []
        last_pos = 0
        while proc.poll() is None:
            time.sleep(2)
            try:
                f_out.flush()
                with open(tmp_out, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()
                for line in new_lines:
                    line = line.rstrip()
                    if line:
                        print(f"  {line}", flush=True)
                        line_count += 1
                        output_lines.append(line)
            except Exception:
                pass

        # 读取剩余输出
        try:
            f_out.flush()
            with open(tmp_out, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(last_pos)
                new_lines = f.readlines()
            for line in new_lines:
                line = line.rstrip()
                if line:
                    print(f"  {line}", flush=True)
                    line_count += 1
                    output_lines.append(line)
        except Exception:
            pass

        proc.wait()
        elapsed = time.time() - start_time
        print(f"\n[完成] 返回码: {proc.returncode} | 耗时: {elapsed:.1f}s ({elapsed/60:.1f}min) | 输出行数: {line_count}", flush=True)
    finally:
        f_out.close()
        # 清理临时文件
        try:
            os.remove(tmp_out)
        except Exception:
            pass

    return proc.returncode == 0


def run_step(script_name, description, env=None, cwd=None):
    """运行一个步骤（实时输出）"""
    script_path = os.path.join(BASE_DIR, script_name)
    return run_live([sys.executable, script_path], description, env=env, cwd=cwd)


def run_pipe(script_file, output_dir, style=1):
    """运行pipe资产生成（实时输出）"""
    desc = f"步骤 1/6：运行pipe资产生成\n 剧本: {script_file}\n 输出: {output_dir}\n 风格: {'亚洲' if style==1 else '欧美'}"
    script_path = os.path.join(BASE_DIR, "generation.py")
    env = get_env(output_dir)
    return run_live([sys.executable, script_path, script_file, output_dir, str(style)], desc, env=env)


def check_incomplete(output_dir):
    """检查人体不完整的图片"""
    eval_csv = os.path.join(output_dir, "评估报告_人物", "eval_detail.csv")
    if not os.path.exists(eval_csv):
        return []
    df = pd.read_csv(eval_csv)
    incomplete = df[
        (df['integrity_pass'] == False) |
        (df['keypoint_count'] < INTEGRITY_THRESHOLD)
    ]
    result = []
    for _, row in incomplete.iterrows():
        result.append({
            'filename': row['filename'],
            'keypoint_count': int(row['keypoint_count']) if pd.notna(row['keypoint_count']) else 0
        })
    return result


def run_hypothesis_test(output_dir):
    """运行假设检验对比并生成报告"""
    print(f"\n{'='*70}")
    print(" 步骤 6/6：假设检验对比 + 生成假设检验报告")
    print(f"{'='*70}")

    sys.path.insert(0, BASE_DIR)
    from analysis import compare_with_baseline, print_comparison_report, save_comparison_result
    from reporting import generate_html_report

    result = compare_with_baseline(output_dir)
    print_comparison_report(result)

    # 保存JSON
    json_path = os.path.join(output_dir, "假设检验对比结果.json")
    save_comparison_result(result, json_path)

    # 生成HTML
    html = generate_html_report(result, output_dir)
    html_path = os.path.join(output_dir, "假设检验对比报告.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ 假设检验报告已保存: {html_path}")
    print(f"文件大小: {os.path.getsize(html_path)/1024:.1f} KB")

    return result


def interactive_config():
    """交互式配置：让用户选择剧本文件和输出目录"""
    print("\n" + "=" * 70)
    print(" 交互式配置")
    print("=" * 70)

    # 列出剧本目录下的文件
    script_dir = os.path.join(BASE_DIR, "剧本")
    if os.path.exists(script_dir):
        txt_files = [f for f in os.listdir(script_dir) if f.endswith('.txt')]
        if txt_files:
            print("\n可用剧本文件:")
            for i, f in enumerate(txt_files, 1):
                fpath = os.path.join(script_dir, f)
                size = os.path.getsize(fpath)
                print(f"  {i}. {f} ({size/1024:.1f} KB)")
            choice = input(f"\n请选择剧本 (1-{len(txt_files)})，或直接输入文件路径: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(txt_files):
                script_file = os.path.join(script_dir, txt_files[int(choice)-1])
            else:
                script_file = choice
        else:
            script_file = input("\n请输入剧本文件路径: ").strip()
    else:
        script_file = input("\n请输入剧本文件路径: ").strip()

    if not os.path.exists(script_file):
        print(f"❌ 文件不存在: {script_file}")
        sys.exit(1)

    # 输出目录
    default_output = os.path.join(BASE_DIR, "RunningHub_Outputs_" + os.path.splitext(os.path.basename(script_file))[0])
    output_dir = input(f"\n请输入输出目录 (回车使用默认: {default_output}): ").strip()
    if not output_dir:
        output_dir = default_output

    # 风格选择
    style_input = input("\n请选择风格 (0=欧美, 1=亚洲，回车默认1): ").strip()
    style = int(style_input) if style_input in ['0', '1'] else 1

    print(f"\n✅ 配置完成:")
    print(f"  剧本: {script_file}")
    print(f"  输出: {output_dir}")
    print(f"  风格: {'亚洲' if style==1 else '欧美'}")

    return script_file, output_dir, style


def main():
    start_time = time.time()

    # 解析命令行参数
    if len(sys.argv) >= 2:
        # 命令行参数模式
        script_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
        style = 1
    else:
        # 交互模式
        script_file, output_dir, style = interactive_config()

    print("=" * 70)
    print(" 短剧资产全流程启动器")
    print(" 剧本 → 生成 → 质检 → 返工 → 报告 → 假设检验")
    print("=" * 70)
    print(f" 剧本文件: {script_file}")
    print(f" 输出目录: {output_dir}")
    print(f" 基准目录: {BASELINE_DIR}")
    print("=" * 70)

    # 检查剧本文件
    if not os.path.exists(script_file):
        print(f"❌ 剧本文件不存在: {script_file}")
        return

    env = get_env(output_dir)

    # 步骤1：运行pipe资产生成
    pipe_ok = run_pipe(script_file, output_dir, style)
    if not pipe_ok:
        print("\n❌ 资产生成失败，终止流程")
        return

    # 步骤2：运行质检
    eval_ok = run_step("core_eval.py", "步骤 2/6：运行全量质检", env=env)
    if not eval_ok:
        print("\n❌ 初始质检失败，终止流程")
        return

    # 步骤3：检测人体不完整
    print(f"\n{'='*70}")
    print(" 步骤 3/6：检测人体完整性")
    print(f"{'='*70}")
    incomplete = check_incomplete(output_dir)
    if not incomplete:
        print("✅ 所有人像人体完整性均合格，无需返工")
    else:
        print(f"⚠️  发现 {len(incomplete)} 张人体不完整的图片:")
        for item in incomplete:
            print(f"  - {item['filename']} (关键点: {item['keypoint_count']}/{INTEGRITY_THRESHOLD})")

        # 步骤4：自动返工
        rework_ok = run_step("rework.py", "步骤 4/6：自动返工（人体不完整）", env=env)

        if rework_ok:
            # 返工成功后重新质检
            run_step("core_eval.py", "步骤 4.5/6：返工成功，重新运行全量质检", env=env)
            remaining = check_incomplete(output_dir)
            if remaining:
                print(f"\n⚠️  重新质检后仍有 {len(remaining)} 张不完整")
            else:
                print("\n✅ 重新质检后所有人像人体完整性均合格！")
        else:
            print("\n⚠️  返工未成功，使用原质检结果")

    # 步骤5：生成人物/道具/场景/综合报告
    run_step("reporting.py", "步骤 5/6：生成人物/道具/场景/综合HTML报告", env=env)

    # 步骤6：假设检验对比 + 报告
    hypothesis_result = run_hypothesis_test(output_dir)

    # 汇总
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(" 全流程完成")
    print(f"{'='*70}")
    print(f" 总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f" 输出目录: {output_dir}")
    print(f" 假设检验结论: {hypothesis_result['conclusion']}")
    print(f" 指标合格率: {hypothesis_result['pass_rate']*100:.1f}%")

    # 列出生成的报告
    print(f"\n📋 生成的报告文件:")
    reports = [
        ("评估报告_人物", "人物资产质量报告.html"),
        ("评估报告_道具", "道具资产质量报告.html"),
        ("评估报告_场景", "场景资产质量报告.html"),
        ("", "短剧资产质量评估综合报告.html"),
        ("", "假设检验对比报告.html"),
    ]
    for subdir, filename in reports:
        path = os.path.join(output_dir, subdir, filename) if subdir else os.path.join(output_dir, filename)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✅ {filename} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  {filename} (未生成)")

    print("\n🎉 全部完成！")


if __name__ == '__main__':
    main()

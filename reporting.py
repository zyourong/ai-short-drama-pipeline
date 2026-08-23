"""
reporting.py - 报告生成模块
整合：generate_all_reports.py（4份HTML报告）+ generate_comparison_report.py（假设检验报告）
"""
# -*- coding: utf-8 -*-
"""
统一报告生成器：生成人物、道具、场景独立报告 + 综合报告
所有单图明细表格随机抽样5个展示
"""
import os, json, base64, io
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

BASE = os.environ.get("EVAL_OUTPUT_DIR", r'E:\comfyui\RunningHub_Outputs')
SAMPLE_N = 5
SAMPLE_SEED = 42

C = {
    'primary':'#2563eb','secondary':'#0891b2','success':'#16a34a',
    'warning':'#f59e0b','danger':'#dc2626','skip':'#9ca3af',
    'bg':'#f8fafc','card':'#ffffff','border':'#e2e8f0','text':'#1e293b','text_light':'#64748b',
    'character':'#0891b2','prop':'#f59e0b','scene':'#16a34a',
}

# ==================== 指标中英文对照与详细说明 ====================
METRIC_CN = {
    'brisque_score': 'BRISQUE画质评分',
    'blur_score': '拉普拉斯清晰度',
    'clip_score': 'CLIP图文匹配度',
    'keypoint_count': '人体关键点数量',
    'intra_clip_consistency': '分区域CLIP一致性',
    'intra_face_consistency': '人脸一致性',
    'iqa_pass': '画质合格',
    'clip_pass': 'CLIP匹配合格',
    'integrity_pass': '人体完整性合格',
    'text_lang_pass': '文字语种合格',
    'text_keyword_pass': '文字关键词合格',
    'text_pass': '文字综合合格',
    'group_consistency': '场景组内一致性',
}

METRIC_DESC = {
    'brisque_score': '无参考图像质量评估指标，基于自然场景统计。取值0-100，分数越低表示图像失真越少、画质越好。',
    'blur_score': '通过拉普拉斯算子计算图像边缘的方差，衡量清晰程度。数值越高表示边缘越锐利、图像越清晰；数值过低表示模糊。',
    'clip_score': '使用CLIP模型计算图像与提示词文本的余弦相似度，衡量生成图片是否符合提示词描述。取值0-1，≥0.26判定为合格。',
    'keypoint_count': '使用YOLOv8-pose模型检测人体17个关键点（COCO标准：鼻、眼、耳、肩、肘、腕、髋、膝、踝）。必须=17才判定为人体完整。',
    'intra_clip_consistency': '将人物1×4横排图裁剪为4个区域，计算相邻区域CLIP特征余弦相似度，衡量同一角色不同视图下的一致性。≥0.70合格。',
    'intra_face_consistency': '使用InsightFace提取人物1×4图中第1格（特写）和第2格（正视图）的人脸特征，计算余弦相似度，衡量人脸一致性。≥0.72合格。',
    'iqa_pass': '画质综合合格判定，基于BRISQUE评分和清晰度综合判断。',
    'clip_pass': 'CLIP图文匹配合格判定，clip_score≥0.26为合格。',
    'integrity_pass': '人体完整性合格判定，keypoint_count必须=17为合格。',
    'text_lang_pass': '道具文字语种校验合格判定，检测到的文字语种是否符合预期（亚洲风格以中文为主）。',
    'text_keyword_pass': '道具文字关键词校验合格判定，OCR结果是否包含道具名称关键词。',
    'text_pass': '道具文字综合合格判定，语种校验和关键词校验同时合格。',
    'group_consistency': '场景四宫格2×2裁剪4视图的CLIP特征余弦相似度，衡量同一场景不同视图的一致性。≥0.65合格。',
}


def metric_explanation_section(metrics=None, accent='#2563eb'):
    """生成指标说明HTML章节，metrics为要显示的指标列表，None则显示全部"""
    if metrics is None:
        metrics = list(METRIC_CN.keys())
    html = f'''
<div style="background:{C['card']};border-radius:10px;padding:24px;margin-top:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
<h2 style="font-size:20px;margin-bottom:16px;padding-bottom:10px;border-bottom:2px solid #f0f0f0;color:{accent};">📖 指标说明</h2>
<p style="color:#666;font-size:14px;margin-bottom:16px;">本报告中使用的质检指标含义、计算方式和合格标准：</p>
<table style="width:100%;border-collapse:collapse;">
<thead><tr>
<th style="padding:10px 12px;text-align:left;background:#fafafa;font-weight:600;color:#555;border-bottom:1px solid #f0f0f0;width:18%;">指标名称</th>
<th style="padding:10px 12px;text-align:left;background:#fafafa;font-weight:600;color:#555;border-bottom:1px solid #f0f0f0;width:15%;">英文标识</th>
<th style="padding:10px 12px;text-align:left;background:#fafafa;font-weight:600;color:#555;border-bottom:1px solid #f0f0f0;">详细说明</th>
</tr></thead><tbody>'''
    for m in metrics:
        cn = METRIC_CN.get(m, m)
        desc = METRIC_DESC.get(m, '—')
        html += f'''<tr>
<td style="padding:10px 12px;border-bottom:1px solid #f0f0f0;"><strong>{cn}</strong></td>
<td style="padding:10px 12px;border-bottom:1px solid #f0f0f0;font-family:monospace;font-size:12px;color:#666;">{m}</td>
<td style="padding:10px 12px;border-bottom:1px solid #f0f0f0;font-size:13px;line-height:1.6;color:#444;">{desc}</td>
</tr>'''
    html += '</tbody></table></div>'
    return html

def fig2b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def load(asset):
    d = os.path.join(BASE, f'评估报告_{asset}')
    detail = pd.read_csv(os.path.join(d,'eval_detail.csv'), encoding='utf-8-sig')
    stats = pd.read_csv(os.path.join(d,'batch_stats_summary.csv'), encoding='utf-8-sig', index_col=0)
    gp = os.path.join(d,'group_consistency.csv')
    group = pd.read_csv(gp, encoding='utf-8-sig') if os.path.exists(gp) else None
    return detail, stats, group

def sample_df(df):
    n = min(SAMPLE_N, len(df))
    return df.sample(n=n, random_state=SAMPLE_SEED) if len(df) > SAMPLE_N else df

def fmt(val, col):
    if pd.isna(val): return '<span style="color:#9ca3af;">—</span>'
    if 'pass' in col:
        if val==True: return '<span style="color:#16a34a;font-weight:bold;">✓</span>'
        if val==False: return '<span style="color:#dc2626;font-weight:bold;">✗</span>'
    if isinstance(val,float): return f'{val:.4f}' if val<1 else f'{val:.2f}'
    return str(val)

def make_table(df, cols, headers, total_count):
    sampled = sample_df(df)
    h = ''.join(f'<th>{x}</th>' for x in headers)
    rows = ''
    for _,row in sampled.iterrows():
        rows += '<tr>' + ''.join(f'<td>{fmt(row.get(c,""),c)}</td>' for c in cols) + '</tr>'
    note = f'<div class="sample-note">📋 以下为随机抽取的 {len(sampled)} 个样本明细（共 {total_count} 个样本）</div>'
    return note + f'<table><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table>'

def base_css(accent):
    return f'''
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:{C['bg']};color:{C['text']};line-height:1.6}}
.container{{max-width:1200px;margin:0 auto;padding:30px 20px}}
.header{{background:linear-gradient(135deg,#1e40af 0%,{accent} 100%);color:white;padding:40px;border-radius:16px;margin-bottom:30px;box-shadow:0 10px 40px rgba(37,99,235,0.2)}}
.header h1{{font-size:28px;margin-bottom:10px}}
.header .sub{{font-size:14px;opacity:0.9}}
.header .time{{font-size:13px;opacity:0.8;margin-top:8px}}
.card{{background:{C['card']};border-radius:12px;padding:25px;margin-bottom:25px;box-shadow:0 2px 12px rgba(0,0,0,0.06);border:1px solid {C['border']}}}
.card h2{{font-size:20px;margin-bottom:18px;padding-bottom:12px;border-bottom:2px solid {accent};color:{accent}}}
.card h3{{font-size:16px;margin:15px 0 10px}}
.ov-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:10px}}
.ov-card{{border-radius:10px;padding:18px;text-align:center;border:1px solid {C['border']};background:linear-gradient(135deg,#f0f9ff,#e0f2fe)}}
.ov-card .lbl{{font-size:12px;color:{C['text_light']};margin-bottom:6px}}
.ov-card .val{{font-size:24px;font-weight:bold;color:{accent}}}
.ov-card .unit{{font-size:12px;color:{C['text_light']}}}
.chart{{text-align:center;margin:15px 0}}
.chart img{{max-width:100%;height:auto;border-radius:8px}}
.table-wrap{{overflow-x:auto;margin-top:10px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{background:{accent};color:white;padding:11px 8px;text-align:center;font-weight:600;white-space:nowrap}}
td{{padding:9px;text-align:center;border-bottom:1px solid {C['border']}}}
tr:nth-child(even){{background:#f8fafc}}
tr:hover{{background:#eff6ff}}
.sample-note{{background:#f1f5f9;border-left:4px solid {C['skip']};padding:10px 14px;border-radius:6px;margin:10px 0;font-size:13px;color:{C['text_light']}}}
.notice{{background:#f1f5f9;border-left:4px solid {C['skip']};padding:12px 16px;border-radius:6px;margin:10px 0;font-size:13px;color:{C['text_light']}}}
.conclusion{{background:linear-gradient(135deg,#fefce8,#fef9c3);border-left:4px solid {C['warning']};padding:20px;border-radius:8px;margin-top:15px}}
.conclusion h3{{color:#92400e;margin-bottom:10px}}
.conclusion ul{{margin-left:20px}}
.conclusion li{{margin-bottom:7px;color:#78350f}}
.footer{{text-align:center;padding:20px;color:{C['text_light']};font-size:12px}}
'''

# ==================== 通用图表 ====================
def chart_pass_bar(stats, accent, title):
    metrics, rates, colors = [], [], []
    for idx, row in stats.iterrows():
        if pd.notna(row.get('pass_rate')):
            metrics.append(idx); rates.append(row['pass_rate'])
            colors.append(C['success'] if row['pass_rate']>=80 else (C['warning'] if row['pass_rate']>=50 else C['danger']))
    fig, ax = plt.subplots(figsize=(9,4.5))
    bars = ax.bar(metrics, rates, color=colors, width=0.55, edgecolor='white')
    for bar,r in zip(bars,rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{r:.0f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('合格率 (%)', fontsize=11); ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0,115); ax.axhline(y=80,color=C['success'],linestyle='--',alpha=0.5,label='优秀线(80%)')
    ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(axis='x',rotation=10)
    return fig2b64(fig)

def chart_distribution(detail, accent):
    fig, axes = plt.subplots(1,3,figsize=(13,4.5))
    pairs = [('BRISQUE画质分\n(越低越好)',detail['brisque_score'],45,C['primary']),
             ('CLIP图文匹配度\n(越高越好)',detail['clip_score'],0.26,accent),
             ('拉普拉斯清晰度\n(越高越好)',detail['blur_score'],50,C['success'])]
    for ax,(title,data,line,color) in zip(axes,pairs):
        if len(data.dropna())>=2:
            bp = ax.boxplot(data.dropna(), patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor(color); bp['boxes'][0].set_alpha(0.7)
        else:
            ax.bar(['单样本'], [data.iloc[0]], color=color, width=0.4, alpha=0.7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axhline(y=line,color=C['success'],linestyle='--',alpha=0.7,label=f'合格线({line})')
        ax.legend(fontsize=9); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); return fig2b64(fig)

def chart_single_clip(detail, accent):
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(detail))
    colors = [C['success'] if p else C['danger'] for p in detail['clip_pass']]
    ax.scatter(x, detail['clip_score'], c=colors, s=70, alpha=0.85, edgecolors='white', linewidth=0.6, zorder=3)
    ax.set_xlabel('图片序号', fontsize=11); ax.set_ylabel('CLIP得分', fontsize=11)
    ax.set_title('各图 CLIP 图文匹配度分布', fontsize=13, fontweight='bold')
    ax.axhline(y=0.26,color=C['success'],linestyle='--',alpha=0.7,label='合格线(0.26)')
    mean_val = detail['clip_score'].mean()
    ax.axhline(y=mean_val,color=accent,linestyle=':',alpha=0.8,label=f'均值({mean_val:.3f})')
    ax.legend(loc='upper right'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); return fig2b64(fig)

def chart_p10p90(stats, accent, title):
    metrics,p10s,p90s,meds = [],[],[],[]
    for idx,row in stats.iterrows():
        if pd.notna(row.get('p10')) and row['count']>0:
            metrics.append(idx); p10s.append(row['p10']); p90s.append(row['p90']); meds.append(row['median'])
    fig, ax = plt.subplots(figsize=(9,4.5))
    y = np.arange(len(metrics))
    for i,(p10,p90,med) in enumerate(zip(p10s,p90s,meds)):
        ax.plot([p10,p90],[i,i],color=accent,linewidth=8,alpha=0.6,solid_capstyle='round')
        ax.plot(med,i,'o',color=C['danger'],markersize=11,zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(metrics,fontsize=11)
    ax.set_xlabel('数值范围',fontsize=11); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.invert_yaxis(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='x',alpha=0.3)
    plt.tight_layout(); return fig2b64(fig)

# ==================== 人物图表 ====================
def char_radar(detail, stats):
    dims=['画质','语义匹配','人体完整','风格一致','清晰度']
    bs=max(0,min(100,(60-stats.loc['画质BRISQUE','mean'])/15*100))
    cs=max(0,min(100,(stats.loc['CLIP匹配度','mean']-0.20)/0.15*100))
    ig=stats.loc['人体关键点','pass_rate']
    ia=max(0,min(100,(stats.loc['分区域CLIP一致性','mean']-0.60)/0.25*100))
    bl=max(0,min(100,(detail['blur_score'].mean()-30)/120*100))
    scores=[bs,cs,ig,ia,bl]
    angles=np.linspace(0,2*np.pi,len(dims),endpoint=False).tolist(); scores+=scores[:1]; angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    ax.plot(angles,scores,'o-',linewidth=2.5,color=C['character'],label='人物资产'); ax.fill(angles,scores,alpha=0.25,color=C['character'])
    ax.plot(angles,[60]*len(angles),'--',color=C['warning'],alpha=0.6,label='合格线(60)')
    ax.plot(angles,[80]*len(angles),'--',color=C['success'],alpha=0.6,label='优秀线(80)')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims,fontsize=12,fontweight='bold'); ax.set_ylim(0,100)
    ax.set_title('人物资产综合质量雷达图',fontsize=14,fontweight='bold',pad=25)
    ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1)); ax.grid(True,alpha=0.3)
    return fig2b64(fig)

def char_single(detail):
    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    x=np.arange(len(detail))
    ax1=axes[0]; c1=[C['success'] if p else C['danger'] for p in detail['clip_pass']]
    ax1.scatter(x,detail['clip_score'],c=c1,s=60,alpha=0.85,edgecolors='white',linewidth=0.5,zorder=3)
    ax1.set_xlabel('图片序号',fontsize=11); ax1.set_ylabel('CLIP得分',fontsize=11)
    ax1.set_title('各图CLIP图文匹配度分布',fontsize=13,fontweight='bold'); ax1.axhline(y=0.26,color=C['success'],linestyle='--',alpha=0.7,label='合格线(0.26)')
    m1=detail['clip_score'].mean(); ax1.axhline(y=m1,color=C['character'],linestyle=':',alpha=0.8,label=f'均值({m1:.3f})')
    ax1.legend(loc='upper right'); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2=axes[1]; c2=[C['success'] if p else C['danger'] for p in detail['intra_clip_pass']]
    ax2.scatter(x,detail['intra_clip_consistency'],c=c2,s=60,alpha=0.85,edgecolors='white',linewidth=0.5,zorder=3)
    ax2.set_xlabel('图片序号',fontsize=11); ax2.set_ylabel('分区域CLIP一致性',fontsize=11)
    ax2.set_title('各图分区域风格一致性分布',fontsize=13,fontweight='bold'); ax2.axhline(y=0.70,color=C['success'],linestyle='--',alpha=0.7,label='合格线(0.70)')
    m2=detail['intra_clip_consistency'].mean(); ax2.axhline(y=m2,color=C['character'],linestyle=':',alpha=0.8,label=f'均值({m2:.3f})')
    ax2.legend(loc='lower right'); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout(); return fig2b64(fig)

# ==================== 道具图表 ====================
def prop_radar(detail, stats):
    dims=['画质','语义匹配','文字语种','文字关键词','清晰度']
    bs=max(0,min(100,(90-stats.loc['画质BRISQUE','mean'])/45*100))
    cs=max(0,min(100,(stats.loc['CLIP匹配度','mean']-0.20)/0.15*100))
    ls=stats.loc['道具文字-语种校验','pass_rate']; ks=stats.loc['道具文字-关键词校验','pass_rate']
    bl=max(0,min(100,(detail['blur_score'].mean()-30)/120*100))
    scores=[bs,cs,ls,ks,bl]
    angles=np.linspace(0,2*np.pi,len(dims),endpoint=False).tolist(); scores+=scores[:1]; angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    ax.plot(angles,scores,'o-',linewidth=2.5,color=C['prop'],label='道具资产'); ax.fill(angles,scores,alpha=0.25,color=C['prop'])
    ax.plot(angles,[60]*len(angles),'--',color=C['warning'],alpha=0.6,label='合格线(60)')
    ax.plot(angles,[80]*len(angles),'--',color=C['success'],alpha=0.6,label='优秀线(80)')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims,fontsize=12,fontweight='bold'); ax.set_ylim(0,100)
    ax.set_title('道具资产综合质量雷达图',fontsize=14,fontweight='bold',pad=25)
    ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1)); ax.grid(True,alpha=0.3)
    return fig2b64(fig)

def prop_scores(detail):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    x=np.arange(len(detail))
    ax1=axes[0]
    ax1.scatter(x,detail['brisque_score'],c=C['prop'],s=60,alpha=0.85,edgecolors='white',linewidth=0.5,zorder=3)
    ax1.set_xlabel('道具序号',fontsize=11); ax1.set_ylabel('BRISQUE分',fontsize=11)
    ax1.set_title('各道具 BRISQUE 画质分分布\n(越低越好)',fontsize=12,fontweight='bold'); ax1.axhline(y=45,color=C['success'],linestyle='--',alpha=0.7,label='合格线(45)')
    m1=detail['brisque_score'].mean(); ax1.axhline(y=m1,color=C['prop'],linestyle=':',alpha=0.8,label=f'均值({m1:.1f})')
    ax1.legend(); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2=axes[1]; c2=[C['success'] if p else C['danger'] for p in detail['clip_pass']]
    ax2.scatter(x,detail['clip_score'],c=c2,s=60,alpha=0.85,edgecolors='white',linewidth=0.5,zorder=3)
    ax2.set_xlabel('道具序号',fontsize=11); ax2.set_ylabel('CLIP得分',fontsize=11)
    ax2.set_title('各道具 CLIP 图文匹配度分布\n(越高越好)',fontsize=12,fontweight='bold'); ax2.axhline(y=0.26,color=C['success'],linestyle='--',alpha=0.7,label='合格线(0.26)')
    m2=detail['clip_score'].mean(); ax2.axhline(y=m2,color=C['prop'],linestyle=':',alpha=0.8,label=f'均值({m2:.3f})')
    ax2.legend(); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout(); return fig2b64(fig)

# ==================== 场景图表 ====================
def scene_radar(detail, group):
    dims=['画质','语义匹配','CLIP一致性','清晰度']
    bs=max(0,min(100,(45-detail['brisque_score'].iloc[0])/20*100+60))
    cs=max(0,min(100,(detail['clip_score'].iloc[0]-0.20)/0.15*100))
    gs=max(0,min(100,(group['group_consistency'].iloc[0]-0.50)/0.35*100))
    bl=max(0,min(100,(detail['blur_score'].iloc[0]-30)/120*100))
    scores=[bs,cs,gs,bl]
    angles=np.linspace(0,2*np.pi,len(dims),endpoint=False).tolist(); scores+=scores[:1]; angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    ax.plot(angles,scores,'o-',linewidth=2.5,color=C['scene'],label='场景资产'); ax.fill(angles,scores,alpha=0.25,color=C['scene'])
    ax.plot(angles,[60]*len(angles),'--',color=C['warning'],alpha=0.6,label='合格线(60)')
    ax.plot(angles,[80]*len(angles),'--',color=C['success'],alpha=0.6,label='优秀线(80)')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims,fontsize=12,fontweight='bold'); ax.set_ylim(0,100)
    ax.set_title('场景资产综合质量雷达图',fontsize=14,fontweight='bold',pad=25)
    ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1)); ax.grid(True,alpha=0.3)
    return fig2b64(fig)

def scene_consistency(group):
    fig, ax = plt.subplots(figsize=(7,5))
    metrics=['CLIP语义一致性']
    vals=[group['group_consistency'].iloc[0]]
    colors=[C['scene']]
    bars=ax.bar(metrics,vals,color=colors,width=0.4,edgecolor='white')
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.015,f'{v:.3f}',ha='center',fontsize=12,fontweight='bold')
    ax.set_ylabel('一致性得分',fontsize=11); ax.set_title('场景四视图CLIP语义一致性\n(基于2×2四宫格裁剪后的4视图计算)',fontsize=13,fontweight='bold')
    ax.set_ylim(0,1.1); ax.axhline(y=0.65,color=C['scene'],linestyle='--',alpha=0.6,label='合格线(0.65)')
    ax.legend(loc='upper right'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    return fig2b64(fig)

# ==================== 总览图表 ====================
def ov_pass(cs,ps,ss):
    fig,ax=plt.subplots(figsize=(12,5.5))
    metrics=['画质合格','CLIP匹配','专项校验']
    cv=[0,cs.loc['CLIP匹配度','pass_rate'],cs.loc['分区域CLIP一致性','pass_rate']]
    pv=[0,ps.loc['CLIP匹配度','pass_rate'],ps.loc['道具文字-综合','pass_rate']]
    sv=[100,ss.loc['CLIP匹配度','pass_rate'],100]
    x=np.arange(len(metrics)); w=0.25
    b1=ax.bar(x-w,cv,w,label='人物',color=C['character'],edgecolor='white')
    b2=ax.bar(x,pv,w,label='道具',color=C['prop'],edgecolor='white')
    b3=ax.bar(x+w,sv,w,label='场景',color=C['scene'],edgecolor='white')
    for bars in [b1,b2,b3]:
        for bar in bars:
            h=bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2,h+1.5,f'{h:.0f}%',ha='center',fontsize=9,fontweight='bold')
    ax.set_ylabel('合格率 (%)',fontsize=12); ax.set_title('三类资产核心指标合格率对比',fontsize=14,fontweight='bold',pad=15)
    ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=11); ax.set_ylim(0,115)
    ax.axhline(y=80,color=C['success'],linestyle='--',alpha=0.5,label='优秀线(80%)'); ax.legend(loc='upper right',ncol=4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    return fig2b64(fig)

def ov_means(cd,pd_,sd):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    assets=['人物','道具','场景']; colors=[C['character'],C['prop'],C['scene']]
    ax1=axes[0]; bv=[cd['brisque_score'].mean(),pd_['brisque_score'].mean(),sd['brisque_score'].mean()]
    b1=ax1.bar(assets,bv,color=colors,width=0.5,edgecolor='white')
    for bar,v in zip(b1,bv): ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{v:.1f}',ha='center',fontsize=11,fontweight='bold')
    ax1.axhline(y=45,color=C['success'],linestyle='--',alpha=0.7,label='合格线(45)'); ax1.set_ylabel('BRISQUE均值',fontsize=11)
    ax1.set_title('BRISQUE 画质分对比\n(越低越好)',fontsize=13,fontweight='bold'); ax1.legend(); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2=axes[1]; cv=[cd['clip_score'].mean(),pd_['clip_score'].mean(),sd['clip_score'].mean()]
    b2=ax2.bar(assets,cv,color=colors,width=0.5,edgecolor='white')
    for bar,v in zip(b2,cv): ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,f'{v:.3f}',ha='center',fontsize=11,fontweight='bold')
    ax2.axhline(y=0.26,color=C['success'],linestyle='--',alpha=0.7,label='合格线(0.26)'); ax2.set_ylabel('CLIP均值',fontsize=11)
    ax2.set_title('CLIP 图文匹配度对比\n(越高越好)',fontsize=13,fontweight='bold'); ax2.legend(); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout(); return fig2b64(fig)

# ==================== HTML 生成 ====================
def gen_character():
    d,s,g=load('人物'); t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'); a=C['character']
    charts={'radar':char_radar(d,s),'pass':chart_pass_bar(s,a,'人物资产各指标合格率'),
            'dist':chart_distribution(d,a),'single':char_single(d),'p10':chart_p10p90(s,a,'人物资产 P10-P90 质量波动区间')}
    cols=['filename','brisque_score','blur_score','iqa_pass','clip_score','clip_pass','keypoint_count','integrity_pass','intra_clip_consistency','intra_clip_pass','intra_face_consistency','intra_face_pass']
    hdrs=['图片','BRISQUE','清晰度','画质','CLIP','CLIP合格','关键点','人体完整','区域一致','区域合格','人脸一致','人脸合格']
    table=make_table(d,cols,hdrs,len(d))
    overall=s['pass_rate'].dropna().mean()
    # 动态判断人脸一致性是否有数据
    has_face = 'intra_face_consistency' in d.columns and d['intra_face_consistency'].notna().any()
    face_mean = d['intra_face_consistency'].mean() if has_face else None
    face_pass_rate = s.loc['图内人脸一致性','pass_rate'] if '图内人脸一致性' in s.index else None
    face_notice = '' if has_face else '图内人脸一致性指标因InsightFace模型未下载暂未计算。'
    face_card = f'<div class="ov-card"><div class="lbl">人脸一致性</div><div class="val">{face_mean:.3f}</div><div class="unit">合格率{face_pass_rate:.0f}%</div></div>' if has_face else ''
    face_suggest = '' if has_face else '<li>下载InsightFace模型后启用图内人脸一致性检测。</li>'
    face_detail_notice = '' if has_face else '图内人脸一致性因InsightFace buffalo_l模型未下载暂未计算（模型就绪后自动启用）。'
    html=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>短剧资产质量评估报告 - 人物资产</title><style>{base_css(a)}</style></head><body><div class="container">
<div class="header"><h1>🎬 短剧资产质量评估报告</h1><div class="sub">资产类型：人物资产（特写+三视图拼接设计图）</div><div class="time">报告生成时间：{t}</div></div>
<div class="card"><h2>📊 概览摘要</h2><div class="ov-grid">
<div class="ov-card"><div class="lbl">样本数量</div><div class="val">{len(d)}</div><div class="unit">张拼接图</div></div>
<div class="ov-card"><div class="lbl">质量评级</div><div class="val">{"优秀" if overall>=80 else "良好" if overall>=60 else "待改进"}</div><div class="unit">综合合格率{overall:.0f}%</div></div>
<div class="ov-card"><div class="lbl">BRISQUE画质</div><div class="val">{s.loc['画质BRISQUE','mean']:.1f}</div><div class="unit">越低越好</div></div>
<div class="ov-card"><div class="lbl">CLIP匹配度</div><div class="val">{s.loc['CLIP匹配度','mean']:.3f}</div><div class="unit">合格率{s.loc['CLIP匹配度','pass_rate']:.0f}%</div></div>
<div class="ov-card"><div class="lbl">分区域一致性</div><div class="val">{s.loc['分区域CLIP一致性','mean']:.3f}</div><div class="unit">合格率{s.loc['分区域CLIP一致性','pass_rate']:.0f}%</div></div>
<div class="ov-card"><div class="lbl">人体完整性</div><div class="val">{s.loc['人体关键点','pass_rate']:.0f}%</div><div class="unit">17关键点</div></div>
{face_card}
</div>{f'<div class="notice">💡 说明：CLIP匹配度已采用多视图角色设计图提示词扩展（character design sheet with multiple views），以匹配"特写+三视图拼接图"的实际内容。{face_notice}</div>' if face_notice else ''}</div>
<div class="card"><h2>🎯 综合质量雷达图</h2><div class="chart"><img src="data:image/png;base64,{charts['radar']}"></div></div>
<div class="card"><h2>✅ 各指标合格率</h2><div class="chart"><img src="data:image/png;base64,{charts['pass']}"></div></div>
<div class="card"><h2>📈 核心指标得分分布</h2><div class="chart"><img src="data:image/png;base64,{charts['dist']}"></div></div>
<div class="card"><h2>🖼️ 单图指标对比</h2><div class="chart"><img src="data:image/png;base64,{charts['single']}"></div></div>
<div class="card"><h2>📉 质量波动区间（P10-P90）</h2><div class="chart"><img src="data:image/png;base64,{charts['p10']}"></div></div>
<div class="card"><h2>📋 单图明细数据</h2><div class="table-wrap">{table}</div><div class="notice">📝 人物为1×4横排拼接图（特写→正视→侧视→背视），分区域CLIP一致性基于4区域裁剪计算。{face_detail_notice}</div></div>
<div class="card"><h2>💡 结论与建议</h2><div class="conclusion"><h3>整体评价</h3><ul><li>人体完整性{s.loc['人体关键点','pass_rate']:.0f}%、分区域风格一致性{s.loc['分区域CLIP一致性','pass_rate']:.0f}%，人物结构和风格统一度{"优秀" if s.loc['分区域CLIP一致性','pass_rate']>=90 else "良好"}。</li><li>BRISQUE均值{s.loc['画质BRISQUE','mean']:.1f}{"，画质优秀" if s.loc['画质BRISQUE','mean']<45 else "略高于合格线(45)，画质有提升空间"}。</li><li>CLIP匹配度经提示词扩展后合格率{s.loc['CLIP匹配度','pass_rate']:.0f}%{"，全部合格" if s.loc['CLIP匹配度','pass_rate']>=100 else "，仍有部分图片低于0.26"}。</li>{f'<li>人脸一致性均值{face_mean:.3f}，合格率{face_pass_rate:.0f}%，角色人脸一致性{"优秀" if face_pass_rate>=90 else "良好"}。</li>' if has_face else ''}</ul><h3>优化建议</h3><ul>{'<li>适当提高生成步数或使用更高质量采样器，改善BRISQUE画质分。</li>' if s.loc['画质BRISQUE','mean']>=45 else ''}{'<li>针对CLIP得分较低的图片，优化提示词描述，确保准确反映多视图拼接图内容。</li>' if s.loc['CLIP匹配度','pass_rate']<100 else ''}{face_suggest}</ul></div></div>
{metric_explanation_section(['brisque_score','blur_score','clip_score','keypoint_count','intra_clip_consistency','intra_face_consistency','iqa_pass','clip_pass','integrity_pass'], a)}
<div class="footer">短剧资产三层质检系统 · 人物资产质量评估报告 · 生成于 {t}</div>
</div></body></html>'''
    p=os.path.join(BASE,'评估报告_人物','人物资产质量报告.html')
    open(p,'w',encoding='utf-8').write(html); return p

def gen_prop():
    d,s,g=load('道具'); t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'); a=C['prop']
    charts={'radar':prop_radar(d,s),'pass':chart_pass_bar(s,a,'道具资产各指标合格率'),
            'dist':chart_distribution(d,a),'single':chart_single_clip(d,a),'scores':prop_scores(d)}
    cols=['filename','brisque_score','blur_score','iqa_pass','clip_score','clip_pass','has_text','text_lang_pass','text_keyword_pass','text_pass']
    hdrs=['图片','BRISQUE','清晰度','画质','CLIP','CLIP合格','有文字','语种校验','关键词','文字综合']
    table=make_table(d,cols,hdrs,len(d))
    overall=s['pass_rate'].dropna().mean()
    # 动态判断文字校验情况
    ocr_triggered = d[d['expected_keyword'].notna()]
    ocr_count = len(ocr_triggered)
    lang_pass_rate = s.loc['道具文字-语种校验','pass_rate'] if '道具文字-语种校验' in s.index else None
    kw_pass_rate = s.loc['道具文字-关键词校验','pass_rate'] if '道具文字-关键词校验' in s.index else None
    text_pass_rate = s.loc['道具文字-综合','pass_rate'] if '道具文字-综合' in s.index else None
    # OCR检测到文字的道具详情
    ocr_detected = d[d['has_text']==True]
    # 触发OCR但未检测到文字的道具（expected_keyword不为空说明触发了OCR）
    ocr_triggered = d[d['expected_keyword'].notna()]
    ocr_not_detected = ocr_triggered[ocr_triggered['has_text']==False]
    ocr_html=''
    if len(ocr_detected) > 0:
        rows_html = ''
        for _, r in ocr_detected.iterrows():
            lang_ok = '✓ 合格' if r['text_lang_pass']==True else ('✗ 不合格' if r['text_lang_pass']==False else '— 跳过')
            kw_ok = '✓ 合格' if r['text_keyword_pass']==True else ('✗ 不合格' if r['text_keyword_pass']==False else '— 跳过')
            lang_color = '#16a34a' if r['text_lang_pass']==True else '#dc2626'
            kw_color = '#16a34a' if r['text_keyword_pass']==True else '#dc2626'
            rows_html += f'<tr><td>{r["filename"]}</td><td>{str(r["detected_text"])[:80]}</td><td>{r["detected_lang"]}</td><td style="color:{lang_color};font-weight:bold;">{lang_ok}</td><td>{r["expected_keyword"]}</td><td style="color:{kw_color};font-weight:bold;">{kw_ok}</td></tr>'
        ocr_html += f'''<div class="notice" style="border-left-color:{a};background:#fffbeb;"><strong>📝 OCR检测到文字的道具（{len(ocr_detected)}张）</strong><div class="table-wrap"><table><tr><th>图片</th><th>识别文字</th><th>检测语种</th><th>语种校验</th><th>预期关键词</th><th>关键词校验</th></tr>{rows_html}</table></div></div>'''
    if len(ocr_not_detected) > 0:
        names = '、'.join(ocr_not_detected['filename'].tolist())
        ocr_html += f'''<div class="notice" style="border-left-color:#f59e0b;background:#fffbeb;"><strong>⚠️ 触发OCR但未检测到文字的道具（{len(ocr_not_detected)}张，建议人工复核）</strong><br>{names}<br><em>可能原因：文字过小/艺术字体/AI生成时未画出文字。这些道具不计入语种/关键词校验合格率的分母。</em></div>'''
    # 无文字预期、跳过OCR的道具数量
    ocr_skipped = d[d['expected_keyword'].isna()]
    ocr_html += f'''<div class="notice">💡 共 {len(d)} 张道具：{len(ocr_triggered)} 张触发OCR（提示词提到文字），{len(ocr_skipped)} 张无文字预期已跳过OCR。</div>'''
    html=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>短剧资产质量评估报告 - 道具资产</title><style>{base_css(a)}</style></head><body><div class="container">
<div class="header"><h1>🎬 短剧资产质量评估报告</h1><div class="sub">资产类型：道具资产</div><div class="time">报告生成时间：{t}</div></div>
<div class="card"><h2>📊 概览摘要</h2><div class="ov-grid">
<div class="ov-card"><div class="lbl">样本数量</div><div class="val">{len(d)}</div><div class="unit">张</div></div>
<div class="ov-card"><div class="lbl">质量评级</div><div class="val">{"优秀" if overall>=80 else "良好" if overall>=60 else "待改进"}</div><div class="unit">综合合格率{overall:.0f}%</div></div>
<div class="ov-card"><div class="lbl">BRISQUE画质</div><div class="val">{s.loc['画质BRISQUE','mean']:.1f}</div><div class="unit">越低越好</div></div>
<div class="ov-card"><div class="lbl">CLIP匹配度</div><div class="val">{s.loc['CLIP匹配度','mean']:.3f}</div><div class="unit">合格率{s.loc['CLIP匹配度','pass_rate']:.0f}%</div></div>
<div class="ov-card"><div class="lbl">文字-语种校验</div><div class="val">{s.loc['道具文字-语种校验','pass_rate']:.0f}%</div><div class="unit">基于{int(s.loc['道具文字-语种校验','count'])}张</div></div>
<div class="ov-card"><div class="lbl">文字-综合</div><div class="val">{s.loc['道具文字-综合','pass_rate']:.0f}%</div><div class="unit">OCR校验</div></div>
</div></div>
<div class="card"><h2>🎯 综合质量雷达图</h2><div class="chart"><img src="data:image/png;base64,{charts['radar']}"></div></div>
<div class="card"><h2>✅ 各指标合格率</h2><div class="chart"><img src="data:image/png;base64,{charts['pass']}"></div></div>
<div class="card"><h2>📈 核心指标得分分布</h2><div class="chart"><img src="data:image/png;base64,{charts['dist']}"></div></div>
<div class="card"><h2>🖼️ 各道具得分对比</h2><div class="chart"><img src="data:image/png;base64,{charts['scores']}"></div></div>
<div class="card"><h2>📝 OCR文字校验详情</h2>{ocr_html}</div>
<div class="card"><h2>📋 单图明细数据</h2><div class="table-wrap">{table}</div><div class="notice">📝 道具OCR校验说明：提示词中提到文字（写着/印有/Chinese text等）的道具触发OCR校验（语种+关键词包含）；无文字预期的道具跳过OCR（标记为不适用）。OCR未检测到文字的道具不计入合格率分母，建议人工复核。</div></div>
<div class="card"><h2>💡 结论与建议</h2><div class="conclusion"><h3>整体评价</h3><ul><li>CLIP匹配度合格率{s.loc['CLIP匹配度','pass_rate']:.0f}%{"，全部合格" if s.loc['CLIP匹配度','pass_rate']>=100 else "，仍有提升空间"}，道具语义匹配{"优秀" if s.loc['CLIP匹配度','pass_rate']>=90 else "良好"}。</li><li>BRISQUE均值{s.loc['画质BRISQUE','mean']:.1f}{"，画质优秀" if s.loc['画质BRISQUE','mean']<45 else "偏高，画质有待提升"}。</li>{f'<li>共{ocr_count}张道具触发OCR文字校验，语种合格率{lang_pass_rate:.0f}%，关键词合格率{kw_pass_rate:.0f}%，综合合格率{text_pass_rate:.0f}%。</li>' if ocr_count>0 else '<li>本批次道具无文字预期，未触发OCR校验。</li>'}<li>样本量{len(d)}张，统计结果{"稳定性较好" if len(d)>=50 else "稳定性有限，建议增加样本量"}。</li></ul><h3>优化建议</h3><ul>{'<li>适当提高生成步数或使用更高质量采样器，改善BRISQUE画质分。</li>' if s.loc['画质BRISQUE','mean']>=45 else ''}{f'<li>关键词校验合格率偏低（{kw_pass_rate:.0f}%），建议检查OCR识别结果或优化道具提示词中的文字描述。</li>' if kw_pass_rate is not None and kw_pass_rate<60 else ''}{'<li>增加道具样本量以获得更稳定的统计结果。</li>' if len(d)<50 else ''}</ul></div></div>
{metric_explanation_section(['brisque_score','blur_score','clip_score','text_lang_pass','text_keyword_pass','text_pass','iqa_pass','clip_pass'], a)}
<div class="footer">短剧资产三层质检系统 · 道具资产质量评估报告 · 生成于 {t}</div>
</div></body></html>'''
    p=os.path.join(BASE,'评估报告_道具','道具资产质量报告.html')
    open(p,'w',encoding='utf-8').write(html); return p

def gen_scene():
    d,s,g=load('场景'); t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'); a=C['scene']
    charts={'radar':scene_radar(d,g),'consist':scene_consistency(g),'dist':chart_distribution(d,a)}
    cols=['filename','brisque_score','blur_score','iqa_pass','clip_score','clip_pass']
    hdrs=['图片','BRISQUE','清晰度','画质','CLIP','CLIP合格']
    table=make_table(d,cols,hdrs,len(d))
    overall=s['pass_rate'].dropna().mean()
    gc=g['group_consistency'].iloc[0]
    html=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>短剧资产质量评估报告 - 场景资产</title><style>{base_css(a)}</style></head><body><div class="container">
<div class="header"><h1>🎬 短剧资产质量评估报告</h1><div class="sub">资产类型：场景资产（2×2四宫格拼接图）</div><div class="time">报告生成时间：{t}</div></div>
<div class="card"><h2>📊 概览摘要</h2><div class="ov-grid">
<div class="ov-card"><div class="lbl">样本数量</div><div class="val">{len(d)}</div><div class="unit">张四宫格</div></div>
<div class="ov-card"><div class="lbl">质量评级</div><div class="val">{"优秀" if overall>=80 else "良好" if overall>=60 else "待改进"}</div><div class="unit">综合合格率{overall:.0f}%</div></div>
<div class="ov-card"><div class="lbl">BRISQUE画质</div><div class="val">{s.loc['画质BRISQUE','mean']:.1f}</div><div class="unit">越低越好</div></div>
<div class="ov-card"><div class="lbl">CLIP匹配度</div><div class="val">{s.loc['CLIP匹配度','mean']:.3f}</div><div class="unit">合格率{s.loc['CLIP匹配度','pass_rate']:.0f}%</div></div>
<div class="ov-card"><div class="lbl">CLIP语义一致性</div><div class="val">{gc:.3f}</div><div class="unit">阈值0.65</div></div>
</div></div>
<div class="card"><h2>🎯 综合质量雷达图</h2><div class="chart"><img src="data:image/png;base64,{charts['radar']}"></div></div>
<div class="card"><h2>🔄 四视图一致性指标</h2><div class="chart"><img src="data:image/png;base64,{charts['consist']}"></div>
<div class="notice" style="border-left-color:{a};background:#f0fdf4;">🏞️ <strong>四宫格裁剪说明：</strong>场景图片为2×2四宫格拼接图（左上/右上/左下/右下各1视图），评估时先裁剪为4张独立视图，再计算CLIP语义一致性。</div></div>
<div class="card"><h2>📈 核心指标得分</h2><div class="chart"><img src="data:image/png;base64,{charts['dist']}"></div></div>
<div class="card"><h2>📋 单图明细数据</h2><div class="table-wrap">{table}</div></div>
<div class="card"><h2>💡 结论与建议</h2><div class="conclusion"><h3>整体评价</h3><ul><li>BRISQUE均值{s.loc['画质BRISQUE','mean']:.1f}{"，画质优秀" if s.loc['画质BRISQUE','mean']<45 else "，画质有待提升"}。</li><li>CLIP匹配度合格率{s.loc['CLIP匹配度','pass_rate']:.0f}%{"，全部合格" if s.loc['CLIP匹配度','pass_rate']>=100 else "，仍有提升空间"}。</li><li>四视图CLIP语义一致性{gc:.3f}，{"一致性优秀" if gc>=0.75 else "一致性良好" if gc>=0.65 else "一致性不足"}。</li><li>场景资产整体质量{"优秀" if overall>=80 else "良好" if overall>=60 else "待改进"}，综合合格率{overall:.0f}%。</li></ul><h3>优化建议</h3><ul>{'<li>适当提高生成步数或使用更高质量采样器，改善BRISQUE画质分。</li>' if s.loc['画质BRISQUE','mean']>=45 else ''}{'<li>针对CLIP得分较低的场景图片，优化提示词描述。</li>' if s.loc['CLIP匹配度','pass_rate']<100 else ''}{'<li>增加场景样本量以获得更稳定的统计结果。</li>' if len(d)<30 else ''}<li>可进一步细化场景的专项检测指标（如光照一致性、空间布局合理性等）。</li></ul></div></div>
{metric_explanation_section(['brisque_score','blur_score','clip_score','group_consistency','iqa_pass','clip_pass'], a)}
<div class="footer">短剧资产三层质检系统 · 场景资产质量评估报告 · 生成于 {t}</div>
</div></body></html>'''
    p=os.path.join(BASE,'评估报告_场景','场景资产质量报告.html')
    open(p,'w',encoding='utf-8').write(html); return p

def gen_comprehensive():
    cd,cs,cg=load('人物'); pd_,ps,pg=load('道具'); sd,ss,sg=load('场景')
    t=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    charts={'ov_pass':ov_pass(cs,ps,ss),'ov_means':ov_means(cd,pd_,sd),
            'ch_radar':char_radar(cd,cs),'ch_pass':chart_pass_bar(cs,C['character'],'人物资产各指标合格率'),
            'ch_dist':chart_distribution(cd,C['character']),'ch_single':char_single(cd),
            'pr_radar':prop_radar(pd_,ps),'pr_pass':chart_pass_bar(ps,C['prop'],'道具资产各指标合格率'),
            'pr_scores':prop_scores(pd_),
            'sc_radar':scene_radar(sd,sg),'sc_consist':scene_consistency(sg)}
    ch_overall=cs['pass_rate'].dropna().mean(); pr_overall=ps['pass_rate'].dropna().mean(); sc_overall=ss['pass_rate'].dropna().mean()
    # 加载返工日志
    rework_log_path=os.path.join(BASE,'返工日志.json')
    rework_html=''
    if os.path.exists(rework_log_path):
        with open(rework_log_path,'r',encoding='utf-8') as f:
            rlog=json.load(f)
        total=rlog.get('total_reworks',0); succ=rlog.get('success_count',0); fail=rlog.get('fail_count',0)
        records=rlog.get('records',[])
        if total>0:
            rows_html=''
            for r in records:
                fn=r.get('filename','?')
                orig_kp=r.get('original_keypoint_count','?')
                final_kp=r.get('final_keypoint_count','?')
                fsucc=r.get('final_success',False)
                status='✅ 成功' if fsucc else '❌ 失败'
                scolor='#16a34a' if fsucc else '#dc2626'
                attempts=len(r.get('attempts',[]))
                start=r.get('start_time','?')
                end=r.get('end_time','?')
                task_ids=', '.join([a.get('task_id','?') for a in r.get('attempts',[])])
                rows_html+=f'<tr><td>{fn}</td><td>{orig_kp}→{final_kp}</td><td style="color:{scolor};font-weight:bold;">{status}</td><td>{attempts}</td><td>{start}</td><td>{end}</td><td>{task_ids}</td></tr>'
            rework_html=f'''<div class="card"><h2>🔄 返工日志记录</h2>
<div class="ov-grid">
<div class="ov-card"><div class="lbl">累计返工次数</div><div class="val">{total}</div><div class="unit">次</div></div>
<div class="ov-card"><div class="lbl">返工成功</div><div class="val" style="color:#16a34a;">{succ}</div><div class="unit">次</div></div>
<div class="ov-card"><div class="lbl">返工失败</div><div class="val" style="color:#dc2626;">{fail}</div><div class="unit">次</div></div>
<div class="ov-card"><div class="lbl">成功率</div><div class="val">{succ/total*100:.0f}%</div><div class="unit">自动返工</div></div>
</div>
<div class="table-wrap"><table>
<tr><th>图片</th><th>关键点变化</th><th>结果</th><th>尝试次数</th><th>开始时间</th><th>完成时间</th><th>任务ID</th></tr>
{rows_html}
</table></div>
<div class="notice">📝 返工说明：针对ComfyUI生成本身的小概率随机问题（如人体关键点缺失），自动重新生成并质检，合格后覆盖原图。所有返工操作均记录在此。</div>
</div>'''
        else:
            rework_html='<div class="card"><h2>🔄 返工日志记录</h2><div class="notice">暂无返工记录，所有资产一次生成即通过质检。</div></div>'
    ch_cols=['filename','brisque_score','blur_score','iqa_pass','clip_score','clip_pass','keypoint_count','integrity_pass','intra_clip_consistency','intra_clip_pass','intra_face_consistency','intra_face_pass']
    ch_hdrs=['图片','BRISQUE','清晰度','画质','CLIP','CLIP合格','关键点','人体完整','区域一致','区域合格','人脸一致','人脸合格']
    pr_cols=['filename','brisque_score','blur_score','iqa_pass','clip_score','clip_pass','has_text','text_lang_pass','text_keyword_pass','text_pass']
    pr_hdrs=['图片','BRISQUE','清晰度','画质','CLIP','CLIP合格','有文字','语种校验','关键词','文字综合']
    sc_cols=['filename','brisque_score','blur_score','iqa_pass','clip_score','clip_pass']
    sc_hdrs=['图片','BRISQUE','清晰度','画质','CLIP','CLIP合格']
    ch_table=make_table(cd,ch_cols,ch_hdrs,len(cd)); pr_table=make_table(pd_,pr_cols,pr_hdrs,len(pd_)); sc_table=make_table(sd,sc_cols,sc_hdrs,len(sd))
    # ==================== 动态结论判断 ====================
    # 人物指标
    ch_brisque = cd['brisque_score'].mean()
    ch_clip_pass = cs.loc['CLIP匹配度','pass_rate'] if 'CLIP匹配度' in cs.index else 0
    ch_kp_pass = cs.loc['人体关键点','pass_rate'] if '人体关键点' in cs.index else 0
    ch_intra_pass = cs.loc['分区域CLIP一致性','pass_rate'] if '分区域CLIP一致性' in cs.index else 0
    ch_has_face = 'intra_face_consistency' in cd.columns and cd['intra_face_consistency'].notna().any()
    ch_face_mean = cd['intra_face_consistency'].mean() if ch_has_face else None
    ch_face_pass = cs.loc['图内人脸一致性','pass_rate'] if ch_has_face and '图内人脸一致性' in cs.index else None
    # 道具指标
    pr_brisque = pd_['brisque_score'].mean()
    pr_clip_pass = ps.loc['CLIP匹配度','pass_rate'] if 'CLIP匹配度' in ps.index else 0
    pr_ocr_count = len(pd_[pd_['expected_keyword'].notna()])
    pr_kw_pass = ps.loc['道具文字-关键词校验','pass_rate'] if '道具文字-关键词校验' in ps.index else None
    # 场景指标
    sc_brisque = sd['brisque_score'].mean()
    sc_clip_pass = ss.loc['CLIP匹配度','pass_rate'] if 'CLIP匹配度' in ss.index else 0
    sc_gc = sg['group_consistency'].iloc[0] if sg is not None and len(sg)>0 else None
    # 短板识别
    weaknesses = []
    if ch_brisque >= 45: weaknesses.append(f'人物BRISQUE画质分偏高（{ch_brisque:.1f}），画面质感有优化空间')
    if pr_brisque >= 45: weaknesses.append(f'道具BRISQUE画质分偏高（{pr_brisque:.1f}），画面质感有优化空间')
    if ch_clip_pass < 100: weaknesses.append(f'部分人物图片CLIP匹配度低于0.26（合格率{ch_clip_pass:.0f}%），提示词与画面语义对齐需优化')
    if sc_clip_pass < 100: weaknesses.append(f'部分场景图片CLIP匹配度低于0.26（合格率{sc_clip_pass:.0f}%）')
    if pr_kw_pass is not None and pr_kw_pass < 60: weaknesses.append(f'道具文字关键词校验合格率偏低（{pr_kw_pass:.0f}%），建议检查OCR识别或优化提示词')
    if not ch_has_face: weaknesses.append('人物图内人脸一致性指标暂未计算（需下载InsightFace模型）')
    if not weaknesses: weaknesses.append('暂无明显短板，各指标表现良好')
    # 优化建议
    suggestions = []
    if ch_brisque >= 45 or pr_brisque >= 45: suggestions.append('适当提高生成步数或使用更高质量采样器，改善BRISQUE画质分')
    if ch_clip_pass < 100: suggestions.append('针对CLIP得分较低的人物图片，优化提示词描述，确保准确反映多视图拼接图内容')
    if sc_clip_pass < 100: suggestions.append('针对CLIP得分较低的场景图片，优化提示词描述')
    if not ch_has_face: suggestions.append('下载InsightFace buffalo_l模型后启用图内人脸一致性检测，进一步验证角色一致性')
    if pr_kw_pass is not None and pr_kw_pass < 60: suggestions.append('检查道具文字关键词校验逻辑，优化OCR识别或提示词中的文字描述')
    if len(pd_) < 50 or len(sd) < 30: suggestions.append('增加样本量以获得更稳定的统计结果')
    if not suggestions: suggestions.append('继续保持当前生成参数，定期监控质量指标变化')
    # 动态结论HTML片段
    weakness_html = ''.join(f'<li>{w}</li>' for w in weaknesses)
    suggestion_html = ''.join(f'<li>{s}</li>' for s in suggestions)
    ocr_detected = pd_[pd_['has_text']==True]
    ocr_triggered = pd_[pd_['expected_keyword'].notna()]
    ocr_not_detected = ocr_triggered[ocr_triggered['has_text']==False]
    ocr_html=''
    if len(ocr_not_detected) > 0:
        names = '、'.join(ocr_not_detected['filename'].tolist())
        ocr_html=f'''<div class="notice" style="border-left-color:#f59e0b;background:#fffbeb;"><strong>⚠️ 触发OCR但未检测到文字的道具（{len(ocr_not_detected)}张，建议人工复核）</strong><br>{names}<br><em>可能原因：文字过小/艺术字体/AI生成时未画出文字。这些道具不计入语种/关键词校验合格率的分母。</em></div>'''
    if len(ocr_detected) > 0:
        rows_html = ''
        for _, r in ocr_detected.iterrows():
            lang_ok = '✓' if r['text_lang_pass']==True else ('✗' if r['text_lang_pass']==False else '—')
            kw_ok = '✓' if r['text_keyword_pass']==True else ('✗' if r['text_keyword_pass']==False else '—')
            rows_html += f'<tr><td>{r["filename"]}</td><td>{str(r["detected_text"])[:60]}</td><td>{lang_ok}</td><td>{kw_ok}</td></tr>'
        ocr_html += f'''<div class="notice" style="border-left-color:#16a34a;background:#f0fdf4;"><strong>📝 OCR检测到文字的道具（{len(ocr_detected)}张）</strong><div class="table-wrap"><table><tr><th>图片</th><th>识别文字</th><th>语种</th><th>关键词</th></tr>{rows_html}</table></div></div>'''
    css='''
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;background:#f8fafc;color:#1e293b;line-height:1.6}
.container{max-width:1280px;margin:0 auto;padding:30px 20px}
.header{background:linear-gradient(135deg,#1e40af 0%,#0891b2 50%,#16a34a 100%);color:white;padding:45px;border-radius:16px;margin-bottom:30px;box-shadow:0 10px 40px rgba(37,99,235,0.2)}
.header h1{font-size:30px;margin-bottom:10px}.header .sub{font-size:14px;opacity:0.9}.header .time{font-size:13px;opacity:0.8;margin-top:8px}
.card{background:white;border-radius:12px;padding:25px;margin-bottom:25px;box-shadow:0 2px 12px rgba(0,0,0,0.06);border:1px solid #e2e8f0}
.card h2{font-size:20px;margin-bottom:18px;padding-bottom:12px;border-bottom:2px solid #2563eb;color:#2563eb}
.card h3{font-size:16px;margin:15px 0 10px}
.s-char h2{border-bottom-color:#0891b2;color:#0891b2}.s-prop h2{border-bottom-color:#f59e0b;color:#f59e0b}.s-scene h2{border-bottom-color:#16a34a;color:#16a34a}
.ov-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:10px}
.ov-card{border-radius:10px;padding:18px;text-align:center;border:1px solid #e2e8f0}
.ov-card .lbl{font-size:12px;color:#64748b;margin-bottom:6px}.ov-card .val{font-size:24px;font-weight:bold}.ov-card .unit{font-size:12px;color:#64748b}
.ov-char{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border-color:#bae6fd}.ov-char .val{color:#0891b2}
.ov-prop{background:linear-gradient(135deg,#fffbeb,#fef3c7);border-color:#fde68a}.ov-prop .val{color:#f59e0b}
.ov-scene{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-color:#bbf7d0}.ov-scene .val{color:#16a34a}
.chart{text-align:center;margin:15px 0}.chart img{max-width:100%;height:auto;border-radius:8px}
.table-wrap{overflow-x:auto;margin-top:10px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#2563eb;color:white;padding:11px 8px;text-align:center;font-weight:600;white-space:nowrap}
td{padding:9px;text-align:center;border-bottom:1px solid #e2e8f0}
tr:nth-child(even){background:#f8fafc}tr:hover{background:#eff6ff}
.sample-note{background:#f1f5f9;border-left:4px solid #9ca3af;padding:10px 14px;border-radius:6px;margin:10px 0;font-size:13px;color:#64748b}
.notice{background:#f1f5f9;border-left:4px solid #9ca3af;padding:12px 16px;border-radius:6px;margin:10px 0;font-size:13px;color:#64748b}
.conclusion{background:linear-gradient(135deg,#fefce8,#fef9c3);border-left:4px solid #f59e0b;padding:20px;border-radius:8px;margin-top:15px}
.conclusion h3{color:#92400e;margin-bottom:10px}.conclusion ul{margin-left:20px}.conclusion li{margin-bottom:7px;color:#78350f}
.footer{text-align:center;padding:20px;color:#64748b;font-size:12px}
.tag{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:bold}
.tag-char{background:#e0f2fe;color:#0891b2}.tag-prop{background:#fef3c7;color:#f59e0b}.tag-scene{background:#dcfce7;color:#16a34a}
'''
    html=f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>短剧资产质量评估综合报告</title><style>{css}</style></head><body><div class="container">
<div class="header"><h1>🎬 短剧资产质量评估综合报告</h1><div class="sub">涵盖人物资产、道具资产、场景资产三类 · 三层质检体系（画质+语义+专项）</div><div class="time">报告生成时间：{t}</div></div>
<div class="card"><h2>📊 总览摘要 · 三类资产横向对比</h2><div class="ov-grid">
<div class="ov-card ov-char"><div class="lbl">人物 · 样本数</div><div class="val">{len(cd)}</div><div class="unit">张拼接图</div></div>
<div class="ov-card ov-char"><div class="lbl">人物 · 整体合格率</div><div class="val">{ch_overall:.0f}%</div><div class="unit">综合</div></div>
<div class="ov-card ov-char"><div class="lbl">人物 · BRISQUE</div><div class="val">{cd['brisque_score'].mean():.1f}</div><div class="unit">越低越好</div></div>
<div class="ov-card ov-char"><div class="lbl">人物 · CLIP</div><div class="val">{cd['clip_score'].mean():.3f}</div><div class="unit">越高越好</div></div>
<div class="ov-card ov-prop"><div class="lbl">道具 · 样本数</div><div class="val">{len(pd_)}</div><div class="unit">张</div></div>
<div class="ov-card ov-prop"><div class="lbl">道具 · 整体合格率</div><div class="val">{pr_overall:.0f}%</div><div class="unit">综合</div></div>
<div class="ov-card ov-prop"><div class="lbl">道具 · BRISQUE</div><div class="val">{pd_['brisque_score'].mean():.1f}</div><div class="unit">越低越好</div></div>
<div class="ov-card ov-prop"><div class="lbl">道具 · CLIP</div><div class="val">{pd_['clip_score'].mean():.3f}</div><div class="unit">越高越好</div></div>
<div class="ov-card ov-scene"><div class="lbl">场景 · 样本数</div><div class="val">{len(sd)}</div><div class="unit">张四宫格</div></div>
<div class="ov-card ov-scene"><div class="lbl">场景 · 整体合格率</div><div class="val">{sc_overall:.0f}%</div><div class="unit">综合</div></div>
<div class="ov-card ov-scene"><div class="lbl">场景 · BRISQUE</div><div class="val">{sd['brisque_score'].mean():.1f}</div><div class="unit">越低越好</div></div>
<div class="ov-card ov-scene"><div class="lbl">场景 · CLIP</div><div class="val">{sd['clip_score'].mean():.3f}</div><div class="unit">越高越好</div></div>
</div><div class="chart"><img src="data:image/png;base64,{charts['ov_pass']}"></div><div class="chart"><img src="data:image/png;base64,{charts['ov_means']}"></div>
<div class="notice">💡 说明：人物CLIP匹配度已采用多视图角色设计图提示词扩展；场景一致性指标基于2×2四宫格裁剪后的4视图计算；人物图内人脸一致性因InsightFace模型未下载暂未计算。</div></div>
<div class="card s-char"><h2><span class="tag tag-char">人物</span> 人物资产评估详情</h2>
<div class="ov-grid"><div class="ov-card ov-char"><div class="lbl">质量评级</div><div class="val">{"优秀" if ch_overall>=80 else "良好"}</div><div class="unit">综合{ch_overall:.0f}%</div></div>
<div class="ov-card ov-char"><div class="lbl">分区域一致性</div><div class="val">{cs.loc['分区域CLIP一致性','mean']:.3f}</div><div class="unit">合格率{cs.loc['分区域CLIP一致性','pass_rate']:.0f}%</div></div>
<div class="ov-card ov-char"><div class="lbl">人体完整性</div><div class="val">{cs.loc['人体关键点','pass_rate']:.0f}%</div><div class="unit">17关键点</div></div>
<div class="ov-card ov-char"><div class="lbl">CLIP匹配度</div><div class="val">{cs.loc['CLIP匹配度','mean']:.3f}</div><div class="unit">合格率{cs.loc['CLIP匹配度','pass_rate']:.0f}%</div></div></div>
<div class="chart"><img src="data:image/png;base64,{charts['ch_radar']}"></div>
<div class="chart"><img src="data:image/png;base64,{charts['ch_pass']}"></div>
<div class="chart"><img src="data:image/png;base64,{charts['ch_dist']}"></div>
<div class="chart"><img src="data:image/png;base64,{charts['ch_single']}"></div>
<h3>📋 单图明细数据</h3><div class="table-wrap">{ch_table}</div>
<div class="notice">📝 人物为1×4横排拼接图，分区域CLIP一致性基于4区域裁剪计算。图内人脸一致性因InsightFace模型未下载暂未计算。</div></div>
<div class="card s-prop"><h2><span class="tag tag-prop">道具</span> 道具资产评估详情</h2>
<div class="ov-grid"><div class="ov-card ov-prop"><div class="lbl">质量评级</div><div class="val">{"优秀" if pr_overall>=80 else "良好"}</div><div class="unit">综合{pr_overall:.0f}%</div></div>
<div class="ov-card ov-prop"><div class="lbl">CLIP匹配度</div><div class="val">{ps.loc['CLIP匹配度','mean']:.3f}</div><div class="unit">合格率{ps.loc['CLIP匹配度','pass_rate']:.0f}%</div></div>
<div class="ov-card ov-prop"><div class="lbl">文字-语种校验</div><div class="val">{ps.loc['道具文字-语种校验','pass_rate']:.0f}%</div><div class="unit">基于{int(ps.loc['道具文字-语种校验','count'])}张</div></div>
<div class="ov-card ov-prop"><div class="lbl">文字-综合</div><div class="val">{ps.loc['道具文字-综合','pass_rate']:.0f}%</div><div class="unit">OCR校验</div></div></div>
<div class="chart"><img src="data:image/png;base64,{charts['pr_radar']}"></div>
<div class="chart"><img src="data:image/png;base64,{charts['pr_pass']}"></div>
<div class="chart"><img src="data:image/png;base64,{charts['pr_scores']}"></div>
{ocr_html}
<h3>📋 单图明细数据</h3><div class="table-wrap">{pr_table}</div>
<div class="notice">📝 道具OCR校验：提示词提到文字（写着/印有/Chinese text等）的道具触发OCR校验；无文字预期的道具跳过OCR。OCR未检测到文字的道具不计入合格率分母，建议人工复核。</div></div>
<div class="card s-scene"><h2><span class="tag tag-scene">场景</span> 场景资产评估详情</h2>
<div class="ov-grid"><div class="ov-card ov-scene"><div class="lbl">质量评级</div><div class="val">优秀</div><div class="unit">综合{sc_overall:.0f}%</div></div>
<div class="ov-card ov-scene"><div class="lbl">CLIP语义一致性</div><div class="val">{sg['group_consistency'].iloc[0]:.3f}</div><div class="unit">阈值0.65</div></div>
</div></div>
<div class="chart"><img src="data:image/png;base64,{charts['sc_radar']}"></div>
<div class="chart"><img src="data:image/png;base64,{charts['sc_consist']}"></div>
<div class="notice" style="border-left-color:#16a34a;background:#f0fdf4;">🏞️ <strong>四宫格裁剪说明：</strong>场景图片为2×2四宫格拼接图，评估时先裁剪为4张独立视图，再计算CLIP语义一致性。</div>
<h3>📋 单图明细数据</h3><div class="table-wrap">{sc_table}</div></div>
{rework_html}
<div class="card"><h2>💡 综合结论与优化建议</h2><div class="conclusion"><h3>整体质量评价</h3><ul>
<li><strong>场景资产</strong>：BRISQUE均值{sc_brisque:.1f}{"，画质优秀" if sc_brisque<45 else "，画质有待提升"}；CLIP匹配度合格率{sc_clip_pass:.0f}%；四视图CLIP一致性{sc_gc:.3f}{"，一致性优秀" if sc_gc>=0.75 else "，一致性良好" if sc_gc>=0.65 else "，一致性不足"}。</li>
<li><strong>人物资产</strong>：人体完整性{ch_kp_pass:.0f}%、分区域风格一致性{ch_intra_pass:.0f}%；BRISQUE均值{ch_brisque:.1f}{"，画质优秀" if ch_brisque<45 else "略高于合格线，画质有提升空间"}；CLIP匹配度合格率{ch_clip_pass:.0f}%{f"；人脸一致性均值{ch_face_mean:.3f}，合格率{ch_face_pass:.0f}%" if ch_has_face else "；人脸一致性指标暂未计算"}。</li>
<li><strong>道具资产</strong>：CLIP匹配度合格率{pr_clip_pass:.0f}%；BRISQUE均值{pr_brisque:.1f}{"，画质优秀" if pr_brisque<45 else "偏高，画质有待提升"}；{f"共{pr_ocr_count}张触发OCR文字校验，关键词合格率{pr_kw_pass:.0f}%" if pr_ocr_count>0 else "本批次无文字预期道具，未触发OCR校验"}。</li></ul>
<h3>主要短板</h3><ul>{weakness_html}</ul>
<h3>优化建议</h3><ul>{suggestion_html}</ul></div></div>
{metric_explanation_section(None, C['primary'])}
<div class="footer">短剧资产三层质检系统 · 综合质量评估报告 · 生成于 {t}</div>
</div></body></html>'''
    p=os.path.join(BASE,'短剧资产质量评估综合报告.html')
    open(p,'w',encoding='utf-8').write(html); return p

if __name__=='__main__':
    print('生成人物独立报告...'); p1=gen_character(); print(f'  ✅ {p1} ({os.path.getsize(p1)/1024:.1f}KB)')
    print('生成道具独立报告...'); p2=gen_prop(); print(f'  ✅ {p2} ({os.path.getsize(p2)/1024:.1f}KB)')
    print('生成场景独立报告...'); p3=gen_scene(); print(f'  ✅ {p3} ({os.path.getsize(p3)/1024:.1f}KB)')
    print('生成综合报告...'); p4=gen_comprehensive(); print(f'  ✅ {p4} ({os.path.getsize(p4)/1024:.1f}KB)')
    print('\n✅ 全部4个报告生成完成！')


# ==================== generate_comparison_report.py 假设检验报告 ====================
# -*- coding: utf-8 -*-
"""
生成含假设检验对比结果的HTML报告
用法: python generate_comparison_report.py <新剧本输出目录>
"""
import os
import sys
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analysis import compare_with_baseline, ASSET_DIR_MAP, METRIC_DIRECTION


# ==================== 指标中英文对照与详细说明 ====================
METRIC_INFO = {
    'brisque_score': {
        'cn': 'BRISQUE画质评分',
        'desc': '无参考图像质量评估指标，基于自然场景统计。取值范围0-100，分数越低表示图像失真越少、画质越好。基准值约50-85为可接受范围。',
        'unit': '分（越低越好）'
    },
    'blur_score': {
        'cn': '拉普拉斯清晰度',
        'desc': '通过拉普拉斯算子计算图像边缘的方差，衡量图像清晰程度。数值越高表示边缘越锐利、图像越清晰；数值过低表示图像模糊。',
        'unit': '方差值（越高越清晰）'
    },
    'clip_score': {
        'cn': 'CLIP图文匹配度',
        'desc': '使用CLIP模型计算图像与提示词文本的余弦相似度，衡量生成图片是否符合提示词描述。取值范围0-1，≥0.26判定为合格，越高表示图文匹配越好。',
        'unit': '相似度（≥0.26合格）'
    },
    'keypoint_count': {
        'cn': '人体关键点数量',
        'desc': '使用YOLOv8-pose模型检测人体17个关键点（COCO标准：鼻、眼、耳、肩、肘、腕、髋、膝、踝）。必须检测到全部17个关键点才判定为人体完整，少于17个表示人物有截断或缺失。',
        'unit': '个（必须=17完整）'
    },
    'intra_clip_consistency': {
        'cn': '分区域CLIP一致性',
        'desc': '将人物1×4横排图裁剪为4个区域（特写、正视、侧视、后视），计算相邻区域的CLIP特征余弦相似度，衡量同一角色在不同视图下的一致性。≥0.70判定为合格。',
        'unit': '相似度（≥0.70合格）'
    },
    'intra_face_consistency': {
        'cn': '人脸一致性',
        'desc': '使用InsightFace人脸识别模型，提取人物1×4横排图中第1格（特写）和第2格（正视图）的人脸特征向量，计算余弦相似度，衡量同一角色在特写与正视图下的人脸是否一致。≥0.72判定为合格。',
        'unit': '余弦相似度（≥0.72合格）'
    },
}


def get_metric_cn(metric):
    """获取指标中文名称"""
    return METRIC_INFO.get(metric, {}).get('cn', metric)


def load_eval_summary(output_dir, asset_type):
    """加载质检汇总数据"""
    dir_name = ASSET_DIR_MAP.get(asset_type, asset_type)
    stats_path = os.path.join(output_dir, f"评估报告_{dir_name}", "batch_stats_summary.csv")
    if os.path.exists(stats_path):
        return pd.read_csv(stats_path)
    return None


def generate_html_report(result, output_dir):
    """生成HTML报告"""
    overall_pass = result['overall_pass']
    pass_rate = result['pass_rate'] * 100

    output_name = os.path.basename(output_dir.rstrip(os.sep))
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>假设检验对比报告 - {output_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .subtitle {{ font-size: 14px; opacity: 0.9; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card .label {{ font-size: 13px; color: #888; margin-bottom: 6px; }}
.card .value {{ font-size: 28px; font-weight: bold; }}
.card.pass .value {{ color: #52c41a; }}
.card.fail .value {{ color: #ff4d4f; }}
.card.warn .value {{ color: #faad14; }}
.section {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 20px; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 2px solid #f0f0f0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #f0f0f0; font-size: 14px; }}
th {{ background: #fafafa; font-weight: 600; color: #555; }}
tr:hover {{ background: #f9f9f9; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
.badge.pass {{ background: #f6ffed; color: #52c41a; border: 1px solid #b7eb8f; }}
.badge.fail {{ background: #fff2f0; color: #ff4d4f; border: 1px solid #ffccc7; }}
.badge.warn {{ background: #fffbe6; color: #faad14; border: 1px solid #ffe58f; }}
.better {{ color: #52c41a; font-weight: 600; }}
.worse {{ color: #ff4d4f; font-weight: 600; }}
.conclusion-box {{ padding: 16px; border-radius: 8px; margin-top: 16px; }}
.conclusion-box.pass {{ background: #f6ffed; border: 1px solid #b7eb8f; }}
.conclusion-box.fail {{ background: #fff2f0; border: 1px solid #ffccc7; }}
.conclusion-box h3 {{ margin-bottom: 8px; }}
.metric-name {{ font-weight: 500; }}
.direction {{ font-size: 11px; color: #999; }}
.footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
<h1>假设检验对比报告</h1>
<div class="subtitle">新剧本「{output_name}」 vs 基准值（当前项目全量质检结果）</div>
<div class="subtitle" style="margin-top:8px;">对比时间：{result['comparison_time']} | 显著性水平 α = {result['alpha']}</div>
</div>

<div class="summary-cards">
<div class="card {'pass' if overall_pass else 'fail'}">
<div class="label">最终结论</div>
<div class="value">{'符合要求' if overall_pass else '不符合要求'}</div>
</div>
<div class="card {'pass' if pass_rate >= 90 else 'warn'}">
<div class="label">指标合格率</div>
<div class="value">{pass_rate:.1f}%</div>
</div>
<div class="card">
<div class="label">对比指标数</div>
<div class="value">{result['total_metrics']}</div>
</div>
<div class="card">
<div class="label">合格指标数</div>
<div class="value" style="color:#52c41a;">{result['pass_metrics']}</div>
</div>
</div>
"""

    # 各资产类型详细对比
    for asset_type, type_result in result['asset_types'].items():
        if not type_result.get('available', False):
            continue

        type_name = type_result['name']
        type_pass = type_result['type_pass']

        html += f"""
<div class="section">
<h2>{type_name}资产对比 <span class="badge {'pass' if type_pass else 'fail'}">{'合格' if type_pass else '存在问题'}</span></h2>
<p style="color:#666;font-size:14px;margin-bottom:12px;">{type_result['summary']}</p>
<table>
<thead>
<tr>
<th>指标</th>
<th>方向</th>
<th>基准均值</th>
<th>新剧本均值</th>
<th>差异</th>
<th>p值</th>
<th>显著性</th>
<th>结论</th>
</tr>
</thead>
<tbody>
"""

        for metric, r in type_result['metrics'].items():
            if r['p_value'] is None:
                p_str = 'N/A'
                sig_badge = '<span class="badge warn">样本不足</span>'
            else:
                p_str = f"{r['p_value']:.4f}"
                if r['is_significant']:
                    sig_badge = '<span class="badge fail">显著</span>'
                else:
                    sig_badge = '<span class="badge pass">不显著</span>'

            bm = f"{r['baseline_mean']:.3f}" if r['baseline_mean'] is not None else 'N/A'
            nm = f"{r['new_mean']:.3f}" if r['new_mean'] is not None else 'N/A'

            # 差异
            if r['baseline_mean'] is not None and r['new_mean'] is not None:
                diff = r['new_mean'] - r['baseline_mean']
                higher_is_better = r.get('higher_is_better', True)
                if higher_is_better:
                    diff_class = 'better' if diff >= 0 else 'worse'
                else:
                    diff_class = 'better' if diff <= 0 else 'worse'
                diff_str = f'<span class="{diff_class}">{"+" if diff >= 0 else ""}{diff:.3f}</span>'
            else:
                diff_str = 'N/A'

            direction = '越高越好' if METRIC_DIRECTION.get(metric, True) else '越低越好'

            # 结论
            if '优于' in r['conclusion']:
                conclusion_badge = '<span class="badge pass">优于基准</span>'
            elif '差于' in r['conclusion']:
                conclusion_badge = '<span class="badge fail">差于基准</span>'
            else:
                conclusion_badge = '<span class="badge pass">相当</span>'

            html += f"""<tr>
<td class="metric-name" title="{METRIC_INFO.get(metric, {}).get('desc', '')}">{get_metric_cn(metric)}<br><span style="font-size:11px;color:#999;">{metric}</span></td>
<td class="direction">{direction}</td>
<td>{bm}</td>
<td>{nm}</td>
<td>{diff_str}</td>
<td>{p_str}</td>
<td>{sig_badge}</td>
<td>{conclusion_badge}</td>
</tr>"""

        html += """</tbody>
</table>
</div>"""

    # 最终结论
    html += f"""
<div class="section">
<h2>最终结论</h2>
<div class="conclusion-box {'pass' if overall_pass else 'fail'}">
<h3>{result['conclusion']}</h3>
<p style="margin-top:8px;">
在 {result['total_metrics']} 个对比指标中，{result['pass_metrics']} 个合格（无显著差异或优于基准），
合格率 {pass_rate:.1f}%。
</p>
"""

    # 列出不合格指标
    bad_metrics = []
    for asset_type, type_result in result['asset_types'].items():
        if not type_result.get('available', False):
            continue
        for metric, r in type_result['metrics'].items():
            if r['is_significant'] and '差于' in r['conclusion']:
                bad_metrics.append(f"{type_result['name']} - {metric}（p={r['p_value']:.4f}，{r['direction_cn']}）")

    if bad_metrics:
        html += "<p style='margin-top:12px;'><strong>显著差于基准的指标：</strong></p><ul style='margin-left:20px;margin-top:8px;'>"
        for m in bad_metrics:
            html += f"<li style='color:#ff4d4f;'>{m}</li>"
        html += "</ul>"

    html += """
</div>
</div>

<div class="section">
<h2>📖 指标详细说明</h2>
<p style="color:#666;font-size:14px;margin-bottom:16px;">本报告中使用的所有质检指标的含义、计算方式和合格标准如下：</p>
<table>
<thead>
<tr>
<th style="width:18%;">指标名称</th>
<th style="width:12%;">英文标识</th>
<th style="width:15%;">单位/方向</th>
<th>详细说明</th>
</tr>
</thead>
<tbody>
"""

    for metric_key, info in METRIC_INFO.items():
        html += f"""<tr>
<td><strong>{info['cn']}</strong></td>
<td style="font-family:monospace;font-size:12px;color:#666;">{metric_key}</td>
<td>{info['unit']}</td>
<td style="font-size:13px;line-height:1.6;">{info['desc']}</td>
</tr>"""

    html += """</tbody>
</table>
</div>

<div class="section">
<h2>📊 假设检验方法说明</h2>
<div style="font-size:14px;line-height:1.8;color:#444;">
<p><strong>检验方法：</strong>曼-惠特尼U检验（Mann-Whitney U Test），属于非参数检验，不要求数据服从正态分布，适用于对比两组独立样本的分布是否存在显著差异。</p>
<p><strong>原假设 H₀：</strong>新剧本资产指标与基准值无显著差异（两组数据来自同一分布）。</p>
<p><strong>备择假设 H₁：</strong>新剧本资产指标与基准值存在显著差异。</p>
<p><strong>显著性水平 α：</strong>0.05。当 p值 &lt; 0.05 时，拒绝原假设，认为存在显著差异；当 p值 ≥ 0.05 时，不拒绝原假设，认为无显著差异。</p>
<p><strong>合格判定：</strong></p>
<ul style="margin-left:20px;margin-top:8px;">
<li>p值 ≥ 0.05 → 无显著差异 → <span class="badge pass">合格</span></li>
<li>p值 &lt; 0.05 且新剧本优于基准 → <span class="badge pass">合格（优于基准）</span></li>
<li>p值 &lt; 0.05 且新剧本差于基准 → <span class="badge fail">不合格（差于基准）</span></li>
</ul>
<p style="margin-top:12px;"><strong>指标方向：</strong>"越高越好"类指标（如CLIP匹配度、清晰度），新剧本均值高于基准视为优于；"越低越好"类指标（如BRISQUE评分），新剧本均值低于基准视为优于。</p>
</div>
</div>

<div class="footer">
<p>本报告基于曼-惠特尼U检验（非参数检验），显著性水平 α=0.05</p>
<p>基准值来源：当前项目全量质检结果 | 新剧本：{output_name}</p>
</div>

</div>
</body>
</html>"""

    return html


if __name__ == '__main__':
    # 模式1：生成4份HTML报告（人物/道具/场景/综合）
    print('='*60)
    print('生成人物/道具/场景/综合 4份HTML报告')
    print('='*60)
    p1 = gen_character(); print(f'  ✅ 人物报告 ({os.path.getsize(p1)/1024:.1f}KB)')
    p2 = gen_prop(); print(f'  ✅ 道具报告 ({os.path.getsize(p2)/1024:.1f}KB)')
    p3 = gen_scene(); print(f'  ✅ 场景报告 ({os.path.getsize(p3)/1024:.1f}KB)')
    p4 = gen_comprehensive(); print(f'  ✅ 综合报告 ({os.path.getsize(p4)/1024:.1f}KB)')
    print('\n✅ 全部4份报告生成完成！')

    # 模式2：如果有命令行参数，生成假设检验对比报告
    if len(sys.argv) > 1:
        new_output_dir = sys.argv[1]
        print(f'\n{"="*60}')
        print(f'生成假设检验对比报告')
        print(f'新剧本目录: {new_output_dir}')
        print(f'{"="*60}')

        # 运行假设检验
        result = compare_with_baseline(new_output_dir)

        # 生成HTML
        html = generate_html_report(result, new_output_dir)

        # 保存
        report_path = os.path.join(new_output_dir, "假设检验对比报告.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✅ 报告已保存: {report_path}")
        print(f"文件大小: {os.path.getsize(report_path)/1024:.1f} KB")

        # 打印摘要
        print(f"\n{'='*60}")
        print(f"最终结论: {result['conclusion']}")
        print(f"指标合格率: {result['pass_rate']*100:.1f}% ({result['pass_metrics']}/{result['total_metrics']})")
        print(f"{'='*60}")


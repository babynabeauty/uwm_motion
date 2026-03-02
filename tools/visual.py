import re
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_log(file_path):
    data = {'eval_step': [], 'success_rate': []}
    
    # 匹配成功率：兼容以下两种格式
    # 1) Step: 2000 success rate: 0.08
    # 2) Step: 2000 | Avg Success Rate: 0.1600
    eval_pattern = re.compile(
        r"Step:\s*(\d+)\s*(?:\|\s*)?(?:Avg\s*)?Success[ -]?Rate:\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    )
    
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return data

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().replace('\r', '\n')
        lines = content.split('\n')
        
        for line in lines:
            eval_match = eval_pattern.search(line)
            if eval_match:
                data['eval_step'].append(int(eval_match.group(1)))
                data['success_rate'].append(float(eval_match.group(2)))
                
    return data

def plot_success_rates(log_files):
    # 存储解析后的数据
    all_data = {}
    for label, path in log_files.items():
        d = parse_log(path)
        print(f"[{label}] 提取到: {len(d['eval_step'])} 条成功率数据")
        if d['eval_step']:
            all_data[label] = d

    if not all_data:
        print("错误: 所有文件中均未发现成功率数据。")
        return

    # --- 1. 绘制混合对比图 ---
    plt.figure(figsize=(12, 7))
    for label, d in all_data.items():
        plt.plot(d['eval_step'], d['success_rate'], label=label, marker='o', markersize=4, linewidth=1.5)
    
    plt.title('All Methods Success Rate Comparison', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    combined_name = 'combined_success_rate.png'
    plt.savefig(combined_name, dpi=300)
    print(f"\n[完成] 混合对比图已保存为: {combined_name}")
    plt.close() # 释放内存

    # --- 2. 绘制并保存每个文件的独立图 ---
    for label, d in all_data.items():
        plt.figure(figsize=(10, 5))
        plt.plot(d['eval_step'], d['success_rate'], color='steelblue', marker='s', markersize=5, label=label)
        
        plt.title(f'Success Rate: {label}', fontsize=12)
        plt.xlabel('Steps')
        plt.ylabel('Success Rate')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        
        # 将文件名中的空格/特殊字符替换为下划线
        safe_label = label.replace(" ", "_").replace("/", "_")
        individual_name = f'success_rate_{safe_label}.png'
        plt.savefig(individual_name, dpi=200)
        print(f"   --> 独立图已保存: {individual_name}")
        plt.close()

if __name__ == "__main__":
    logs_to_plot = {
        "Baseline":"/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_bowl_drawer/libero_bowl_drawer_baseline_V2.log",
        "RAFT_MASK_no_mixture":"/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_bowl_drawer/libero_bowl_drawer_RAFT_MASK_no_mixture_V2.log",
        # "RAFT_no_MASK_no_mixture":"/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_bowl_drawer/libero_bowl_drawer_RAFT_no_MASK_no_mixture_V2.log",
        # "RAFT_no_MASK_5_mixture":"/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_bowl_drawer/libero_bowl_drawer_RAFT_no_MASK_5_mixture_V2.log",
        "RAFT_MASK_5_mixture":"/data/shared_workspace/zhangshiqi/uwm_motion_rst_saving/libero_bowl_drawer/libero_bowl_drawer_RAFT_MASK_5_mixture_V2.log"

    }
    
    plot_success_rates(logs_to_plot)
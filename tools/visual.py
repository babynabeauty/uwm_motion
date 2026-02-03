import re
import matplotlib.pyplot as plt
import os
import numpy as np

def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def parse_log(file_path):
    data = {
        'train_step': [], 'loss': [], 'action_loss': [], 'motion_loss': [],
        'eval_step': [], 'success_rate': []
    }
    
    # 匹配训练 Loss (允许匹配行内任何位置)
    train_pattern = re.compile(r"step:\s+(\d+),\s+loss:\s+([\d.]+),action_loss:\s+([\d.]+),motion_loss:\s+([\d.]+)")
    # 匹配成功率 (增加忽略大小写和对横杠/空格的兼容)
    eval_pattern = re.compile(r"Step:\s+(\d+)\s+success[ -]rate:\s+([\d.]+)", re.IGNORECASE)
    
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        return data

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # 重要：处理 tqdm 产生的 \r 字符，将其统一视为换行
        content = f.read().replace('\r', '\n')
        lines = content.split('\n')
        
        for line in lines:
            # 1. 尝试提取训练数据
            train_match = train_pattern.search(line)
            if train_match:
                data['train_step'].append(int(train_match.group(1)))
                data['loss'].append(float(train_match.group(2)))
                data['action_loss'].append(float(train_match.group(3)))
                data['motion_loss'].append(float(train_match.group(4)))
            
            # 2. 尝试提取评测数据 (不再使用 continue，防止同一行有两个匹配)
            eval_match = eval_pattern.search(line)
            if eval_match:
                data['eval_step'].append(int(eval_match.group(1)))
                data['success_rate'].append(float(eval_match.group(2)))
                
    return data

def plot_all_metrics(log_files, smooth_window=100):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ((ax1, ax2), (ax3, ax4)) = axes

    for label, path in log_files.items():
        d = parse_log(path)
        
        # 打印调试信息，确认是否提取到数据
        print(f"[{label}] 提取到: {len(d['train_step'])} 条训练 Loss, {len(d['eval_step'])} 条成功率数据")
        
        if not d['train_step'] and not d['eval_step']:
            print(f"   --> 错误: 文件 {path} 中未发现匹配数据")
            continue
        
        # 1. Success Rate
        if d['eval_step']:
            ax1.plot(d['eval_step'], d['success_rate'], label=label, marker='o', markersize=4)
        
        # 2. Total Loss
        if d['loss']:
            steps = d['train_step'][:len(moving_average(d['loss'], smooth_window))]
            ax2.plot(steps, moving_average(d['loss'], smooth_window), label=label)
        
        # 3. Action Loss
        if d['action_loss']:
            steps = d['train_step'][:len(moving_average(d['action_loss'], smooth_window))]
            ax3.plot(steps, moving_average(d['action_loss'], smooth_window), label=label)
            
        # 4. Motion Loss
        if d['motion_loss']:
            steps = d['train_step'][:len(moving_average(d['motion_loss'], smooth_window))]
            ax4.plot(steps, moving_average(d['motion_loss'], smooth_window), label=label)

    # 装饰
    ax1.set_title('Success Rate', fontsize=12); ax1.grid(True); ax1.set_ylim(-0.05, 1.05)
    ax2.set_title('Total Loss', fontsize=12); ax2.grid(True)
    ax3.set_title('Action Loss', fontsize=12); ax3.grid(True)
    ax4.set_title('Motion Loss', fontsize=12); ax4.grid(True)
    
    for ax in axes.flat:
        ax.set_xlabel('Steps')
        ax.legend()

    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300)
    print("\n[完成] 图像已保存为 training_analysis.png")
    plt.show()

if __name__ == "__main__":
    logs_to_plot = {
        "Baseline": "/data/workspace/zhangshiqi/uwm_motion/NO_MV_libero_bowl_drawer.log",
        "Motion_WithMask": "/data/workspace/zhangshiqi/uwm_motion/MV_MASK_libero_bowl_drawer.log"
    }
    
    plot_all_metrics(logs_to_plot, smooth_window=100)
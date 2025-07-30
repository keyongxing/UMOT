# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
"""
这段代码实现了一个简单的文件同步检查工具，主要用于：
1. 递归复制指定源目录下的所有Python文件
2. 检查源文件与目标文件的差异
3. 输出存在差异的文件对比信息
"""

import argparse
from glob import glob
from subprocess import run

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加源目录和目标目录参数
parser.add_argument('src')  # 必须指定的源目录参数
parser.add_argument('dst')  # 必须指定的目标目录参数
args = parser.parse_args()  # 解析命令行参数

# 遍历所有Python源文件
for src in (glob(args.src + '/*/*.py') + glob(args.src + '*.py')):
    dst = src.replace(args.src, args.dst)  # 生成目标文件路径

    # 执行diff命令比较源文件和目标文件
    diff_process = run(['diff', src, dst], capture_output=True, text=True)

    # 如果diff返回非零状态码（存在差异）
    if diff_process.returncode != 0:
        print(f'代码差异检测 - {src} vs {dst}')
        print(diff_process.stdout)  # 输出差异内容
#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['dHIT', 'Our(128 bp, 1 GPU)', 'Our(128 bp, 4 GPU)', 'Our (16 bp, 4GPU)']
genome_length_per_hour = [12.5, 1256, 4654, 464]
colors = ['#C3E2EC', '#E3C6E0', '#CECCE5', '#FCDFBE']  # Color scheme inferred from the images

# 对数缩放数据
log_genome_length_per_hour = np.log10(genome_length_per_hour)

# 绘制柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(methods, log_genome_length_per_hour, color=colors, width=0.7)

plt.ylim(0, 5)

# Removing y-axis ticks
# plt.gca().yaxis.set_ticks([])
plt.gca().xaxis.set_ticks([])
# 添加标签和标题
# plt.xlabel('Method')
# plt.ylabel('Log10(Genome Length Processed per Hour)')
# plt.title('Comparison of Genome Processing Speed by Method and GPU (Log Scale)')

# 添加数值标签
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(10**yval, 2), ha='center', va='bottom')

# 显示图形
plt.show()

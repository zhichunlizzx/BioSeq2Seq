import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ['AttentiveChrome', 'DeepChrome', 'EPCOT', 'TransformerChrome', 'BioSeq2Seq+SVM']
colors = ['#EAEAEA', '#91bFFA', '#FF7F00', '#F5DC75', '#0075ba']  # Color scheme inferred from the images

cellline = ['K562', 'GM12878', 'HepG2', 'K562', 'GM12878', 'HepG2']
index = np.asarray(range(len(cellline)))

# Calculating the required size in inches for the image
width_mm = 90 # width in mm
height_mm = 50 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

bar_width=0.15
linewidth=0.5

AttentiveChrome = [0.92, 0.9024, 0.85, 0.84, 0.8307, 0.58]
DeepChrome = [0.92, 0.91, 0.85, 0, 0, 0]
EPCOt = [0.92, 0,  0.84, 0.83, 0, 0.58]
TransformerChrome = [0.92, 0.91, 0.85, 0.86, 0.8415, 0.5878]
BioSeq2Seq = [0.9342, 0.9398, 0.9405, 0.9364, 0.9447, 0.9219]

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
# 绘制柱状图
bars1 = plt.bar(index-2*bar_width, AttentiveChrome, color=colors[0], width=bar_width, edgecolor='black', linewidth=linewidth)
bars2 = plt.bar(index-1*bar_width, DeepChrome, color=colors[1], width=bar_width, edgecolor='black', linewidth=linewidth)
bars3 = plt.bar(index, EPCOt, color=colors[2], width=bar_width, edgecolor='black', linewidth=linewidth)
bars4 = plt.bar(index+1*bar_width, TransformerChrome, color=colors[3], width=bar_width, edgecolor='black', linewidth=linewidth)
bars5 = plt.bar(index+2*bar_width, BioSeq2Seq, color=colors[4], width=bar_width, edgecolor='black', linewidth=linewidth)

plt.ylim(0.5, 1)
# 移除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置x轴和y轴边框的宽度
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)

# 增加x轴和y轴刻度标签的字体大小
ax.tick_params(axis='x')
ax.tick_params(axis='y')

# 修改 x, y 轴刻度字体和大小
plt.xticks(index, cellline)

# 添加图例
plt.legend((bars1[0], bars2[0], bars3[0], bars4[0], bars5[0]), (methods[0], methods[1], methods[2], methods[3], methods[4]),
           ncol=5,
           bbox_to_anchor = (1.00, 1.1), frameon=False)

plt.show()

import numpy as np
import xlrd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
width_mm = 250 / 1.25  # width in mm
height_mm = 60 / 1.25  # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch


# 读取数据
data_path = r'BioSeq2Seq\plot\TFBS\plot_data\TF_results.xlsx'
workbook = xlrd.open_workbook(data_path)
table = workbook.sheet_by_name('3stage_best')

# num of tf
num = 90
# 数据所在的行-1
row_num = 133
# 数据所在的列-1
column_num = 1
# column of tf name
tf_column = 0

tf_prc = {}


for i in range(num):
    tf_prc_list = []
    for j in range(column_num, column_num+6):
        print(table.row(i+row_num)[j].value)
        if table.row(i+row_num)[j].value > 0:
            tf_prc_list.append(table.row(i+row_num)[j].value)
            # values.append(table.row(i+row_num)[j].value)
    if len(tf_prc_list) > 1:
        tf_prc_list.sort(reverse=True)
    
    tf_prc[table.row(i+row_num)[tf_column].value] = tf_prc_list


tf_prc = dict(sorted(tf_prc.items(), key=lambda item: statistics.median(item[1]), reverse=True))
# tf_prc = dict(sorted(tf_prc.items(), key=lambda item: max(item[1]), reverse=True))

tf_list = []
prc_list = []
pre_all = []
for tf in tf_prc:
    tf_list.append(tf)
    prc_list.append(tf_prc[tf])
    pre_all.extend(tf_prc[tf])

print('Median: ', np.median(pre_all))

for tf in tf_list:
    print(tf)

# 定义自定义渐变色
colors = ['#c2252f', '#e86323', '#f78d24', '#f1c17a', '#b8e1ee', '#75bdde', '#1e8bb8', '#2d3194']



# 创建自定义的颜色映射
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=1000)

# 定义采样的数量
num_samples = len(tf_list)

# 采样渐变色映射，生成颜色代码
sampled_colors = [custom_cmap(i / (num_samples - 1)) for i in range(num_samples)]

# 将RGBA颜色格式转换为十六进制颜色代码
hex_colors = ['#%02x%02x%02x' % (int(255*r), int(255*g), int(255*b)) for r, g, b, a in sampled_colors]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

# 绘制boxplot
bplot = axes.boxplot(prc_list, patch_artist=True, zorder=0)

# 设置每个box的颜色
for patch, color in zip(bplot['boxes'], hex_colors):
    patch.set_facecolor(color)
    patch.set_linewidth(0.5)  # 设置边框宽度

# 将箱体的边框、须状线、中位数线等设置为黑色
for element in ['whiskers', 'medians', 'caps']:
    plt.setp(bplot[element], color='black')
    plt.setp(bplot[element], color='black', linewidth=0.5)  # 设置线宽

# 设置x轴刻度
axes.set_xticks(np.arange(1, len(tf_list) + 1))  # 设置x轴刻度位置
axes.set_xticklabels(tf_list, rotation=90)  # 设置x轴刻度标签为tf_list，并旋转90度

# 移除上方和右方的边框
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.show()

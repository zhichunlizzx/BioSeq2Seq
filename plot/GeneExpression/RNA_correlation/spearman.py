import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
width_mm = 80 # width in mm
height_mm = 56 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

linewidth = 1

# 数据：假设每个细胞系有多个数据
celllines = ['K562', 'GM12878', 'IMR90', 'MCF-7', 'HepG2']
pearson = [0.8783, 0.748, 0.7256, 0.7524, 0.7846]
# 假设每个细胞系有三个不同的数据点（这只是示例数据）
data_points = {
    'K562': [0.8783, 0.8573, 0.8913, 0.8873, 0.8779, 0.8295, 0.8609, 0.6614, 0.7618, 0.7564],
    'GM12878': [0.748, 0.7035, 0.6498, 0.7163, 0.5392, 0.7, 0.4811, 0.7919],
    'IMR90': [0.7256, 0.7555, 0.4795, 0.6399],
    'MCF-7': [0.7524, 0.7428, 0.679, 0.5419],
    'HepG2': [0.7846, 0.7554, 0.736, 0.5703, 0.5441, 0.7747]
}


# 计算标准差
std_devs = [np.std(values, ddof=1) for group, values in data_points.items()]
error_params=dict(elinewidth=linewidth,ecolor='black',capsize=5)#设置误差标记参数

# 方差分析 (ANOVA)
anova_data = [data_points[cellline] for cellline in celllines]
f_stat, p_value = stats.f_oneway(*anova_data)

# 输出方差分析结果
print(f'ANOVA F-statistic: {f_stat:.2f}')
print(f'ANOVA p-value: {p_value:.4f}')

colors = ['#374f99', '#6fa3c2', '#c6e5ea', '#f0b169', '#df5b3e']  # Color scheme inferred from the images
# 绘制柱状图
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
bars = plt.bar(celllines,
               pearson,
               color=colors,
               width=0.7,
               linewidth=linewidth,
               edgecolor='black',
               yerr=std_devs,
               error_kw=error_params
               )

# 绘制散点图
for i, cellline in enumerate(celllines):
    x = np.full_like(data_points[cellline], i, dtype=float)  # x位置为每个细胞系的索引
    plt.scatter(x, data_points[cellline], color='black', zorder=5, s=5)  # 使用红色散点

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2),
             ha='center', va='bottom')

# 设置y轴范围
plt.ylim(0, 1)

# 移除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置x轴和y轴边框的宽度
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)

# 修改y轴刻度字体大小
ax.tick_params(axis='y')
ax.tick_params(axis='x')

plt.ylabel("Spearman Correlation")

# 显示方差分析的结果
plt.figtext(0.15, 0.85, f"ANOVA p-value: {p_value:.4f}")

# 显示图形
plt.show()
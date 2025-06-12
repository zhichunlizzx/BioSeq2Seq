import matplotlib.pyplot as plt
import numpy as np

# 数据
tre = ['Promoter', 'Insulator', 'PolyA', 'GeneBody']
colors = ['#f5a579', '#408bc8', '#9bc6bc', '#dcbe7a']  # Color scheme inferred from the images

eva_list = ['Accuracy', 'Precision', 'Recall', 'F-score']
index = np.asarray(range(len(eva_list)))

bar_width=0.2
linewidth=2
fontsize = 16

Promoter = [0.92, 0.88, 0.96, 0.92]
Insulator = [0.77, 0.89, 0.72, 0.79]
PolyA = [0.69, 0.82, 0.65, 0.73]
GeneBody = [0.93, 0.93, 0.93, 0.93]


# 绘制柱状图
bars1 = plt.bar(index-1.5*bar_width, Promoter, color=colors[0], width=bar_width, edgecolor='black', linewidth=linewidth)
bars2 = plt.bar(index-0.5*bar_width, Insulator, color=colors[1], width=bar_width, edgecolor='black', linewidth=linewidth)
bars3 = plt.bar(index+0.5*bar_width, PolyA, color=colors[2], width=bar_width, edgecolor='black', linewidth=linewidth)
bars4 = plt.bar(index+1.5*bar_width, GeneBody, color=colors[3], width=bar_width, edgecolor='black', linewidth=linewidth)

plt.ylim(0.5, 1)
# 移除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置x轴和y轴边框的宽度
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)

# 增加x轴和y轴刻度标签的字体大小
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)

# 修改 x, y 轴刻度字体和大小
plt.xticks(index, eva_list, fontproperties = 'Arial', size = 24)

# 添加图例
# plt.legend((bars1[0], bars2[0], bars3[0], bars4[0]), ('Promoter', 'Insulator', 'PolyA', 'GeneBody'),
#            prop={'family': 'Calibri', 'size': 24},  ncol=5,
#            bbox_to_anchor = (0.80, 1.1), frameon=False)

plt.show()

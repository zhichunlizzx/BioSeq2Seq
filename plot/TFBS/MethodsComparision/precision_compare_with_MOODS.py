import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
width_mm = 40 # width in mm
height_mm = 40 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

# 读取数据
data_path = r'BioSeq2Seq\plot\TFBS\plot_data\TF_results.xlsx'
workbook = xlrd.open_workbook(data_path)
table = workbook.sheet_by_name('3stage_model')

# num of tf
num = 57
# 数据所在的行-1
row_num = 3
# 数据所在的列-1
column_num = 21
# column of tf name
tf_column = 20

moods = []
bioseq2seq_list = []
tf_list = []

# 读取数据并存储
for i in range(num):
    moods.append(table.row(i+row_num)[column_num+2].value)
    bioseq2seq_list.append(table.row(i+row_num)[column_num].value)
    tf_list.append(table.row(i+row_num)[tf_column].value)



moods = np.asarray(moods, dtype='float')
bioseq2seq_list = np.asarray(bioseq2seq_list, dtype='float')
tf_list = np.asarray(tf_list)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

p1 = plt.scatter(moods, bioseq2seq_list, s=50, alpha=0.5, marker='o', color='#5595d1', linewidths=0)
p2 = plt.scatter(moods, [0] * len(moods), s=50, alpha=0.5, marker='|', color='#5595d1')
p3 = plt.scatter([0] * len(moods), bioseq2seq_list, s=50, alpha=0.5, marker='_', color='#5595d1')

# 移除上方和右方的边框
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置 x 轴和 y 轴宽度
ax.spines['left'].set_linewidth(2)   # y 轴线宽
ax.spines['bottom'].set_linewidth(2)  # x 轴线宽

plt.plot((0, 1.1), (0, 1.1), linewidth=1, color='#5595d1')
plt.ylim(0, 1.1)
plt.xlim(0, 1.1)

plt.ylabel("BioSeq2Seq Precision at 5% Recall")
plt.xlabel("TF Motif Precision at 5% Recall")

plt.text(x=0.58, y=0.05, s='mean=%.4f' % round(np.mean(moods), 4),
         fontdict=dict(color='black')
        )

plt.text(x=0.02, y=1.08, s='mean=%.4f' % round(np.mean(bioseq2seq_list), 4),
         fontdict=dict(color='black')
        )

plt.show()

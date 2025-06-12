import numpy as np
import matplotlib.pyplot as plt
import xlrd

def excel_data(data_path, row_id, col_id, cell_line):
    lines = [0, 1, 2, 3, 4, 5, 6, 7]
    workbook = xlrd.open_workbook(data_path)
    table = workbook.sheet_by_name(cell_line)
    results = []
    for i in lines:
        results.append(table.row(row_id+i)[col_id].value)
    return results

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
# width_mm = 94.64 # width in mm
width_mm = 124.64 # width in mm
height_mm = 60.118 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

histone_modi = ('H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K9ac', 'H3K9me3')
index = np.asarray(range(len(histone_modi)))
linewidth = 0.55

bar_width = 0.2

# excel path

data_path = r'BioSeq2Seq\plot\HistoneModification\plot_data\evaluation.xlsx'

lines = 117
cell_line = 'MSE'

col_id = 1
rd = excel_data(data_path, lines, 1, cell_line)
print(rd)
# dhit = excel_data(data_path, lines, 2, cell_line)
r = excel_data(data_path, lines, 3, cell_line)
d = excel_data(data_path, lines, 4, cell_line)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

# p1 = plt.bar(index-1.5*bar_width, dhit, width=bar_width, color='#a1a0a5', linewidth=linewidth, edgecolor='black')
p2 = plt.bar(index-0.5*bar_width, r, width=bar_width, color='#d65316', linewidth=linewidth, edgecolor='black')
p3 = plt.bar(index+0.5*bar_width, d, width=bar_width, color='#ecb21d', linewidth=linewidth, edgecolor='black')
p4 = plt.bar(index+1.5*bar_width, rd, width=bar_width, color='#0075ba', linewidth=linewidth, edgecolor='black')

# plt.yticks(np.arange(0, 3, 0.1), fontproperties = 'Calibri', size = 24)
plt.ylabel("MSE (Global)")

# 修改 x, y 轴刻度字体和大小
plt.xticks(index, histone_modi, rotation=45, ha='right')  # 这里将字体倾斜 45 度并右对齐

# 移除上方和右方的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置x轴和y轴边框的宽度
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)

plt.show()
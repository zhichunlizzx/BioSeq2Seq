import numpy as np
import matplotlib.pyplot as plt
import xlrd

def excel_data(data_path, row_id, col_id, cell_line):
    lines = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    workbook = xlrd.open_workbook(data_path)
    table = workbook.sheet_by_name(cell_line)
    results = []
    for i in lines:
        results.append(table.row(row_id+i)[col_id].value)
    return results


# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
width_mm = 150 # width in mm
height_mm = 78.334 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

histone_modi = ('H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K9ac', 'H3K9me3', 'H4K20me1')


data_path = r'BioSeq2Seq\plot\HistoneModification\plot_data\correlation.xlsx'
cell_line = 'MCF7'

index = np.asarray(range(len(histone_modi)))
bar_width = 0.2
linewidth =0.7

model_128_seq = excel_data(data_path, 28, 5, cell_line)
model_128_ro = excel_data(data_path, 2, 5, cell_line)
model_128_rd = excel_data(data_path, 15, 5, cell_line)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

p1 = plt.bar(index-bar_width, model_128_seq, width=bar_width, color='#ecb21d', linewidth=linewidth, edgecolor='black')
p2 = plt.bar(index, model_128_ro, width=bar_width, color='#d65316', linewidth=linewidth, edgecolor='black')
p3 = plt.bar(index+bar_width, model_128_rd, width=bar_width, color='#0075ba', linewidth=linewidth, edgecolor='black')

# 移除上方和右方的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置x轴和y轴边框的宽度
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)

plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Spearman correlation (MCF-7)")
plt.xticks(index, histone_modi, rotation=45, ha='right')

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import xlrd

def excel_data(data_path, row_id, col_id, cell_line, lines=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    workbook = xlrd.open_workbook(data_path)
    table = workbook.sheet_by_name(cell_line)
    results = []
    for i in lines:
        results.append(table.row(row_id+i)[col_id].value)
    return results


def read_tre_values(row_ids, col_id, chart_name):
    dhit2_promoter = np.asarray([])
    dhit_promoter = np.asarray([])

    for i in range(len(row_ids)):
        if i == 0:
            lines = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif i == 4:
            lines = [0, 2, 4, 5, 6]
        elif i == 8:
            lines = [0, 1, 2, 3, 4, 5, 6]
        else:
            lines = [0, 1, 2, 3, 4, 5, 6, 8]

        dhit2_promoter = np.append(dhit2_promoter, excel_data(data_path, row_ids[i], col_id, chart_name, lines=lines))
        dhit_promoter = np.append(dhit_promoter, excel_data(data_path, row_ids[i], col_id+12, chart_name, lines=lines))

    return dhit2_promoter, dhit_promoter


# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
width_mm = 70.75 # width in mm
height_mm = 66.75 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

border_linewidth=0.75

histone_modi = ('H3K122ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K9ac', 'H3K9me3',
                'H4K20me1')
linewidth=0
size=20

mean_dHIT2 = 0
mean_dHIT = 0

# excel path
data_path = r'BioSeq2Seq\plot\HistoneModification\plot_data\evaluation.xlsx'
chart_name = 'TRE'

row_ids = [162, 178, 193, 208, 223, 237, 251, 265, 279]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
# promoter
col_id = 2
dhit2_promoter, dhit_promoter = read_tre_values(row_ids, col_id, chart_name)
plt_promoter = plt.scatter(dhit_promoter, dhit2_promoter, linewidths=1, alpha=0.8, marker='o', color='#E63B1F', s=size, linewidth=linewidth)

mean_dHIT2 += np.mean(dhit2_promoter)
mean_dHIT += np.mean(dhit_promoter)

# enhancer
col_id = 4
dhit2_enhancer, dhit_enhancer = read_tre_values(row_ids, col_id, chart_name)
plt_enhancer = plt.scatter(dhit_enhancer, dhit2_enhancer, linewidths=1, alpha=0.8, marker='o', color='#FF8C00', s=size, linewidth=linewidth)

mean_dHIT2 += np.mean(dhit2_promoter)
mean_dHIT += np.mean(dhit_promoter)

# genebody
col_id = 6
dhit2_genebody, dhit_genebody = read_tre_values(row_ids, col_id, chart_name)
plt_genebody = plt.scatter(dhit_genebody, dhit2_genebody, linewidths=1, alpha=0.8, marker='o', color='#FFD200', s=size, linewidth=linewidth)

mean_dHIT2 += np.mean(dhit2_promoter)
mean_dHIT += np.mean(dhit_promoter)

# insulator
col_id = 8
dhit2_insulator, dhit_insulator = read_tre_values(row_ids[:-1], col_id, chart_name)
plt_insulator = plt.scatter(dhit_insulator, dhit2_insulator, linewidths=1, alpha=0.8, marker='o', color='#0072B2', s=size, linewidth=linewidth)

mean_dHIT2 += np.mean(dhit2_promoter)
mean_dHIT += np.mean(dhit_promoter)

# polya
col_id = 10
dhit2_polya, dhit_polya = read_tre_values(row_ids, col_id, chart_name)
plt_polya = plt.scatter(dhit_polya, dhit2_polya, linewidths=1, alpha=0.8, marker='o', color='#9370DB', s=size, linewidth=linewidth)

mean_dHIT2 += np.mean(dhit2_promoter)
mean_dHIT += np.mean(dhit_promoter)

mean_dHIT2 = mean_dHIT2 / 4
mean_dHIT = mean_dHIT / 4

x_min = 0
x_max = 1
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.plot((x_min, x_max), (x_min, x_max), 'k--', linewidth=border_linewidth)

ax.spines['top'].set_linewidth(border_linewidth)    # 设置上边框宽度
ax.spines['right'].set_linewidth(border_linewidth)  # 设置右边框宽度
ax.spines['left'].set_linewidth(border_linewidth)   # 设置左边框宽度
ax.spines['bottom'].set_linewidth(border_linewidth)  # 设置下边框宽度

# plt.xticks(fontproperties = 'Calibri', size = 20)
# plt.yticks(fontproperties = 'Calibri', size = 20)
plt.xlabel('dHIT Spearman Correlation')
plt.ylabel('BioSeq2Seq (RD) Spearman Correlation')
# plt.title(cellline, fontdict={'family': 'Calibri', 'size': 15})
plt.legend((plt_promoter, plt_enhancer, plt_genebody, plt_insulator, plt_polya),
           ('Promoter', 'Enhancer', 'Genebody', 'Insulator', 'Polya'),
           bbox_to_anchor = (0.57, 0.5),
           frameon=False)
plt.text(x=0.65, y=0.03, s='mean=%.4f' % round(mean_dHIT, 4),
         fontdict=dict(color='black')
        )

plt.text(x=0.02, y=0.95, s='mean=%.4f' % round(mean_dHIT2, 4),
         fontdict=dict(color='black')
        )

plt.show()
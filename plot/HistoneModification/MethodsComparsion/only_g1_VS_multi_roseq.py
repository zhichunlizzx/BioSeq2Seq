import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

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

mean_dHIT2 = 0
mean_muiti_cell = 0

linewidths=0
size=20

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

mean_dHIT2 = 0
mean_only_G1 = 0
# K562
# 128 bp
# roc_dHIT2 = [0.6219, 0.5917, 0.768, 0.7654, 0.7882, 0.4678, 0.7512, 0.8297, 0.5401, 0.5432]
# roc_only_G1 = [0.6539, 0.6467, 0.8151, 0.8238, 0.8296, 0.4753, 0.8195, 0.8848, 0.5568, 0.5751]

# G8 1k
roc_dHIT2 = [0.75, 0.7316, 0.886, 0.8909, 0.8153, 0.6748, 0.8169, 0.871, 0.5653, 0.6937]
roc_only_G1 = [0.7415, 0.7084, 0.8658, 0.874, 0.7661, 0.6223, 0.7644, 0.8443, 0.5601, 0.6523]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_k562 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#d08fbc', s=size)

# GM12878
# 128
# roc_dHIT2 = [0.7038, 0.7514, 0.8502, 0.7554, 0.5266, 0.7148, 0.8248, 0.5003]
# roc_only_G1 = [0.6879, 0.6729, 0.8204, 0.7337, 0.4891, 0.6393, 0.8159, 0.49]
# 1 kb
roc_dHIT2 = [0.7634, 0.8075, 0.9025, 0.8228, 0.5789, 0.807, 0.8752, 0.6658]
roc_only_G1 = [0.748, 0.7353, 0.8756, 0.8048, 0.5291, 0.7241, 0.867, 0.6446]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_gm12878 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#37a8e0', s=size)


# Hct116
# 128
# roc_dHIT2 = [0.7053, 0.7779, 0.8108, 0.7201, 0.3093, 0.5442, 0.7789, 0.2737]
# roc_only_G1 = [0.7038, 0.78, 0.8065, 0.7128, 0.3175, 0.5342, 0.7737, 0.2667]

# 1 kb
roc_dHIT2 = [0.7612, 0.8345, 0.8505, 0.7832, 0.34, 0.6355, 0.8259, 0.3272]
roc_only_G1 = [0.7601, 0.8369, 0.8466, 0.7778, 0.3522, 0.6246, 0.8238, 0.3192]


mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

# 找个颜色
plt_hct116 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#8fadd8', s=size)


# Hela
# 128
# roc_dHIT2 = [0.616, 0.6776, 0.8404, 0.7635, 0.2847, 0.6772, 0.8056, 0.4265]
# roc_only_G1 = [0.5365, 0.5476, 0.759, 0.6108, 0.2079, 0.4535, 0.7011, 0.3852]

# 1 kb
roc_dHIT2 = [0.657, 0.7443, 0.8722, 0.8116, 0.2853, 0.7405, 0.8483, 0.5467]
roc_only_G1 = [0.5934, 0.6466, 0.8061, 0.68, 0.1911, 0.5094, 0.7504, 0.485]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_hela = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#d28d95', s=size)


# CD4
# 128
# roc_dHIT2 = [0.4623, 0.7668, 0.1987, 0.3786, 0.4763]
# roc_only_G1 = [0.4446, 0.7749, 0.1801, 0.3467, 0.4847]

# 1 kb
roc_dHIT2 = [0.5377, 0.8494, 0.2372, 0.4604, 0.6442]
roc_only_G1 = [0.5188, 0.8515, 0.2074, 0.418, 0.6521]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_cd4 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#4956a4', s=size)

# IMR90
roc_dHIT2 = [0.6599, 0.8386, 0.8035, 0.7603, 0.4686, 0.5601, 0.8382, 0.0175, 0.5076]
roc_only_G1 = [0.5965, 0.7757, 0.8396, 0.6812, 0.3418, 0.3703, 0.8038, 0.1056, 0.2946]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_imr90 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#76be5d', s=size)


# MCF-7
roc_dHIT2 = [0.5686, 0.7715, 0.8598, 0.5281, 0.071, 0.6214, 0.786, 0.1837, 0.3513]
roc_only_G1 = [0.569, 0.7555, 0.8504, 0.5258, 0.0495, 0.579, 0.7652, 0.1969, 0.2922]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_mcf7 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#49a9b1', s=size)


# HepG2
roc_dHIT2 = [0.52, 0.7194, 0.7254, 0.6456, 0.4716, 0.5745, 0.7797, 0.0197, 0.414]
roc_only_G1 = [0.5941, 0.7305, 0.8225, 0.7103, 0.5188, 0.6271, 0.8227, 0.0895, 0.4508]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_hepg2 = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#8fc7d4', s=size)

# Mouse
roc_dHIT2 = [0.3574, 0.7216, 0.7017, 0.3897, 0.1968, 0.2307, 0.5281, 0.3253]
roc_only_G1 = [0.3942, 0.7684, 0.7403, 0.3024, 0.2109, 0.1491, 0.479, 0.3366]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_only_G1 += np.mean(roc_only_G1)

plt_mouse = plt.scatter(roc_only_G1, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#c73a8f', s=size)




mean_dHIT2 = mean_dHIT2 / 9
mean_dHIT = mean_only_G1 / 9

x_min = 0
x_max = 1
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.plot((x_min, x_max), (x_min, x_max), 'k--')

# 设置边框宽度
# ax = plt.gca()
# for spine in ax.spines.values():
#     spine.set_linewidth(2)

plt.xticks()
plt.yticks()



plt.xlabel("Only One RO-seq Pearson Correlation")
plt.ylabel("Multi-RO-seq Pearson Correlation")
# plt.title(cellline, fontdict={'family': 'Times New Roman', 'size': 15})



plt.legend((plt_k562, plt_gm12878, plt_hct116, plt_hela, plt_cd4, plt_imr90, plt_mcf7, plt_hepg2, plt_mouse),
           ('K562', 'GM12878', 'HCT116', 'HeLa', 'CD4$^{+}$ T Cells', 'IMR90', 'MCF-7', 'HepG2', 'Mouse (Liver)'),
           bbox_to_anchor = (0.8, 0.8),
           frameon=False)
plt.text(x=0.65, y=0.03, s='mean=%.4f' % mean_dHIT)

plt.text(x=0.02, y=0.95, s='mean=%.4f' % mean_dHIT2)
plt.savefig(r'D:\pap\BioSeq2Seq\result\image\scatter\only_G1_VS_multi_roseq\pearson.pdf', bbox_inches='tight', format='pdf')

# plt.show()
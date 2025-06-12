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

# Hct116
# 1 kb
roc_dHIT2 = [0.7612, 0.8345, 0.8505, 0.7832, 0.34, 0.6355, 0.8259, 0.3774, 0.3272]
roc_muiti_cell = [0.7676, 0.8458, 0.8501, 0.8071, 0.3644, 0.6945, 0.8453, 0.3547, 0.3483]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_muiti_cell += np.mean(roc_muiti_cell)

plt_hct116 = plt.scatter(roc_muiti_cell, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#8fadd8', s=size)


# CD4
# 1 kb
roc_dHIT2 = [0.5377, 0.8494, 0.2372, 0.4604, 0.6442, 0.2887]
roc_muiti_cell = [0.5369, 0.8509, 0.2624, 0.4524, 0.638, 0.2436]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_muiti_cell += np.mean(roc_muiti_cell)

plt_cd4 = plt.scatter(roc_muiti_cell, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#4956a4', s=size)

# IMR90
roc_dHIT2 = [0.6599, 0.8386, 0.8035, 0.7603, 0.4686, 0.5601, 0.8382, 0.0175, 0.5076]
roc_muiti_cell = [0.5717, 0.6628, 0.7383, 0.454, 0.3798, 0.4018, 0.5771, -0.0009, 0.3459]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_muiti_cell += np.mean(roc_muiti_cell)

plt_imr90 = plt.scatter(roc_muiti_cell, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#76be5d', s=size)


# MCF-7
roc_dHIT2 = [0.5686, 0.7715, 0.8598, 0.5281, 0.071, 0.6214, 0.786, 0.1837, 0.3513]
roc_muiti_cell = [0.572, 0.7677, 0.8551, 0.5417, 0.0675, 0.6382, 0.78, 0.1848, 0.3113]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_muiti_cell += np.mean(roc_muiti_cell)

plt_mcf7 = plt.scatter(roc_muiti_cell, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#49a9b1', s=size)


# HepG2
roc_dHIT2 = [0.52, 0.7194, 0.7254, 0.6456, 0.4716, 0.5745, 0.7797, 0.0197, 0.414]
roc_muiti_cell = [0.5996, 0.7281, 0.81, 0.7383, 0.4058, 0.616, 0.8165, 0.058, 0.4298]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_muiti_cell += np.mean(roc_muiti_cell)

plt_hepg2 = plt.scatter(roc_muiti_cell, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#8fc7d4', s=size)

# HepG2
roc_dHIT2 = [0.3574, 0.7216, 0.7017, 0.3897, 0.1968, 0.2307, 0.5281, 0.3253]
roc_muiti_cell = [0.3942, 0.7684, 0.7403, 0.3024, 0.2109, 0.1491, 0.479, 0.3366]

mean_dHIT2 += np.mean(roc_dHIT2)
mean_muiti_cell += np.mean(roc_muiti_cell)

plt_mouse = plt.scatter(roc_muiti_cell, roc_dHIT2, linewidths=linewidths, alpha=0.8, marker='o', color='#c73a8f', s=size)


mean_dHIT2 = mean_dHIT2 / 6
mean_dHIT = mean_muiti_cell / 6


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



plt.xlabel("Muiti-Cell Pearson Correlation")
plt.ylabel("Multi-RO-seq Pearson Correlation")
# plt.title(cellline, fontdict={'family': 'Times New Roman', 'size': 15})



plt.legend((plt_hct116, plt_cd4, plt_imr90, plt_mcf7, plt_hepg2, plt_mouse),
           ('HCT116', 'CD4$^{+}$ T Cells', 'IMR90', 'MCF-7', 'HepG2', 'Mouse (Liver)'),
           bbox_to_anchor = (0.55, 0.52),
           frameon=False)
plt.text(x=0.65, y=0.03, s='mean=%.4f' % mean_dHIT)

plt.text(x=0.02, y=0.95, s='mean=%.4f' % mean_dHIT2)

plt.show()
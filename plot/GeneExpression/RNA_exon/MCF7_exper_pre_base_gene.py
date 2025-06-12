import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体为Arial，字体大小为7
plt.rc('font', family='Arial', size=7)

# 尺寸设置（毫米转英寸）
width_mm, height_mm = 57.23, 47.359
width_inch = width_mm / 25.4
height_inch = height_mm / 25.4

# 文件路径
file_pre_exon = r'F:\result\RNA\exon_gene\MCF7\exon_stat\aggregated_predicted_exon.bed'
file_exper_exon = r'F:\result\RNA\exon_gene\MCF7\exon_stat\aggregated_experimental_exon.bed'
file_pre_gene = r'F:\result\RNA\exon_gene\MCF7\exon_stat\predicted_gene.bed'
file_exper_gene = r'F:\result\RNA\exon_gene\MCF7\exon_stat\experimental_gene.bed'

# 加载数据
def load_data(file):
    return np.asarray(np.loadtxt(file, dtype='str')[:, 4], dtype='float')

data_pre_exon = load_data(file_pre_exon)
data_exper_exon = load_data(file_exper_exon)
data_pre_gene = load_data(file_pre_gene)
data_exper_gene = load_data(file_exper_gene)

# 创建DataFrame
data = np.concatenate([data_pre_gene, data_pre_exon, data_exper_gene, data_exper_exon])
data_type = np.concatenate([['gene']*len(data_pre_gene),
                           ['exon']*len(data_pre_exon),
                           ['gene']*len(data_exper_gene),
                           ['exon']*len(data_exper_exon)])
data_ori = np.concatenate([['predicted']*len(data_pre_gene),
                          ['predicted']*len(data_exper_gene),
                          ['experiment']*len(data_pre_exon),
                          ['experiment']*len(data_exper_exon)])

df = pd.DataFrame({'value': data, 'type': data_type, 'source': data_ori})

# 自定义颜色
palette = {
    'gene': '#5d85e2', 
    'exon': '#edd791'
}

# 创建图形
fig, ax = plt.subplots(figsize=(width_inch, height_inch))

# 绘制箱线图 - 所有边框设为黑色
boxprops = {'linewidth': 0.5, 'edgecolor': 'black'}
medianprops = {'linewidth': 0.5, 'color': 'black'}
whiskerprops = {'linewidth': 0.5, 'color': 'black'}
capprops = {'linewidth': 0.5, 'color': 'black'}
flierprops = {'marker': 'o', 'markersize': 3, 'markerfacecolor': 'black'}

sns.boxplot(data=df, x='source', y='value', hue='type', 
            palette=palette, width=0.6, linewidth=0.5,
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops,
            flierprops=flierprops, showfliers=False, ax=ax)

# 统一设置所有箱体边框为黑色（确保覆盖所有元素）
for artist in ax.artists:
    artist.set_edgecolor('black')
    artist.set_linewidth(0.5)

# 设置坐标轴范围
ax.set_ylim(0, 350)

# 设置轴标签
ax.set_xlabel('')
ax.set_ylabel('Expression Value')

# 调整图例（如果需要保留）
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon=False)
ax.legend_.remove()  # 直接移除图例

# 调整布局
plt.tight_layout()

plt.show()
import gzip
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为 Arial，字体大小为 7
plt.rc('font', family='Arial', size=7)

# Calculating the required size in inches for the image
width_mm = 48.12  # width in mm
height_mm = 38.547 # height in mm

# Conversion factor from mm to inches
mm_to_inch = 25.4

# Convert to inches
width_inch = width_mm / mm_to_inch
height_inch = height_mm / mm_to_inch

def read_matrix_file(matrix_file):
    with gzip.open(matrix_file, 'rt') as f:
        lines = f.readlines()
    
    # Parsing the header to get labels
    header = lines[0].strip().split('\t')
    labels = header[6:]  # Skipping first 6 columns which are metadata

    # Parsing the data
    data = []
    for line in lines[1:]:
        values = list(map(float, line.strip().split('\t')[6:]))
        data.append(values)
    
    return labels, np.array(data)
 
def plot_line(cellline, model):
    # 读取并解析矩阵文件
    matrix_list = [
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K4me1.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K4me2.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K4me3.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K27ac.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K27me3.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K36me3.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K9ac.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H3K9me3.gz' % (cellline, model),
        r'deeptools_results\tss_5000\%s\%s\TSS_H4K20me1.gz' % (cellline, model),
    ]

    # color_list = ['#719ecb', '#353840', '#ab7d70', '#a4cf9f', '#f09a59', '#fec975', '#e7656c', '#ca98c9']
    color_list = ['#b9aeeb', '#f2b56f', '#fae69e', '#84c3b7', '#88d8db', '#71b7ed', '#f57c6e', '#A2D2BF', '#bc6356']
    linewidth = 0.5

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

    labels1, data1 = read_matrix_file(matrix_list[0])
    labels2, data2 = read_matrix_file(matrix_list[1])
    labels3, data3 = read_matrix_file(matrix_list[2])
    labels4, data4 = read_matrix_file(matrix_list[3])
    labels5, data5 = read_matrix_file(matrix_list[4])
    labels6, data6 = read_matrix_file(matrix_list[5])
    labels7, data7 = read_matrix_file(matrix_list[6])
    labels8, data8 = read_matrix_file(matrix_list[7])
    labels9, data9 = read_matrix_file(matrix_list[8])

    # print(data1)

    # 计算每个区域的平均值
    mean_profile1 = np.mean(data1, axis=0)
    mean_profile2 = np.mean(data2, axis=0)
    mean_profile3 = np.mean(data3, axis=0)
    mean_profile4 = np.mean(data4, axis=0)
    mean_profile5 = np.mean(data5, axis=0)
    mean_profile6 = np.mean(data6, axis=0)
    mean_profile7 = np.mean(data7, axis=0)
    mean_profile8 = np.mean(data8, axis=0)
    mean_profile9 = np.mean(data9, axis=0)

    # plt.ylim(1, 10)
    # 绘制图表
    # plt.figure(figsize=(10, 5))
    plt.plot(mean_profile1, color=color_list[0], linewidth=linewidth, label='H3K4me1')
    plt.plot(mean_profile2, color=color_list[1], linewidth=linewidth, label='H3K4me2')
    plt.plot(mean_profile3, color=color_list[2], linewidth=linewidth, label='H3K4me3')
    plt.plot(mean_profile4, color=color_list[3], linewidth=linewidth, label='H3K27ac')
    plt.plot(mean_profile5, color=color_list[4], linewidth=linewidth, label='H3K27me3')
    plt.plot(mean_profile6, color=color_list[5], linewidth=linewidth, label='H3K36me3')
    plt.plot(mean_profile7, color=color_list[6], linewidth=linewidth, label='H3K9ac')
    plt.plot(mean_profile8, color=color_list[7], linewidth=linewidth, label='H3K9me3')
    plt.plot(mean_profile9, color=color_list[8], linewidth=linewidth, label='H4K20me1')
    

    # 设置对称的x轴刻度
    x_ticks = [0, 500, 1000]
    x_labels = ['-5.0 k', '0', '5.0 k']
    plt.gca().set_xticks(x_ticks)
    plt.gca().set_xticklabels(x_labels)

    # 添加标签和标题
    plt.xlabel('Distance to TSS')
    plt.ylabel('Signal')
    # plt.title('PlotProfile using deeptools matrix')
    # plt.yticks(fontproperties = 'Calibri', size = 20)
    # plt.xticks(fontproperties = 'Calibri', size = 20)
    # 移除上方和右方的边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置x轴和y轴边框的宽度
    ax.spines['bottom'].set_linewidth(linewidth*2)
    ax.spines['left'].set_linewidth(linewidth*2)

    # 添加图例
    # plt.legend(frameon=False)

    plt.show()


celllines = ['K562']
models = ['dHIT2']
for cellline in celllines:
    for model in models:
        print(cellline, model)
        plot_line(cellline, model)
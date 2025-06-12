import numpy as np
import matplotlib.pyplot as plt
import xlrd
from matplotlib.lines import Line2D
'''
scatter
'''


def excel_data(data_path, row_id, col_id, cell_line, lines=[0, 1, 2, 3, 4, 5, 6, 7, 9]):
    workbook = xlrd.open_workbook(data_path)
    table = workbook.sheet_by_name(cell_line)
    results = []
    for i in lines:
        results.append(round(table.row(row_id+i)[col_id].value, 2))
    return results


def main():
    data_path = r'D:\code\BioSeq2Seq\plot\HistoneModification\plot_data\correlation.xlsx'
    linewidths = 1
    alpha=1
    magnification = 3000

    # gm
    sup1 = plt.subplot(1, 5, 1)
    sup2 = plt.subplot(1, 5, 2)
    sup3 = plt.subplot(1, 5, 3)
    sup4 = plt.subplot(1, 5, 4)
    sup5 = plt.subplot(1, 5, 5)

    column = 2

    plt.sca(sup1)

    cell_line = 'k562'
    pearson_dHIT2 = excel_data(data_path, 272, column, cell_line)
    pearson_dHIT = [0.6, 0.54, 0.69, 0.68, 0.68, 0.37, 0.67, 0.68, 0.47]

    # color_list = ['peru', 'deepskyblue', 'brown', 'springgreen', 'deeppink', 'tomato', 'gold', 'grey', 'purple']
    color_list = ['#ec5d57', '#be8fa6', '#ebcc90', '#4cb1ba', '#e79798', '#f1987d', '#f9c255', '#d2d166', '#4d97bb']
    # plt.ylabel('dHIT2 model', fontdict={'family': 'Calibri', 'size': 15})
    # plt.xlabel('dHIT model', fontdict={'family': 'Calibri', 'size': 15})
    p1 = plt.scatter(pearson_dHIT[:1], pearson_dHIT2[:1], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[0]-pearson_dHIT[0]), alpha=alpha, marker='o', color=color_list[0])
    p2 = plt.scatter(pearson_dHIT[1:2], pearson_dHIT2[1:2], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[1]-pearson_dHIT[1]), alpha=alpha, marker='o', color=color_list[1])
    p3 = plt.scatter(pearson_dHIT[2:3], pearson_dHIT2[2:3], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[2]-pearson_dHIT[2]), alpha=alpha, marker='o', color=color_list[2])
    p4 = plt.scatter(pearson_dHIT[3:4], pearson_dHIT2[3:4], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[3]-pearson_dHIT[3]), alpha=alpha, marker='o', color=color_list[3])
    p5 = plt.scatter(pearson_dHIT[4:5], pearson_dHIT2[4:5], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[4]-pearson_dHIT[4]), alpha=alpha, marker='o', color=color_list[4])
    p6 = plt.scatter(pearson_dHIT[5:6], pearson_dHIT2[5:6], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[5]-pearson_dHIT[5]), alpha=alpha, marker='o', color=color_list[5])
    p7 = plt.scatter(pearson_dHIT[6:7], pearson_dHIT2[6:7], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[6]-pearson_dHIT[6]), alpha=alpha, marker='o', color=color_list[6])
    p8 = plt.scatter(pearson_dHIT[7:8], pearson_dHIT2[7:8], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[7]-pearson_dHIT[7]), alpha=alpha, marker='o', color=color_list[7])
    p9 = plt.scatter(pearson_dHIT[8:9], pearson_dHIT2[8:9], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[8]-pearson_dHIT[8]), alpha=alpha, marker='o', color=color_list[8])


    # plt.text(0.38, 0.90, 'K562', fontdict={'family': 'Calibri', 'size': 15})
    plt.plot((0, 1), (0, 1), linewidth=0.5, color='black')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    size = 20
    legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='H3K122ac', markerfacecolor=color_list[0], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K4me1', markerfacecolor=color_list[1], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K4me2', markerfacecolor=color_list[2], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K4me3', markerfacecolor=color_list[3], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K27ac', markerfacecolor=color_list[4], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K27me3', markerfacecolor=color_list[5], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K36me3', markerfacecolor=color_list[6], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H3K9ac', markerfacecolor=color_list[7], markersize=size, alpha=alpha),
                Line2D([0], [0], marker='o', color='w', label='H4K20me1', markerfacecolor=color_list[8], markersize=size, alpha=alpha)
                ]

    plt.legend(handles=legend_elements, frameon=False, ncol=9, prop={'family': 'Calibri', 'size': 15}, bbox_to_anchor=(5.3, 1.15))

    # gm12878
    plt.sca(sup2)
    column = 2
    lines = [0, 1, 2, 3, 4, 5, 6, 8]
    cell_line = 'gm12878'
    pearson_dHIT2 = excel_data(data_path, 17, column, cell_line, lines=lines)
    pearson_dHIT = [0.67, 0.79, 0.83, 0.71, 0.39, 0.78, 0.83, 0.56]

    color_list = ['#be8fa6', '#ebcc90', '#4cb1ba', '#e79798', '#f1987d', '#f9c255', '#d2d166', '#4d97bb']

    # plt.xlabel('dHIT model', fontdict={'family': 'Calibri', 'size': 15})
    p1 = plt.scatter(pearson_dHIT[:1], pearson_dHIT2[:1], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[0]-pearson_dHIT[0]), alpha=alpha, marker='o', color=color_list[0])
    p2 = plt.scatter(pearson_dHIT[1:2], pearson_dHIT2[1:2], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[1]-pearson_dHIT[1]), alpha=alpha, marker='o', color=color_list[1])
    p3 = plt.scatter(pearson_dHIT[2:3], pearson_dHIT2[2:3], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[2]-pearson_dHIT[2]), alpha=alpha, marker='o', color=color_list[2])
    p4 = plt.scatter(pearson_dHIT[3:4], pearson_dHIT2[3:4], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[3]-pearson_dHIT[3]), alpha=alpha, marker='o', color=color_list[3])
    p5 = plt.scatter(pearson_dHIT[4:5], pearson_dHIT2[4:5], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[4]-pearson_dHIT[4]), alpha=alpha, marker='o', color=color_list[4])
    p6 = plt.scatter(pearson_dHIT[5:6], pearson_dHIT2[5:6], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[5]-pearson_dHIT[5]), alpha=alpha, marker='o', color=color_list[5])
    p7 = plt.scatter(pearson_dHIT[6:7], pearson_dHIT2[6:7], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[6]-pearson_dHIT[6]), alpha=alpha, marker='o', color=color_list[6])
    p8 = plt.scatter(pearson_dHIT[7:8], pearson_dHIT2[7:8], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[7]-pearson_dHIT[7]), alpha=alpha, marker='o', color=color_list[7])


    # plt.text(0.38, 0.90, 'GM12878', fontdict={'family': 'Calibri', 'size': 15})
    plt.plot((0, 1), (0, 1), linewidth=0.5, color='black')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    # hct116
    plt.sca(sup3)
    column = 2
    lines = [0, 1, 2, 3, 4, 5, 6, 8]
    cell_line = 'hct116'
    pearson_dHIT2 = excel_data(data_path, 17, column, cell_line, lines=lines)
    pearson_dHIT = [0.68, 0.84, 0.8, 0.72, 0.29, 0.7, 0.81, 0.5]

    color_list = ['#be8fa6', '#ebcc90', '#4cb1ba', '#e79798', '#f1987d', '#f9c255', '#d2d166', '#4d97bb']

    # plt.xlabel('dHIT model', fontdict={'family': 'Calibri', 'size': 15})
    p1 = plt.scatter(pearson_dHIT[:1], pearson_dHIT2[:1], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[0]-pearson_dHIT[0]), alpha=alpha, marker='o', color=color_list[0])
    p2 = plt.scatter(pearson_dHIT[1:2], pearson_dHIT2[1:2], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[1]-pearson_dHIT[1]), alpha=alpha, marker='o', color=color_list[1])
    p3 = plt.scatter(pearson_dHIT[2:3], pearson_dHIT2[2:3], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[2]-pearson_dHIT[2]), alpha=alpha, marker='o', color=color_list[2])
    p4 = plt.scatter(pearson_dHIT[3:4], pearson_dHIT2[3:4], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[3]-pearson_dHIT[3]), alpha=alpha, marker='o', color=color_list[3])
    p5 = plt.scatter(pearson_dHIT[4:5], pearson_dHIT2[4:5], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[4]-pearson_dHIT[4]), alpha=alpha, marker='o', color=color_list[4])
    p6 = plt.scatter(pearson_dHIT[5:6], pearson_dHIT2[5:6], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[5]-pearson_dHIT[5]), alpha=alpha, marker='o', color=color_list[5])
    p7 = plt.scatter(pearson_dHIT[6:7], pearson_dHIT2[6:7], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[6]-pearson_dHIT[6]), alpha=alpha, marker='o', color=color_list[6])
    p8 = plt.scatter(pearson_dHIT[7:8], pearson_dHIT2[7:8], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[7]-pearson_dHIT[7]), alpha=alpha, marker='o', color=color_list[7])


    # plt.text(0.38, 0.90, 'Hct116', fontdict={'family': 'Calibri', 'size': 15})
    plt.plot((0, 1), (0, 1), linewidth=0.5, color='black')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    # hela
    plt.sca(sup4)
    column = 2
    lines = [0, 1, 2, 3, 4, 5, 6, 8]
    cell_line = 'hela'
    pearson_dHIT2 = excel_data(data_path, 17, column, cell_line, lines=lines)
    pearson_dHIT = [0.58, 0.7, 0.76, 0.78, 0.19, 0.66, 0.79, 0.41]

    color_list = ['#be8fa6', '#ebcc90', '#4cb1ba', '#e79798', '#f1987d', '#f9c255', '#d2d166', '#4d97bb']
    
    # plt.xlabel('dHIT model', fontdict={'family': 'Calibri', 'size': 15})
    p1 = plt.scatter(pearson_dHIT[:1], pearson_dHIT2[:1], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[0]-pearson_dHIT[0]), alpha=alpha, marker='o', color=color_list[0])
    p2 = plt.scatter(pearson_dHIT[1:2], pearson_dHIT2[1:2], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[1]-pearson_dHIT[1]), alpha=alpha, marker='o', color=color_list[1])
    p3 = plt.scatter(pearson_dHIT[2:3], pearson_dHIT2[2:3], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[2]-pearson_dHIT[2]), alpha=alpha, marker='o', color=color_list[2])
    p4 = plt.scatter(pearson_dHIT[3:4], pearson_dHIT2[3:4], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[3]-pearson_dHIT[3]), alpha=alpha, marker='o', color=color_list[3])
    p5 = plt.scatter(pearson_dHIT[4:5], pearson_dHIT2[4:5], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[4]-pearson_dHIT[4]), alpha=alpha, marker='o', color=color_list[4])
    p6 = plt.scatter(pearson_dHIT[5:6], pearson_dHIT2[5:6], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[5]-pearson_dHIT[5]), alpha=alpha, marker='o', color=color_list[5])
    p7 = plt.scatter(pearson_dHIT[6:7], pearson_dHIT2[6:7], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[6]-pearson_dHIT[6]), alpha=alpha, marker='o', color=color_list[6])
    p8 = plt.scatter(pearson_dHIT[7:8], pearson_dHIT2[7:8], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[7]-pearson_dHIT[7]), alpha=alpha, marker='o', color=color_list[7])


    # plt.text(0.38, 0.90, 'Hela', fontdict={'family': 'Calibri', 'size': 15})
    plt.plot((0, 1), (0, 1), linewidth=0.5, color='black')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    # cd4
    plt.sca(sup5)
    column = 2
    lines = [0, 2, 4, 5, 6]
    cell_line = 'CD4'
    pearson_dHIT2 = excel_data(data_path, 17, column, cell_line, lines=lines)
    pearson_dHIT = [0.5, 0.72, 0.27, 0.56, 0.54]

    color_list = ['#be8fa6', '#4cb1ba', '#f1987d', '#f9c255', '#d2d166']

    # plt.xlabel('dHIT model', fontdict={'family': 'Calibri', 'size': 15})
    p1 = plt.scatter(pearson_dHIT[:1], pearson_dHIT2[:1], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[0]-pearson_dHIT[0]), alpha=alpha, marker='o', color=color_list[0])
    p2 = plt.scatter(pearson_dHIT[1:2], pearson_dHIT2[1:2], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[1]-pearson_dHIT[1]), alpha=alpha, marker='o', color=color_list[1])
    p3 = plt.scatter(pearson_dHIT[2:3], pearson_dHIT2[2:3], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[2]-pearson_dHIT[2]), alpha=alpha, marker='o', color=color_list[2])
    p4 = plt.scatter(pearson_dHIT[3:4], pearson_dHIT2[3:4], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[3]-pearson_dHIT[3]), alpha=alpha, marker='o', color=color_list[3])
    p5 = plt.scatter(pearson_dHIT[4:5], pearson_dHIT2[4:5], linewidths=linewidths, s=magnification*abs(pearson_dHIT2[4]-pearson_dHIT[4]), alpha=alpha, marker='o', color=color_list[4])

    # plt.text(0.38, 0.90, 'CD4', fontdict={'family': 'Calibri', 'size': 15})
    plt.plot((0, 1), (0, 1), linewidth=0.5, color='black')
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    plt.show()


if __name__ == '__main__':
    main()
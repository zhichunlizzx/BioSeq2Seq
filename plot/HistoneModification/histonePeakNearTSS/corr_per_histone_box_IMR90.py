import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd
import seaborn as sns


def excel_data(data_path, row_id, col_id, table_id, lines=[0, 1, 2, 3, 4, 5, 6, 8]):

    workbook = xlrd.open_workbook(data_path)
    table = workbook.sheet_by_name(table_id)
    results = []
    for i in lines:
        results.append(table.row(row_id+i)[col_id].value)
    return results


def draw_violin(data, label_list, cellline):
    # 设置全局字体为 Arial，字体大小为 7
    plt.rc('font', family='Arial', size=7)

    # Calculating the required size in inches for the image
    width_mm = 77 # width in mm
    height_mm = 58 # height in mm

    # Conversion factor from mm to inches
    mm_to_inch = 25.4

    # Convert to inches
    width_inch = width_mm / mm_to_inch
    height_inch = height_mm / mm_to_inch

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))

    sns.boxplot(x="label",  # 指定x轴的数据
                   y="value",  # 指定y轴的数据
                   hue="model_type",  # 指定分组变量
                   data=data,  # 指定绘图的数据集
                   order=label_list,  # 指定x轴刻度标签的顺序
                   width=0.5,
                   linewidth=0.8,
                   palette=['#FB8F33', '#5A9DD4'],  # 指定不同性别对应的颜色（因为hue参数为设置为性别变量）
                   boxprops=dict(edgecolor="black"),  # 设置箱体边框颜色为黑色
                   whiskerprops=dict(color="black", linewidth=0.8),  # 设置 whiskers 线条颜色和宽度
                   capprops=dict(color="black", linewidth=0.8),  # 设置 caps 线条颜色和宽度
                   medianprops=dict(color="black", linewidth=0.8),  # 设置中位数线条颜色和宽度
                   flierprops=dict(markerfacecolor='black', markeredgecolor='black', markersize=0.8)  # 设置离群点的颜色
                   )
    
    plt.ylim([0.0, 1.])
    plt.xlabel('Variant TSS distance')
    plt.ylabel('Spearman Correlation')
    plt.title(cellline)
    plt.legend(loc="upper right", frameon=False)
    plt.show()

if __name__ == '__main__':
    



    excel_file = r"\BioSeq2Seq\plot\HistoneModification\plot_data\evaluation.xlsx"
    
    # lines = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    lines = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cellline = 'IMR90'
    file_list = ['1k', '1k_10k', '10k_30k', 'away_30k']
    label_list = ['0-1000', '1000-10000', '10000-30000', '>30000']
    row = 81
    table_id = 'histone_peak_1k_10k_30k'

    data_corr = np.empty(shape=(1, 3))
    for i in range(4):
        # dhit2
        dhit2_corr_data = np.expand_dims(excel_data(excel_file, row, 2 * i + 2, table_id, lines=lines), axis=-1)
        model_type = np.expand_dims(['dHIT2'] * len(dhit2_corr_data), axis=-1)
        range_type = np.expand_dims([label_list[i]] * len(dhit2_corr_data), axis=-1)
        dhit2_corr_data = np.concatenate((dhit2_corr_data, model_type, range_type), axis=-1)
        data_corr = np.concatenate((data_corr, dhit2_corr_data), axis=0)

        # dhit
        dhit_corr_data = np.expand_dims(excel_data(excel_file, row, 2 * i + 12, table_id, lines=lines), axis=-1)
        model_type = np.expand_dims(['dHIT'] * len(dhit_corr_data), axis=-1)
        range_type = np.expand_dims([label_list[i]] * len(dhit_corr_data), axis=-1)
        dhit_corr_data = np.concatenate((dhit_corr_data, model_type, range_type), axis=-1)
        data_corr = np.concatenate((data_corr, dhit_corr_data), axis=0)

    data_corr = data_corr[1:]

    data_corr = pd.DataFrame(data=data_corr, columns=['value', 'model_type', 'label'])
    data_corr['value'] = data_corr['value'].apply(pd.to_numeric)
    # print(data_corr)

    draw_violin(data_corr, label_list, cellline)





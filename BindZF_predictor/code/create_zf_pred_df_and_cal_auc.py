import natsort
from glob import glob
import argparse
import pandas as pd
import numpy as np
from sklearn import metrics

"""This code calculates mean auc over all 157 proteins and
 creates a data frame with the ZFs classified as binding"""

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p_add', '--pred_add', help='predictions saving folders add ', type=str, required=True)
    parser.add_argument('-m_p', '--main_path', help='main path add ', type=str, required=True)

    args = parser.parse_args()
    arguments = vars(args)
    return arguments


def main (args):
    main_path = args['main_path']
    zf_data_df = pd.read_csv(main_path + 'zf_data_df.csv', sep=' ')
    path = args['pred_add']
    pred_paths = glob(path + '/*')
    pred_paths = natsort.natsorted(pred_paths)
    pred_list = []
    for i in range(pred_paths.__len__()):
        pred_list.append(np.load(pred_paths[i]))

    pred_df = pd.DataFrame(np.concatenate(pred_list, axis=0 ), columns=['pred_value'])
    thr =0.5
    pred_df['hard_label'] = pred_df.where(pred_df < thr, 1)
    pred_df['hard_label'].loc[(pred_df['hard_label'] != 1)] = 0

    ### calculate mean auc
    zf_data_df['pred_label_value'] = pred_df['pred_value']
    auc_l = []
    for i in zf_data_df['groups'].unique():
        tmp_zf = zf_data_df[zf_data_df['groups'] == i]
        fpr, tpr, thresholds = metrics.roc_curve(tmp_zf['connection_label'], tmp_zf['pred_label_value'], pos_label=1)
        auc_l.append(metrics.auc(fpr, tpr))
    print("mean auc %s" % np.nanmean(auc_l))
    print("std auc %s" % np.nanstd(auc_l))

    zf_data_df['pred_label'] = pred_df['hard_label']
    zf_pred_df = zf_data_df[zf_data_df['pred_label'] == 1]
    zf_pred_df.to_csv(main_path + 'zf_pred_df.csv', sep=' ', index=False)
    zf_pred_df.to_excel(main_path + 'zf_pred_df.xlsx', index=False)


if __name__ == "__main__":
    args = user_input()
    main(args)
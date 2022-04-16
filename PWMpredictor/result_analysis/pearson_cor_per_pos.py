from scipy.stats import pearsonr
import numpy as np
import pandas as pd


def person_cor_func(arr, strat_index):
    return pearsonr(arr[strat_index: strat_index + 4], arr[strat_index + 12: strat_index + 16])[0]


def cal_p_c_per_pos(pred_mat, gt_mat, s_f_add):
    con_pred_gt = np.concatenate((pred_mat, gt_mat), axis=1)
    pos_1_perason_arr = np.apply_along_axis(person_cor_func, 1, con_pred_gt, strat_index=0)
    pos_2_perason_arr = np.apply_along_axis(person_cor_func, 1, con_pred_gt, strat_index=4)
    pos_3_perason_arr = np.apply_along_axis(person_cor_func, 1, con_pred_gt, strat_index=8)

    pos_pearson_l =[pos_1_perason_arr, pos_2_perason_arr, pos_3_perason_arr]

    pc_pos_df = pd.DataFrame(pos_1_perason_arr, columns=['pos_1'])
    pc_pos_df['pos_2'] = pos_2_perason_arr
    pc_pos_df['pos_3'] = pos_3_perason_arr
    pc_pos_df.to_csv(s_f_add + 'pc_pos_df.csv', index=False)
    # mean and std pearson correlation of each position
    for i in range(pos_pearson_l.__len__()):
        print('mean pearson correlation of position %d is %.4f' % (i+1, np.nanmean(pos_pearson_l[i])))
        print('std pearson correlation of position %d is %.4f' % (i+1, np.nanstd(pos_pearson_l[i])))
        print('\n')
    return

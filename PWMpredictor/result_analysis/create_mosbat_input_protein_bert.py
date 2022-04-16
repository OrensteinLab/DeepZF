import numpy as np
import pandas as pd
from glob import glob
import natsort
import argparse

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p_add', '--pred_folder_add', help='c_rc predictions folder add', type=str, required=True)
    parser.add_argument('-c_rc_add', '--c_rc_add', help='c_rc data frame folder add', type=str, required=True)
    parser.add_argument('-zf_add', '--zf_add', help='predicted binding zinc fingers  data frame folder add', type=str, required=True)
    parser.add_argument('-s_add', '--s_add', help='saving folder add', type=str, required=True)
    parser.add_argument('-exp_name', '--exp_name', help='type of experiment', type=str, required=True)

    args = parser.parse_args()
    arguments = vars(args)

    return arguments



def create_txt_file_c_rc(f, mat, prot_name, g_a, df):

    groups = df['groups']
    k = 0
    for i in range(prot_name.shape[0]):
        if groups[groups == i].shape[0] == 0:
            continue
        print('Gene' + '\t' + prot_name[i], file=f)
        print('Motif' + '\t' + prot_name[i], file=f)
        print('Pos' + '\t' + 'A' + '\t' + 'C' + '\t' + 'G' + '\t' + 'T', file=f)

        for pos in range((g_a[g_a == i]).shape[0]*3):
            f.write(('{pos}' + '\t').format(pos=pos + 1))
            for j in range(3):
                f.write(('{mat}' + '\t').format(mat=mat[k][j]))  # print(f'{mat[i][j]}' + '\t', file=f)
            f.write(('{mat}' + '\n').format(mat=mat[k][j + 1]))
            k = k + 1

        f.write('\n' + '\n')

    f.close()
    return

def create_txt_file_zf_pred(f, mat, prot_name, g_a):
    p = 0
    k = 0
    for i in g_a.unique():
        print('Gene' + '\t' + prot_name[p], file=f)
        print('Motif' + '\t' + prot_name[p], file=f)
        print('Pos' + '\t' + 'A' + '\t' + 'C' + '\t' + 'G' + '\t' + 'T', file=f)

        for pos in range((g_a[g_a == i]).shape[0]*3):
            f.write(('{pos}' + '\t').format(pos=pos + 1))
            for j in range(3):
                f.write(('{mat}' + '\t').format(mat=mat[k][j]))  # print(f'{mat[i][j]}' + '\t', file=f)
            f.write(('{mat}' + '\n').format(mat=mat[k][j + 1]))
            k = k + 1

        f.write('\n' + '\n')
        p = p+1


    f.close()
    return

def main(args):

    path = args["pred_folder_add"]
    paths = glob(path + '/*')
    paths = natsort.natsorted(paths)
    print(paths.__len__())
    pred_list = []
    for i in range(paths.__len__()):
        tmp = np.load(paths[i])
        a = np.reshape(tmp, (int(tmp.shape[0]/12), 12))
        pred_list.append(a)

    c_rc_df = pd.read_csv(args["c_rc_add"], sep=' ')

    prot_name = c_rc_df['UniProt_ID'].unique()
    g_a = c_rc_df['groups']
    miss_group = []  # in case you have predected for a protein that all ZFs do not bind
    if not miss_group:
        for i in miss_group:
            tmp = c_rc_df[c_rc_df['groups'] == i].index
            c_rc_df.drop(tmp, inplace=True)
    pred_mat = pred_list[0]
    for i in range(pred_list.__len__() - 1):
        pred_mat = np.concatenate((pred_mat, pred_list[i+1]), axis=0)

    mat = np.reshape(pred_mat, (pred_mat.shape[0]*3,  int(pred_mat.shape[1]/3)))
    gt_mat = (c_rc_df.filter(items=['A1', 'C1', 'G1', 'T1', 'A2', 'C2', 'G2', 'T2', 'A3', 'C3', 'G3', 'T3'])).values # ground truth mat


    gt_mat = np.reshape(gt_mat, (gt_mat.shape[0]*3,  int(gt_mat.shape[1]/3)))

    f1 = open(args["s_add"] + args["exp_name"][-20:] + '_pred_pwm.txt', "w")
    f2 = open(args["s_add"] + 'gt_pwm.txt', "w")

    create_txt_file_c_rc(f2, gt_mat, prot_name, g_a, c_rc_df)
    zf_pred_df = pd.read_csv(args['zf_add'], sep=' ')
    prot_name = zf_pred_df['prot_name_id'].unique()
    g_a = zf_pred_df['groups']
    create_txt_file_zf_pred(f1, mat, prot_name, g_a)


if __name__ == "__main__":
    args = user_input()
    main(args)






from glob import glob
import natsort
import argparse
from pearson_cor_per_pos import *



def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p_add', '--pred_folder_add', help='c_rc predictions folder add', type=str, required=True)
    parser.add_argument('-c_rc_add', '--c_rc_add', help='c_rc data frame folder add', type=str, required=True)
    parser.add_argument('-s_add', '--s_add', help='saving folder add', type=str, required=True)

    args = parser.parse_args()
    arguments = vars(args)

    return arguments


def main(args):

    path = args["pred_folder_add"]
    paths = glob(path + '/*')
    paths = natsort.natsorted(paths)

    pred_list = []
    for i in range(paths.__len__()):
        tmp = np.load(paths[i])
        a = np.reshape(tmp, (int(tmp.shape[0]/12), 12))
        pred_list.append(a)


    c_rc_df = pd.read_csv(args["c_rc_add"], sep=' ')

    pred_mat = pred_list[0]
    for i in range(pred_list.__len__() - 1):
        pred_mat = np.concatenate((pred_mat, pred_list[i+1]), axis=0)

    gt_mat = (c_rc_df.filter(items=['A1', 'C1', 'G1', 'T1', 'A2', 'C2', 'G2', 'T2', 'A3', 'C3', 'G3', 'T3'])).values # ground truth mat
    cal_p_c_per_pos(pred_mat, gt_mat, args["s_add"])  # calculate mean pearson correlation for each position


if __name__ == "__main__":
    args = user_input()
    main(args)






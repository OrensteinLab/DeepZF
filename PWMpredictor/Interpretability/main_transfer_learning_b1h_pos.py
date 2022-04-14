import tensorflow as tf
import argparse
from models_transfer_learning import *
# -d_add C:/Users/User/PycharmProjects/zf_binding_pref/Transfer_learning/data_labels/ -add C:/Users/User/PycharmProjects/zf_binding_pref/PBM/results -lr 0.001 -e 2 -res_num 4 -r 0 -t_v retrain -ac_x True


def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_add', '--data_folder_address', help='main data and lables folder', type=str, required=True)
    parser.add_argument('-add', '--folder_address', help='main folder address for savings', type=str, required=True)
    parser.add_argument('-lr', '--learning_rate', help='learning rate of adam optimizer', type=float, required=True)
    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, required=True)
    parser.add_argument('-res_num', '--residual_num', help='number of residuals to use', type=int, required=True)
    parser.add_argument('-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True)
    parser.add_argument('-t_v', '--transfer_version', help='fine tuning or continue train', type=str, required=True)
    parser.add_argument('-ac_x', '--amino_acid_x', help='use b1h data with amino acid x', type=str, required=True)



    args = parser.parse_args()
    arguments = vars(args)

    return arguments
def main(args):
    if args["run_gpu"] == 1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        # force the server to run on cpu and not on Gpu
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main_path = args['data_folder_address']

    gt_crc = np.load(main_path + 'ground_truth_c_rc.npy')

    if args["residual_num"] == 12:
        one_hot_crc = np.load(main_path + 'onehot_encoding_c_rc_12res.npy')
        b1h_one_hot = np.load(main_path + 'onehot_encoding_b1h_12res.npy')
        b1h_pwm = np.load(main_path + 'ground_truth_b1h_pwm_12res.npy')
        train_all_data_model_model(b1h_one_hot, b1h_pwm, one_hot_crc, gt_crc, args['folder_address'], args['learning_rate'], args['epochs'], args['transfer_version'])


    if args["residual_num"] == 7:
        one_hot_crc = np.load(main_path + 'onehot_encoding_c_rc_7b1hres.npy')

        if args['amino_acid_x']:
            one_hot_7res = np.load(main_path + 'onehot_encoding_7b1h_res_with_ac_x.npy')
            pwm_7res = np.load(main_path + 'ground_truth_b1h_pwm_with_ac_x.npy')
        else:
            one_hot_7res = np.load(main_path + 'onehot_encoding_b1h_7res.npy')
            pwm_7res = np.load(main_path + 'ground_truth_b1h_pwm_7res.npy')

        train_all_data_model_model(one_hot_7res, pwm_7res, one_hot_crc, gt_crc, args['folder_address'], args['learning_rate'], args['epochs'], args['transfer_version'])

    if args["residual_num"] == 4:       
        one_hot_4res = np.load(main_path + 'onehot_encoding_b1h_4res.npy')
        pwm_4res = np.load(main_path + 'ground_truth_b1h_pwm_4res.npy')
        one_hot_crc = np.load(main_path + 'onehot_encoding_c_rc_4res.npy')
        train_all_data_model_model(one_hot_4res, pwm_4res, one_hot_crc, gt_crc, args['folder_address'], args['learning_rate'], args['epochs'], args['transfer_version'])




if __name__ == "__main__":
    args = user_input()
    main(args)



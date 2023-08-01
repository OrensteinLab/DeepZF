import tensorflow as tf
from models_PWMpredictor import *
import argparse


def user_input():
    parser = argparse.ArgumentParser()
#    parser.add_argument('-d_add', '--data_folder_address', help='main data and lables folder', type=str, required=True)
#    parser.add_argument('-add', '--folder_address', help='main folder address for savings', type=str, required=True)
#    parser.add_argument('-zf_p_df', '--pred_zf_df', help='predicted binding zinc fingers df', type=str, required=True)
#    parser.add_argument('-lr', '--learning_rate', help='learning rate of adam optimizer', type=float, required=True)
#    parser.add_argument('-e', '--epochs', help='number of epochs', type=int, required=True)
#    parser.add_argument('-res_num', '--residual_num', help='number of residuals to use', type=int, required=True)
#    parser.add_argument('-r', '--run_gpu', help='equal 1 if should run on gpu', type=int, required=True)
#    parser.add_argument('-t_v', '--transfer_version', help='last_layer or retrain', type=str, required=True)
#    parser.add_argument('-ac_x', '--amino_acid_x', help='use b1h data with amino acid x', type=str, required=True)
    parser.add_argument('-in', '--input_file', help='input zinc finger file', type=str, required=True)
    parser.add_argument('-out', '--output_file', help='output PWN predictions file', type=str, required=True)
    parser.add_argument('-m', '--model_file', help='trained model file', type=str, required=True)

    args = parser.parse_args()
    arguments = vars(args)
    return arguments


def main(args):
#    if args["run_gpu"] == 1:
#        gpus = tf.config.experimental.list_physical_devices('GPU')
#        if gpus:
#            try:
#                for gpu in gpus:
#                    tf.config.experimental.set_memory_growth(gpu, True)
#            except RuntimeError as e:
#                print(e)
#    else:
#        # force the server to run on cpu and not on Gpu
#        import os
#        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
#        os.environ["CUDA_VISIBLE_DEVICES"] = ""

#    main_path = args['data_folder_address']
#    c_rc_df = pd.read_csv(main_path + 'c_rc_df.csv', sep=' ')
#    zf_pred_df = pd.read_csv(main_path + args['pred_zf_df'], sep=' ')
    c_rc_df = pd.read_csv(args['input_file'], sep=' ')

#    if args["residual_num"] == 12:
#        b1h_one_hot = np.load(main_path + 'onehot_encoding_b1h_12res.npy')
#        b1h_pwm = np.load(main_path + 'ground_truth_b1h_pwm_12res.npy')

#    if args["residual_num"] == 7:
#        if not args['amino_acid_x']:
#            b1h_one_hot = np.load(main_path + 'onehot_encoding_b1h_7res.npy')
#            b1h_pwm = np.load(main_path + 'ground_truth_b1h_pwm_7res.npy')
#        else:
#            b1h_one_hot = np.load(main_path + 'onehot_encoding_7b1h_res_with_ac_x.npy')
#            b1h_pwm = np.load(main_path + 'ground_truth_b1h_pwm_with_ac_x.npy')

#    if args["residual_num"] == 4:  # residual number is equal to 4
#        b1h_one_hot = np.load(main_path + 'onehot_encoding_b1h_4res.npy')
#        b1h_pwm = np.load(main_path + 'ground_truth_b1h_pwm_4res.npy')

    pipeline_model_predict(c_rc_df, args["model_file"], args["output_file"])
#(b1h_one_hot, b1h_pwm, c_rc_df, zf_pred_df, args['folder_address'],
#                   args['learning_rate'], args['epochs'], args['transfer_version'],
#                   args['residual_num'])



if __name__ == "__main__":
    args = user_input()
    main(args)




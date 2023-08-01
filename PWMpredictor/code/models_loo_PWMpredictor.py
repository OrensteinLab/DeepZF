from utiles_models_loo_PWMpredictor import *


def pipeline_model(b1h_input_data, b1h_label_mat, c_rc_df, zf_df, folder_address, lr, epochs, t_v, res_num):

    start1 = time.time()
    b1h_model = b1h_main_model_func(b1h_input_data, b1h_label_mat, folder_address, lr, epochs)
    b1h_model.trainable = True
    set_trainable_layers(b1h_model, t_v)
    pipeline_func(c_rc_df, zf_df, b1h_model, folder_address, lr, epochs, res_num, start1)

    return


from utiles_transfer_learning import *

def train_all_data_model_model(b1h_input_data, b1h_label_mat, crc_input_data, crc_label_mat, folder_address, lr, epochs, t_v):
    start1 = time.time()
    b1h_model = b1h_main_model_func(b1h_input_data, b1h_label_mat, folder_address, lr, epochs)
    b1h_model.trainable = True
    set_trainable_layers(b1h_model, t_v)
    train_all_data_func(crc_input_data, crc_label_mat, b1h_model, folder_address, lr, epochs, start1)
    return

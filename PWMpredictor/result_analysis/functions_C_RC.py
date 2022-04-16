
import numpy as np
from sklearn.utils import shuffle
from functions import *

def group_zf_by_protein_name(file_C_RC):
    """this function is responsible to group the zf according to their protein """
    protein = file_C_RC.UniProt_ID.unique()
    val = np.arange(0, len(protein) - 1)
    dic = dict((val, i) for i, val in enumerate(protein))
    file_C_RC['Source'] = file_C_RC['UniProt_ID']
    file_C_RC.rename(columns={"Source": "groups"}, inplace=True)
    return file_C_RC.replace({"groups": dic})


def split_data(data_mat, y_1, y_2, y_3, groups):
    # shuffle the data
    data_mat, y_1, y_2, y_3, groups = shuffle(data_mat, y_1, y_2, y_3, groups)
    # split data into train and test (test has 50 samples)
    data_train = data_mat[1:data_mat.shape[0] - 50 + 1, :]
    data_test = data_mat[data_mat.shape[0] - 50:, :]
    y_1_train = y_1[1:data_mat.shape[0] - 50 + 1, :]
    y_1_test = y_1[data_mat.shape[0] - 50:, :]
    y_2_train = y_2[1:data_mat.shape[0] - 50 + 1, :]
    y_2_test = y_2[data_mat.shape[0] - 50:, :]
    y_3_train = y_3[1:data_mat.shape[0] - 50 + 1, :]
    y_3_test = y_3[data_mat.shape[0] - 50:, :]
    groups = groups[1:data_mat.shape[0] - 50 + 1]

    return data_train, data_test, y_1_train, y_2_train, y_3_train, y_1_test, y_2_test, y_3_test, groups


def org_zf_neighbors(oneHot_C_RC_amino, file_C_RC):
    vec_gr = file_C_RC['groups'].array
    vec_zf = file_C_RC['ZF index'].array
    input_mat = np.zeros((oneHot_C_RC_amino.shape[0], oneHot_C_RC_amino.shape[1] * 3))

    for i in range(vec_gr.__len__()):
        input_mat[i, oneHot_C_RC_amino.shape[1]:oneHot_C_RC_amino.shape[1] * 2] = oneHot_C_RC_amino[i]

        if i + 1 < vec_gr.__len__() and vec_gr[i] == vec_gr[i + 1]:  # current and next groups are the same
            input_mat[i, oneHot_C_RC_amino.shape[1] * 2:oneHot_C_RC_amino.shape[1] * 3] = oneHot_C_RC_amino[i + 1]

        if i > 0 and vec_gr[i] == vec_gr[i - 1]:  # current and previous groups are the same
            input_mat[i, 0:oneHot_C_RC_amino.shape[1]] = oneHot_C_RC_amino[i - 1]
    return input_mat


def create_prot_file_of_C_RC(file_C_RC):
    prot_name = file_C_RC.UniProt_ID.unique()
    folder_address = 'C:/Users/User/Desktop/projects/Amino_DNA/C_RC/uniprot'
    uni_prot_file = open(folder_address + 'C_RC_all_prot.txt', 'w')
    for i in range(prot_name.size):
        line = prot_name[i]
        if i < prot_name.size - 1:
            uni_prot_file.write(line + ' ' + 'OR' + ' ')  # write the protein name
        elif i == prot_name.size - 1:
            uni_prot_file.write(line)  # write the protein name
    uni_prot_file.close()
    return

def main_zf_one_hot(C_RC_zf_seq_df):
    """This function is responsible to create a one hot matrix of the zinc finger """
    oneHot_matrix_zf = ht_one_hot_encode_amino_acids(C_RC_zf_seq_df['zf'])
    oneHot_matrix_zf_vec = []
    # create a matrix: each row represents Amino acid
    for array in oneHot_matrix_zf:
        oneHot_matrix_zf_vec.append(array.flatten())
    # convert list to np
    return np.array(oneHot_matrix_zf_vec)

def concat_input_models(zf_data_df):
    C_RC_zf_seq_df_7_res_temp = zf_data_df.zf.str.slice(start=2, stop=3) + \
                                zf_data_df.zf.str.slice(start=4, stop=9) + \
                                zf_data_df.zf.str.slice(start=11)
    C_RC_zf_seq_df_7_res = pd.DataFrame({'zf': C_RC_zf_seq_df_7_res_temp})

    C_RC_zf_seq_df_12_7_res_temp = zf_data_df.zf.str.slice(start=0, stop=2) + \
                                   zf_data_df.zf.str.slice(start=3, stop=4) + \
                                   zf_data_df.zf.str.slice(start=9, stop=11)
    C_RC_zf_seq_df_12_7_res = pd.DataFrame({'zf': C_RC_zf_seq_df_12_7_res_temp})

    C_RC_zf_seq_df_4_res_temp = zf_data_df.zf.str.slice(start=5, stop=6) + \
                                zf_data_df.zf.str.slice(start=7, stop=9) + \
                                zf_data_df.zf.str.slice(start=11)
    C_RC_zf_seq_df_4_res = pd.DataFrame({'zf': C_RC_zf_seq_df_4_res_temp})

    C_RC_zf_seq_df_12_4_res_temp = zf_data_df.zf.str.slice(start=0, stop=5) + \
                                   zf_data_df.zf.str.slice(start=6, stop=7) + \
                                   zf_data_df.zf.str.slice(start=9, stop=11)
    C_RC_zf_seq_df_12_4_res = pd.DataFrame({'zf': C_RC_zf_seq_df_12_4_res_temp})

    input_12_res = main_zf_one_hot(zf_data_df)
    input_7_res = main_zf_one_hot(C_RC_zf_seq_df_7_res)
    input_4_res = main_zf_one_hot(C_RC_zf_seq_df_4_res)
    input_12_7_res = main_zf_one_hot(C_RC_zf_seq_df_12_7_res)
    input_12_4_res = main_zf_one_hot(C_RC_zf_seq_df_12_4_res)
    return [input_12_res, input_7_res, input_4_res, input_12_7_res, input_12_4_res]

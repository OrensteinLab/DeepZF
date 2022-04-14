
from functions import *
from scipy.stats import pearsonr

def zf_extract_from_pred(zf_pred, zf_data_df, index, zf_1_sum_list, zf_1_max_list):
    zf_temp = zf_data_df[zf_data_df['groups'] == index]
    zf_temp.reset_index(drop=True, inplace=True)

    for i in range(zf_temp.shape[0]):
        zf_1_sum_list.append(np.sum(zf_pred[:, 1][zf_temp['zf_indx_start'][i]:zf_temp['zf_indx_end'][i]]))
        zf_1_max_list.append(np.max(zf_pred[:, 1][zf_temp['zf_indx_start'][i]:zf_temp['zf_indx_end'][i]]))
    return zf_1_sum_list, zf_1_max_list


def find_dup_zf_in_c_rc(c_rc_df):
    c_rc_df['_12_seq'] = c_rc_df['AA1'] + c_rc_df['AA2'] + c_rc_df['AA3'] + c_rc_df['AA4'] + c_rc_df['AA5'] + \
                         c_rc_df['AA6'] + c_rc_df['AA7'] + c_rc_df['AA8'] + c_rc_df['AA9'] + c_rc_df['AA10'] + \
                         c_rc_df['AA11'] + c_rc_df['AA12']
    c_rc_df['groups_zf'] = c_rc_df._12_seq
    group_unique_prot_name = dic_unique_amino_acids(c_rc_df._12_seq.unique())
    c_rc_df.replace({"groups_zf": group_unique_prot_name}, inplace=True)

    c_rc_df['if_dup_2'] = c_rc_df['groups_zf'].duplicated(keep='last')
    index_dup = []
    for i in range(c_rc_df.shape[0]):
        if c_rc_df.if_dup_2[i]:
            index_dup.append(i)

    index_dup.append(c_rc_df[((c_rc_df.UniProt_ID == 'Q9UKT9') & (c_rc_df.ZF_index == 4))].index.values[0])
    index_dup.append(c_rc_df[((c_rc_df.UniProt_ID == 'Q9Y2X9') & (c_rc_df.ZF_index == 4))].index.values[0])

    return index_dup


def extract_zf_dup(c_rc_df):
    # delete duplicated zf seq and 2 zf that don't exist in zf_data_df
    c_rc_df['_12_seq'] = c_rc_df['AA1'] + c_rc_df['AA2'] + c_rc_df['AA3'] + c_rc_df['AA4'] + c_rc_df['AA5'] + \
                         c_rc_df['AA6'] + c_rc_df['AA7'] + c_rc_df['AA8'] + c_rc_df['AA9'] + c_rc_df['AA10'] + \
                         c_rc_df['AA11'] + c_rc_df['AA12']
    c_rc_df.drop_duplicates(subset="_12_seq", keep='first', inplace=True)
    index = c_rc_df[((c_rc_df.UniProt_ID == 'Q9UKT9') & (c_rc_df.ZF_index == 4))].index
    c_rc_df.drop(index=index, inplace=True)
    index = c_rc_df[((c_rc_df.UniProt_ID == 'Q9Y2X9') & (c_rc_df.ZF_index == 4))].index
    c_rc_df.drop(index=index, inplace=True)
    c_rc_df.reset_index(drop=True, inplace=True)
    return c_rc_df


def extract_zf_from_con_pred(zf_data_df, pred_list):
    zf_1_sum_list = []
    zf_1_max_list = []
    for index in (range(pred_list.__len__())):
        zf_1_sum_list, zf_1_max_list = zf_extract_from_pred \
            (pred_list[index], zf_data_df, index, zf_1_sum_list, zf_1_max_list)
    return np.asarray(zf_1_sum_list), np.asarray(zf_1_max_list)


def get_label_mat(c_rc_df):
    label_mat = (c_rc_df.filter(items=['A1', 'C1', 'G1', 'T1', 'A2', 'C2', 'G2', 'T2', 'A3', 'C3', 'G3', 'T3'])).values
    return label_mat


def create_input_model(df, res_num):
    if res_num == 12:
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_12'])
    if res_num == 7:
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_7_b1h'])
    if res_num == 4:  # res_num == 4
        oneHot_C_RC_amino = oneHot_Amino_acid_vec(df['res_4'])

    return oneHot_C_RC_amino

# def create_test_model(zf_df):
#     # one hot encoding
#     oneHot_C_RC_amino = oneHot_Amino_acid_vec(zf_df['zf_seq'])
#     return oneHot_C_RC_amino


def get_pred_con_zf(zf_data_df, pred_list):
    zf_1_sum, zf_1_max = extract_zf_from_con_pred(zf_data_df, pred_list)

    zf_1_sum = np.where(zf_1_sum < (np.mean(zf_1_sum)), 0, 1)
    zf_1_max = np.where(zf_1_max < (np.mean(zf_1_max)), 0, 1)

    zf_data_df['con_pred_sum'] = zf_1_sum
    zf_data_df['con_pred_max'] = zf_1_max

    zf_pred_sum_df = zf_data_df[zf_data_df['con_pred_sum'] == 1]
    zf_pred_max_df = zf_data_df[zf_data_df['con_pred_max'] == 1]
    print(zf_pred_sum_df.shape)
    print(zf_pred_max_df.shape)

    zf_pred_sum_df.reset_index(drop=True, inplace=True)
    zf_pred_max_df.reset_index(drop=True, inplace=True)

    return zf_pred_sum_df, zf_pred_max_df

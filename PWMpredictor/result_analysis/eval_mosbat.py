import pandas as pd
import numpy as np
import argparse


def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_add', '--affinity_folder_address', help='affinity correlation results folder address', type=str, required=True)
    parser.add_argument('-e_add', '--energy_folder_address', help='energy correlation results folder address', type=str, required=True)
    parser.add_argument('-s_add', '--save_results_add', help='correlation resuts vinf folder address', type=str, required=True)
    args = parser.parse_args()
    arguments = vars(args)
    return arguments



def main(args):

    affinity_correl_df = pd.read_csv(args['affinity_folder_address'], sep='\t')
    energy_correl_df = pd.read_csv(args['energy_folder_address'], sep='\t')
    affinity_correl_mat = affinity_correl_df.values[:, 1:]
    energy_correl_mat = energy_correl_df.values[:, 1:]
    affinity_correl_diag = affinity_correl_mat.diagonal()
    energy_correl_diag = energy_correl_mat.diagonal()
    correlation_df = pd.DataFrame( energy_correl_diag, columns=['energy_correl'])

    
    correlation_df.to_excel(args["save_results_add"] + 'correlation_df.xlsx', index=False)
    correlation_df.to_csv(args["save_results_add"] + 'correlation_df.csv', index=False)



    np.save(args["save_results_add"] + 'affinity_cor', affinity_correl_diag)
    np.save(args["save_results_add"] + 'energy_cor', energy_correl_diag)

    ls = [affinity_correl_diag, energy_correl_diag]
    str = ['affinity correlation results', 'energy correlation results']

    for i in range(2):
        print(str[i])
        print('mean:')
        print(np.mean(ls[i]))
        print('std:')
        print(np.std(ls[i]))

if __name__ == "__main__":
    args = user_input()
    main(args)


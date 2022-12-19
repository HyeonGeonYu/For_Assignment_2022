import numpy as np
def ML(inp_class,QPSK_sym_perm):
    channel_H__for_ML = np.copy(inp_class.channel_H)
    channel_result__for_ML = np.copy(inp_class.channel_result)
    Trans_num = channel_H__for_ML.shape[0]
    x_hat = np.zeros_like(inp_class.channel_result)
    for idx_Trans in range(Trans_num):
        test = np.einsum('mn,rnd->rmd', channel_H__for_ML[idx_Trans], QPSK_sym_perm)
        distance = channel_result__for_ML[idx_Trans] - test
        min_idx = np.argmin(np.sqrt(np.einsum('rmn,rmn->r', np.conj(distance), distance).real))
        x_hat[idx_Trans]= QPSK_sym_perm[min_idx]

    return x_hat
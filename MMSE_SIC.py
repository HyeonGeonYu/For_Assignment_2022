import numpy as np

def MMSE_SIC(inp_class,QPSK_sym_arr):
    channel_H_for_MMSE_SIC = np.copy(inp_class.channel_H)
    channel_result_for_MMSE_SIC = np.copy(inp_class.channel_result)
    num_n = channel_H_for_MMSE_SIC.shape[0]
    x_hat_arr = np.empty((num_n,0),dtype='complex')
    max_idx_MMSE_SIC_arr = np.empty((0,num_n),dtype='uint')

    for iter_Tx_num in range(inp_class.Tx):
        H_h__for_MMSE_SIC = np.einsum('ijk->ikj', np.conj(channel_H_for_MMSE_SIC))
        H_h__H = np.einsum('abc,acd->abd', H_h__for_MMSE_SIC, channel_H_for_MMSE_SIC)
        ###아래식 iter_TX_num에 따라 power가 달라지니 고려해야한다.
        H_h__H__N0PNt__inv = np.linalg.pinv(
            H_h__H + inp_class.N0 / (inp_class.rootpower_of_symbol ** 2) * np.eye(inp_class.Tx))
        W_h_MMSE_SIC = np.einsum('abc,acd->abd', H_h__H__N0PNt__inv, H_h__for_MMSE_SIC)
        W_MMSE_SIC = np.einsum('aij->aji', np.conj(W_h_MMSE_SIC))
        Tx_power =(inp_class.rootpower_of_symbol)**2
        norm_wi_np0 = np.einsum('cab,cba->cb', W_MMSE_SIC, W_h_MMSE_SIC).real/(inp_class.rootpower_of_symbol)**2

        W_h_MMSE_SIC__H = np.einsum('abc,acd->abd', W_h_MMSE_SIC, channel_H_for_MMSE_SIC)
        W_h_MMSE_SIC__H__h = np.einsum('aij->aji', np.conj(W_h_MMSE_SIC__H))
        norm_wii_h_MMSE_SIC__hii__h = np.einsum('caa,caa->ca', W_h_MMSE_SIC__H__h, W_h_MMSE_SIC__H).real
        interference_noise_i = np.einsum('cab,cba->cb', W_h_MMSE_SIC__H__h, W_h_MMSE_SIC__H).real - norm_wii_h_MMSE_SIC__hii__h + norm_wi_np0
        SINR_i = np.divide(norm_wii_h_MMSE_SIC__hii__h,interference_noise_i,out=np.zeros_like(norm_wii_h_MMSE_SIC__hii__h), where=interference_noise_i != 0)

        max_idx_MMSE_SIC_n = SINR_i.argmax(axis=1)
        x_n_hat = np.zeros((num_n, 1), dtype='complex')
        for iter_idx in range(num_n):
            x_n_hat[iter_idx] = np.einsum('b,bc->c',W_h_MMSE_SIC[iter_idx][max_idx_MMSE_SIC_n[iter_idx],:],channel_result_for_MMSE_SIC[iter_idx])
        x_n_hat[np.where((x_n_hat.real>0)&(x_n_hat.imag>0))[0]] = QPSK_sym_arr[0]
        x_n_hat[np.where((x_n_hat.real<0)&(x_n_hat.imag>0))[0]] = QPSK_sym_arr[1]
        x_n_hat[np.where((x_n_hat.real>0)&(x_n_hat.imag<0))[0]] = QPSK_sym_arr[2]
        x_n_hat[np.where((x_n_hat.real<0)&(x_n_hat.imag<0))[0]] = QPSK_sym_arr[3]

        for iter_idx in range(num_n):
            channel_result_for_MMSE_SIC[iter_idx] = channel_result_for_MMSE_SIC[iter_idx]-channel_H_for_MMSE_SIC[iter_idx,:,max_idx_MMSE_SIC_n[iter_idx]:max_idx_MMSE_SIC_n[iter_idx]+1]*x_n_hat[iter_idx]
            channel_H_for_MMSE_SIC[iter_idx,:,max_idx_MMSE_SIC_n[iter_idx]:max_idx_MMSE_SIC_n[iter_idx]+1] = np.zeros((inp_class.Rx,1))
        x_hat_arr = np.hstack((x_hat_arr,x_n_hat))
        max_idx_MMSE_SIC_arr = np.vstack((max_idx_MMSE_SIC_arr,max_idx_MMSE_SIC_n))

    x_hat = np.zeros((num_n, inp_class.Tx, 1), dtype='complex')
    for iter_idx in range(num_n):
        x_hat[iter_idx, max_idx_MMSE_SIC_arr[:, iter_idx]] = np.transpose(x_hat_arr[iter_idx:iter_idx + 1, :])

    return x_hat
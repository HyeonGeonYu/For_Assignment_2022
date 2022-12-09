import numpy as np
def ZF_SIC(inp_class,QPSK_sym_arr):
####################ZF_SIC
    channel_H_for_ZF_SIC = np.copy(inp_class.channel_H)
    channel_result_for_ZF_SIC = np.copy(inp_class.channel_result)
    num_n = channel_H_for_ZF_SIC.shape[0]
    ###테스트코드--------------------------
    #channel_H_for_ZF_SIC = np.array([[[1,-1,2],[-3,-1,4],[2,2,-3]]])
    #channel_result_for_ZF_SIC = np.array([[[4.3],[1.5],[-5]]])
    ### 위의 2줄은 테스트 입력
    x_hat_arr = np.empty((num_n,0),dtype='complex')
    min_idx_ZF_SIC_arr = np.empty((0,num_n),dtype='uint')
    ####################################################################################################
    for iter_Tx_num in range(inp_class.Tx):
        H_h__for_ZF_SIC = np.einsum('ijk->ikj', np.conj(channel_H_for_ZF_SIC))
        H_h__H__inv_for_ZF_SIC = np.linalg.pinv(np.einsum('abc,acd->abd', H_h__for_ZF_SIC, channel_H_for_ZF_SIC))
        W_h_ZF_for_ZF_SIC = np.einsum('dab,dbc->dac', H_h__H__inv_for_ZF_SIC, H_h__for_ZF_SIC)
        SIC_test_n = np.einsum('aij->aji', np.conj(W_h_ZF_for_ZF_SIC))
        norm_wi_n = np.sqrt(np.einsum('cab,cba->cb', SIC_test_n, W_h_ZF_for_ZF_SIC).real) #test 할 때는 제곱근으로
        min_idx_ZF_SIC_n = norm_wi_n.argsort()[:,iter_Tx_num]
        x_n_hat = np.zeros((num_n,1),dtype='complex')
        for iter_idx in range(num_n):
            x_n_hat[iter_idx] = np.einsum('b,bc->c',W_h_ZF_for_ZF_SIC[iter_idx][min_idx_ZF_SIC_n[iter_idx],:],channel_result_for_ZF_SIC[iter_idx])
        #x_n_hat = np.where((x_n_hat>0)[0],1,x_n_hat)
        #x_n_hat = np.where((x_n_hat<0)[0],-1,x_n_hat)
        x_n_hat[np.where((x_n_hat.real>0)&(x_n_hat.imag>0))[0]] = QPSK_sym_arr[0]
        x_n_hat[np.where((x_n_hat.real<0)&(x_n_hat.imag>0))[0]] = QPSK_sym_arr[1]
        x_n_hat[np.where((x_n_hat.real>0)&(x_n_hat.imag<0))[0]] = QPSK_sym_arr[2]
        x_n_hat[np.where((x_n_hat.real<0)&(x_n_hat.imag<0))[0]] = QPSK_sym_arr[3]

        for iter_idx in range(num_n):
            channel_result_for_ZF_SIC[iter_idx] = channel_result_for_ZF_SIC[iter_idx]-channel_H_for_ZF_SIC[iter_idx,:,min_idx_ZF_SIC_n[iter_idx]:min_idx_ZF_SIC_n[iter_idx]+1]*x_n_hat[iter_idx]
            channel_H_for_ZF_SIC[iter_idx,:,min_idx_ZF_SIC_n[iter_idx]:min_idx_ZF_SIC_n[iter_idx]+1] = np.zeros((inp_class.Tx,1))
        x_hat_arr = np.hstack((x_hat_arr,x_n_hat))
        min_idx_ZF_SIC_arr = np.vstack((min_idx_ZF_SIC_arr,min_idx_ZF_SIC_n))


    x_hat = np.zeros_like(inp_class.modulation_result)
    x_hat_arr
    min_idx_ZF_SIC_arr
    for iter_idx in range(num_n):
        x_hat[iter_idx] = x_hat_arr[iter_idx, :][min_idx_ZF_SIC_arr[:, iter_idx]].reshape(-1,1)

    return x_hat
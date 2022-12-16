import numpy as np
def SD(inp_class,QPSK_sym_arr,QPSK_sym_perm,d):
    channel_H__for_SD = np.copy(inp_class.channel_H)
    channel_result__for_SD = np.copy(inp_class.channel_result)
    Trans_num = channel_H__for_SD.shape[0]

    Q,R = np.linalg.qr(channel_H__for_SD)
    R_inv = np.linalg.inv(R)
    Q_hermitian = np.einsum('ijk->ikj', np.conj(Q))

    x_hat = np.einsum('ijk,ikl,ilm->ijm', R_inv,Q_hermitian,channel_result__for_SD)
    idx_list = []
    for Trans_num_idx in range(Trans_num):
        for Tx_idx in np.flip(range(inp_class.Tx)):
            if (Tx_idx == (inp_class.Tx-1)):
                x_hat_k_k1 = x_hat[Trans_num_idx,Tx_idx]
                distance = R[Trans_num_idx][Tx_idx][Tx_idx]*(QPSK_sym_arr-x_hat_k_k1)
                distance_square = (distance * np.conj(distance)).real
                idx_arr = np.where(distance_square<d**2)[0]
                if idx_arr.size == 0:
                    idx_arr = np.where(distance_square == np.min(distance_square))
                    idx_list.append((idx_arr[0]))
                    distance_square = distance_square[idx_arr]
                    d = np.sqrt(d ** 2 + np.min(distance_square))
                else:
                    idx_list.append((idx_arr,))
                    distance_square = distance_square[idx_arr]
                    d = np.sqrt(d ** 2 - np.min(distance_square))

            else :
                summation_for_x_hat_k_k1 = np.zeros((1),dtype='complex')
                for idx_for_cal in range (Tx_idx+1,inp_class.Tx):
                    if (inp_class.Tx-Tx_idx-1) == 1:
                        test_arr = QPSK_sym_arr[idx_list[inp_class.Tx - 2 - Tx_idx][inp_class.Tx - 1 - idx_for_cal]]
                        summation_for_x_hat_k_k1 = summation_for_x_hat_k_k1 + R[Trans_num_idx, Tx_idx, idx_for_cal]/ R [Trans_num_idx, Tx_idx, Tx_idx] * (test_arr - x_hat[Trans_num_idx,idx_for_cal])
                    elif (inp_class.Tx-Tx_idx-1) == 2:
                        if (idx_for_cal == (Tx_idx+1)):
                            test_arr = QPSK_sym_arr[idx_list[inp_class.Tx - 2 - Tx_idx][1]]
                        elif (idx_for_cal == (Tx_idx+2)) :
                            test_arr = QPSK_sym_arr[idx_list[inp_class.Tx - 2 - Tx_idx][0]]
                        summation_for_x_hat_k_k1 = summation_for_x_hat_k_k1 + R[Trans_num_idx, Tx_idx, idx_for_cal] / R[
                            Trans_num_idx, Tx_idx, Tx_idx] * (test_arr - x_hat[Trans_num_idx, idx_for_cal])
                    elif (inp_class.Tx - Tx_idx - 1) == 3:
                        if (idx_for_cal == (Tx_idx+1)):
                            test_arr = QPSK_sym_arr[idx_list[inp_class.Tx - 2 - Tx_idx][1]]
                        elif (idx_for_cal == (Tx_idx+2)) :
                            test_arr = QPSK_sym_arr[idx_list[1][1] [idx_list[2][0]]]
                        elif (idx_for_cal == (Tx_idx+3)) :
                            test_arr = QPSK_sym_arr[idx_list[1][0] [idx_list[2][0]]]
                        summation_for_x_hat_k_k1 = summation_for_x_hat_k_k1 + R[Trans_num_idx, Tx_idx, idx_for_cal] / R[
                            Trans_num_idx, Tx_idx, Tx_idx] * (test_arr - x_hat[Trans_num_idx, idx_for_cal])
                x_hat_k_k1 = x_hat[Trans_num_idx, Tx_idx] - summation_for_x_hat_k_k1
                x_hat_k_k1 = x_hat_k_k1.reshape(-1,1,1)
                distance = R[Trans_num_idx][Tx_idx][Tx_idx] * (QPSK_sym_arr - x_hat_k_k1)
                #QPSK_sym_arr[1]-x_hat_k_k1[0] == (QPSK_sym_arr - x_hat_k_k1)[0,1]
                distance_square = (distance * np.conj(distance)).real
                idx_arr = np.where(distance_square < d ** 2)
                if idx_arr[0].size == 0 :
                    idx_arr = np.where(distance_square == np.min(distance_square))
                    idx_list.append((idx_arr[0],idx_arr[1]))
                    distance_square = distance_square[idx_arr]
                    d = np.sqrt(d ** 2 + np.min(distance_square))
                else:
                    idx_list.append((idx_arr[0],idx_arr[1]))
                    distance_square = distance_square[idx_arr]
                    d = np.sqrt(d ** 2 - np.min(distance_square))

        list1 = idx_list[3][1].reshape(-1,1) #00이 쏜거
        list2 = idx_list[2][1][idx_list[3][0]].reshape(-1,1) #11이 쏜거
        list3 = idx_list[1][1][idx_list[2][0][idx_list[3][0]]].reshape(-1,1) #22이 쏜거
        list4 = idx_list[2][1][idx_list[1][0][idx_list[2][0][idx_list[3][0]]]].reshape(-1,1) #33이 쏜거
        candidate_perm = np.hstack([list1, list2,list3,list4])
        QPSK_candidate_perm = QPSK_sym_arr[candidate_perm]
        np.array([list1,list2,list3,list4])
        QPSK_sym_arr_mapper = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        QPSK_sym_arr_mapper_perm = QPSK_sym_arr_mapper[candidate_perm]

        '''
        test = np.einsum('mn,rnd->rmd', channel_H__for_SD[Trans_num_idx], QPSK_candidate_perm)
        test2 =  channel_result__for_SD[Trans_num_idx] - test
        min_idx = np.argmin(np.sqrt(np.einsum('rmn,rmn->r', np.conj(test2), test2).real))
        x_hat[Trans_num_idx] =QPSK_candidate_perm[min_idx]
        
    return x_hat
        '''
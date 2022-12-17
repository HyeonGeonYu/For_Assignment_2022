import numpy as np
def SD(inp_class,QPSK_sym_arr,QPSK_sym_perm,d):
    channel_H__for_SD = np.copy(inp_class.channel_H)
    channel_result__for_SD = np.copy(inp_class.channel_result)
    Trans_num = channel_H__for_SD.shape[0]
    result_x = np.zeros_like(inp_class.channel_result)

    for Trans_num_idx in range(Trans_num):
        channel_H__for_SD[Trans_num_idx]


        H_Re_Im = np.vstack((np.hstack((channel_H__for_SD[Trans_num_idx].real, -channel_H__for_SD[Trans_num_idx].imag)),
                             np.hstack((channel_H__for_SD[Trans_num_idx].imag, channel_H__for_SD[Trans_num_idx].real))))

        y_Re_Im = np.vstack(
            (channel_result__for_SD[Trans_num_idx].real, channel_result__for_SD[Trans_num_idx].imag)) / np.sqrt(
            inp_class.rootpower_of_symbol ** 2 / 2)  # 보낸 sym을 +-1, +-1j 로 normalization
        x = np.zeros((2*inp_class.Tx,1))
        Q, R = np.linalg.qr(H_Re_Im)
        x_hat = np.dot(np.linalg.pinv(H_Re_Im),y_Re_Im)
        result_x_Re_Im = np.zeros_like(x_hat)
        up_bound_arr = np.zeros_like(x_hat)
        d_square_arr = np.zeros_like(x_hat)
        r_ii_x_hat_k_bar_k1_arr = np.zeros_like(x_hat)
        case_number =1

        while True:
            if case_number == 1:
                k = 2*inp_class.Tx
                d_square_arr[k-1] = d**2- np.linalg.norm(np.transpose(Q[:,2*inp_class.Tx:]))**2
                r_ii_x_hat_k_bar_k1_arr[k-1] = x_hat[inp_class.Tx*2-1]
                case_number = 2
            elif case_number == 2:
                tmp = np.array(((-np.sqrt(d_square_arr[k-1])+r_ii_x_hat_k_bar_k1_arr[k-1])/R[k-1,k-1] ,
                       (np.sqrt(d_square_arr[k-1])+r_ii_x_hat_k_bar_k1_arr[k-1])/R[k-1,k-1]))
                tmp = np.sort(tmp,axis=0)
                up_bound_arr[k-1] = 1-2*(tmp[1]<1)
                low_bound = 1-2*(tmp[0]<-1)
                x[k-1] = low_bound-1
                case_number = 3
            elif case_number ==3:
                x[k-1] = x[k-1]+1
                if x[k-1]<= up_bound_arr[k-1]:
                    case_number = 5
                else:
                    case_number = 4
            elif case_number == 4:
                k = k+1
                if k == 2*inp_class.Tx+1:
                    break
                else:
                    case_number = 3
            elif case_number == 5:
                if k == 1:
                    case_number = 6
                else:
                    k = k-1
                    d_square_arr[k-1]= d_square_arr[k] - (r_ii_x_hat_k_bar_k1_arr[k]-R[k,k] * x[k])**2
                    if d_square_arr[k-1]<0:
                        pass
                    r_ii_x_hat_k_bar_k1_arr[k-1] = x_hat[k-1] - np.dot(R[k-1,k:2*inp_class.Tx],x[k:])
                    case_number = 2
            elif case_number == 6:
                result_x_Re_Im = x
                case_number = 3
        result_x_Re_Im

        '''
        test = np.einsum('mn,rnd->rmd', channel_H__for_SD[Trans_num_idx], QPSK_candidate_perm)
        test2 =  channel_result__for_SD[Trans_num_idx] - test
        min_idx = np.argmin(np.sqrt(np.einsum('rmn,rmn->r', np.conj(test2), test2).real))
        x_hat[Trans_num_idx] =QPSK_candidate_perm[min_idx]
        
    return x_hat
        '''
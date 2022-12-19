import numpy as np
def SD(inp_class,d):
    channel_H__for_SD = np.copy(inp_class.channel_H)
    channel_result__for_SD = np.copy(inp_class.channel_result)
    Trans_num = channel_H__for_SD.shape[0]
    result_x = np.zeros_like(inp_class.channel_result)

    for Trans_num_idx in range(Trans_num):
        channel_H__for_SD[Trans_num_idx]


        H_Re_Im = np.vstack((np.hstack((channel_H__for_SD[Trans_num_idx].real, -channel_H__for_SD[Trans_num_idx].imag)),
                             np.hstack((channel_H__for_SD[Trans_num_idx].imag, channel_H__for_SD[Trans_num_idx].real))))

        y_normalized_Re_Im = np.vstack(
            (channel_result__for_SD[Trans_num_idx].real, channel_result__for_SD[Trans_num_idx].imag)) / np.sqrt(
            inp_class.rootpower_of_symbol ** 2 / 2)  # 보낸 sym을 +-1, +-1j 로 normalization
        x = np.zeros((2*inp_class.Tx,1))

        W_ZF = np.linalg.pinv(H_Re_Im)

        G = np.dot(np.transpose(H_Re_Im),H_Re_Im)
        R = np.transpose(np.linalg.cholesky(G))
        #Q, R = np.linalg.qr(H_Re_Im)

        #R__x_hat = np.dot(np.transpose( Q[:, 0:2 * inp_class.Tx] ) , y_normalized_Re_Im)
        x_hat = np.dot(W_ZF,y_normalized_Re_Im)
        result_x_Re_Im = np.zeros_like(x_hat)-1
        up_bound_arr = np.zeros_like(x_hat)
        #up_bound_arr = np.zeros_like(R__x_hat)

        d_square_arr = np.zeros_like(x_hat)
        #d_square_arr = np.zeros_like(R__x_hat)
        x_hat_k_bar_k1_arr = np.zeros_like(x_hat)
        #r_ii_x_hat_k_bar_k1_arr = np.zeros_like(R__x_hat)
        case_number =1
        result_x_Re_Im = np.copy(x_hat)
        x_list = []
        distance_list = []
        while (True):
            if case_number == 1:
                k = 2*inp_class.Tx
                if inp_class.Rx>inp_class.Tx:
                    d_square_arr[k-1] = d**2 - np.linalg.norm(np.dot(np.linalg.pinv(np.transpose(R))[:,2*inp_class.Tx:],
                                                                     y_normalized_Re_Im))**2
                else:
                    d_square_arr[k - 1] = d ** 2
                    max_distance = -np.inf
                x_hat_k_bar_k1_arr[k-1] = x_hat[inp_class.Tx*2-1]
                case_number = 2
            elif case_number == 2:
                if d_square_arr[k-1] <0:
                    tmp = np.array([np.nan,np.nan])
                else:
                    tmp = np.array((-np.sqrt(d_square_arr[k-1])/R[k-1,k-1]+x_hat_k_bar_k1_arr[k-1] ,
                           np.sqrt(d_square_arr[k-1])/R[k-1,k-1]+x_hat_k_bar_k1_arr[k-1]))
                up_bound_arr[k-1] = 1-2*(tmp[1]<1)
                low_bound = (1-2*(tmp[0]<-1))-2
                x[k-1] = low_bound
                case_number = 3
            elif case_number ==3:
                x[k-1] = x[k-1]+2
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
                    d_square_arr[k-1]= d_square_arr[k] - (R[k,k] * (x[k]-x_hat_k_bar_k1_arr[k]))**2

                    x_hat_k_bar_k1_arr[k-1] = x_hat[k-1] - 1/R[k-1,k-1] * np.dot(R[k-1,k:],x[k:]-x_hat[k:])

                    case_number = 2
            elif case_number == 6:
            ###아래 5줄 디버깅 해야함.
                tmp_distance = (d_square_arr[0][0] - (R[k-1, k-1] * (x[k-1] - x_hat_k_bar_k1_arr[k-1])) ** 2)[0]
                if (tmp_distance)>=(max_distance):
                    max_distance = tmp_distance
                    result_x_Re_Im = np.copy(x)
                    x_list.append(np.copy(result_x_Re_Im))
                    distance_list.append(max_distance)
                case_number = 3

        test_1 = np.where(x_hat>0,1,-1)
        R_inv_distance = 9-np.linalg.norm(np.dot(R,(test_1-x_hat)))**2
        result_distance = 9-np.linalg.norm(np.dot(R,(result_x_Re_Im-x_hat)))**2
        result_x[Trans_num_idx] = result_x_Re_Im[0:inp_class.Tx,:]+1j*result_x_Re_Im[inp_class.Tx:,:]
    return result_x
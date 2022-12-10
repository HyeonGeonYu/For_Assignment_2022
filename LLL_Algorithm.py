import numpy as np

def LLL_Algorithm(channel_H__for_LR_ZF,Tx):
    Q, R = np.linalg.qr(channel_H__for_LR_ZF)
    ld = 3/4
    mm = Tx
    T = np.eye(mm,dtype='complex')
    S_flag = 0
    iter_idx = 0
    iter_max = 5
    while (S_flag == 0 )& (iter_idx < iter_max):
        S_flag = 1
        iter_idx = iter_idx + 1

        for k in range(2,mm+1):
            for l in range(k-1,0, -1):
                mu = round(R[l-1, k-1] / R[l-1, l-1]) #실수 허수 따로 round해야함.
                if abs(mu) != 0: #복소수의 norm2 크기로 하면됨.
                    R[0: l, k-1] = R[0: l, k-1] - mu * R[0: l, l-1]
                    T[:, k-1]   = T[:, k-1]   - mu * T[:, l-1]
            temp_div = np.linalg.norm(R[k-2:k,k-1])
            #Siegel Condition
            if ld*abs(R[k - 2, k - 2]) > abs(R[k-1, k-1]):  #복소수의 norm2 크기로 하면됨.

                #exchange columns k and k-1
                temp_r = np.copy(R[:, k - 2])
                temp_t = np.copy(T[:, k - 2])
                R[:, k - 2] = np.copy(R[:, k-1])
                R[:, k-1] = np.copy(temp_r)
                T[:, k - 2] = np.copy(T[:, k-1])
                T[:, k-1] = np.copy(temp_t)

                # Given's rotation
                alpha = R[k - 2, k - 2] / temp_div
                beta = R[k-1, k - 2] / temp_div
                Theta = [[np.transpose(np.conj(alpha)), beta],[ -beta, alpha]]

                R[k - 2:k, k - 2:] = np.dot(Theta,R[k-2:k, k-2:])
                S_flag = 0
            k = k + 1
    return T

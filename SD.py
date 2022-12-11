import numpy as np
def SD(inp_class,QPSK_sym_arr,QPSK_sym_perm,d):
    channel_H__for_SD = np.copy(inp_class.channel_H)
    channel_result__for_SD = np.copy(inp_class.channel_result)
    Trans_num = channel_H__for_SD.shape[0]

    Q,R = np.linalg.qr(channel_H__for_SD)

    R[0][3][3]
    channel_result__for_SD[0]
    np.dot(np.transpose(Q[0]), channel_result__for_SD[0])
    R__x_hat= np.dot(np.transpose(Q[0]), channel_result__for_SD[0])

    x_4_5 =(R__x_hat[3])/R[0][3][3]
    distance_4 = R[0][3][3] * (QPSK_sym_arr - x_4_5)
    x_4_arr = QPSK_sym_arr[np.where(np.einsum('ai,ai->a', np.conj(distance_4), distance_4).real < d**2)]

    tmp_for_d =R[0][3][3]*(x_4_arr-x_4_5)
    d = np.sqrt(np.max(d**2-(tmp_for_d.real**2+tmp_for_d.imag**2)))

    x_3_4 = (R__x_hat[2] - R[0][2][3]*x_4_arr)/R[0][2][2]
    distance_3 = R[0][2][2] * (QPSK_sym_arr.reshape(4, 1, 1) - x_3_4)
    a = distance_3.real**2+distance_3.imag**2
    search_arr = a<(d**2)
    x_3_arr1 = QPSK_sym_arr[np.where(search_arr[0])]
    x_3_arr2 = QPSK_sym_arr[np.where(search_arr[1])]
    x_3_arr3 = QPSK_sym_arr[np.where(search_arr[2])]
    x_3_arr4 = QPSK_sym_arr[np.where(search_arr[3])]

    aa = np.max((x_3_arr1 - x_3_4[0]).real**2+(x_3_arr1 - x_3_4[0]).imag**2,initial = 0)
    bb = np.max((x_3_arr2 - x_3_4[1]).real**2+(x_3_arr2 - x_3_4[1]).imag**2,initial = 0)
    cc = np.max((x_3_arr3 - x_3_4[2]).real**2+(x_3_arr3 - x_3_4[2]).imag**2,initial = 0)
    dd = np.max((x_3_arr4 - x_3_4[3]).real**2+(x_3_arr4 - x_3_4[3]).imag**2,initial = 0)
    d= max(aa,bb,cc,dd)

    x_hat = None
    return x_hat

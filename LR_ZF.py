import numpy as np
from LLL_Algorithm import LLL_Algorithm
def LR_ZF(inp_class,QPSK_sym_arr,QPSK_sym_perm):
    channel_H__for_LR_ZF = np.copy(inp_class.channel_H)
    channel_result__for_LR_ZF = np.copy(inp_class.channel_result)
    Trans_num = channel_H__for_LR_ZF.shape[0]
    T = np.zeros_like(channel_H__for_LR_ZF)
    for i in range(Trans_num):
        T[i] = LLL_Algorithm(channel_H__for_LR_ZF[i],inp_class.Tx)
    T_inv = np.linalg.inv(T)
    H_tilda = np.einsum('abc,acd->abd', channel_H__for_LR_ZF,T)

    np.dot(H_tilda[0],np.linalg.inv(T[0]))
    H_tilda_h= np.einsum('ijk->ikj', np.conj(H_tilda))
    H_tilda_h__H_tilda__inv = np.linalg.inv(np.einsum('abc,acd->abd', H_tilda_h, H_tilda))
    W_tilda_h_ZF = np.einsum('abc,acd->abd', H_tilda_h__H_tilda__inv,H_tilda_h)

    z_hat = np.einsum('abc,acd->abd', W_tilda_h_ZF, channel_result__for_LR_ZF)
    tol = 1e-10
    for idx_Trans in range(Trans_num):
        for idx_Rx in range(inp_class.Rx):
            #np.nonzero(T_inv[idx_Trans][idx_Rx:idx_Rx + 1, :])[1]
            #QPSK_sym_arr

            lattice_z = np.einsum('ab,ibc->iac', T_inv[idx_Trans][idx_Rx:idx_Rx+1, :], QPSK_sym_perm)
            #lattice_z.real[abs(lattice_z.real) < tol] = 0
            #lattice_z.imag[abs(lattice_z.imag) < tol] = 0
            u1,idx1 = np.unique(lattice_z,return_index=True)
            #z_hat[idx_Trans,idx_Rx] = lattice_z[abs(lattice_z-z_hat[idx_Trans,idx_Rx,0]).argmin()]
            #z_hat[idx_Trans,idx_Rx] = lattice_z[np.linalg.norm(lattice_z-z_hat[idx_Trans,idx_Rx,0],axis=1).argmin()]
            distance = z_hat[idx_Trans, idx_Rx] - u1
            z_hat[idx_Trans,idx_Rx] = lattice_z[idx1[(np.einsum('a,a->a',np.conj(distance),distance).real).argmin()]]


    x_hat = np.einsum('abc,acd->abd', T, z_hat)
    return x_hat

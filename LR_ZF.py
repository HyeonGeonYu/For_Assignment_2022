import numpy as np
from LLL_Algorithm import LLL_Algorithm
import cupy as cp
def LR_ZF(inp_class,QPSK_sym_arr,QPSK_sym_perm):
    channel_H__for_LR_ZF = cp.copy(inp_class.channel_H)
    channel_result__for_LR_ZF = cp.copy(inp_class.channel_result)
    Trans_num = channel_H__for_LR_ZF.shape[0]
    T = cp.zeros_like(channel_H__for_LR_ZF)
    T = cp.zeros((Trans_num,inp_class.Tx,inp_class.Tx),dtype='complex')
    for i in range(Trans_num):
        T[i] = LLL_Algorithm(channel_H__for_LR_ZF[i],inp_class.Tx)
    T_inv = cp.linalg.inv(T)
    H_tilda = cp.einsum('abc,acd->abd', channel_H__for_LR_ZF,T)

    H_tilda_h= cp.einsum('ijk->ikj', cp.conj(H_tilda))
    H_tilda_h__H_tilda__inv = cp.linalg.inv(cp.einsum('abc,acd->abd', H_tilda_h, H_tilda))
    W_tilda_h_ZF = cp.einsum('abc,acd->abd', H_tilda_h__H_tilda__inv,H_tilda_h)

    z_hat = cp.einsum('abc,acd->abd', W_tilda_h_ZF, channel_result__for_LR_ZF)
    for idx_Trans in range(Trans_num):
        for idx_Tx in range(inp_class.Tx):
            lattice_z = cp.einsum('ab,ibc->iac', T_inv[idx_Trans][idx_Tx:idx_Tx+1, :], QPSK_sym_perm)
            u1,idx1 = cp.unique(lattice_z,return_index=True)
            z_hat[idx_Trans,idx_Tx] = lattice_z[abs(lattice_z-z_hat[idx_Trans,idx_Tx,0]).argmin()]
            z_hat[idx_Trans,idx_Tx] = lattice_z[cp.linalg.norm(lattice_z-z_hat[idx_Trans,idx_Tx,0],axis=1).argmin()]

            distance = z_hat[idx_Trans, idx_Tx] - u1
            z_hat[idx_Trans, idx_Tx] = lattice_z[idx1[(cp.einsum('a,a->a', cp.conj(distance), distance).real).argmin()]]


    x_hat = cp.einsum('abc,acd->abd', T, z_hat)
    return x_hat

import numpy as np
def ZF(inp_class):
    channel_H__for_ZF = np.copy(inp_class.channel_H)
    channel_result__for_ZF = np.copy(inp_class.channel_result)

    channel_H_hermitian = np.einsum('ijk->ikj', np.conj(channel_H__for_ZF))
    H_h__H__inv = np.linalg.inv(np.einsum('abc,acd->abd', channel_H_hermitian, inp_class.channel_H))
    W_h_ZF = np.einsum('abc,acd->abd', H_h__H__inv, channel_H_hermitian)
    x_hat = np.einsum('abc,acd->abd', W_h_ZF, channel_result__for_ZF)

    return x_hat
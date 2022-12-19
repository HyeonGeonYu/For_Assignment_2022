import numpy as np
def MMSE(inp_class):
    channel_H__for_MMSE = np.copy(inp_class.channel_H)
    channel_result__for_MMSE = np.copy(inp_class.channel_result)

    channel_H_hermitian = np.einsum('ijk->ikj', np.conj(channel_H__for_MMSE))
    H_h__H = np.einsum('abc,acd->abd', channel_H_hermitian, channel_H__for_MMSE)
    H_h__H__N0PNt__inv = np.linalg.inv(
        H_h__H + inp_class.N0 / (inp_class.rootpower_of_symbol ** 2) * np.eye(inp_class.Tx))
    W_h_MMSE = np.einsum('abc,acd->abd', H_h__H__N0PNt__inv, channel_H_hermitian)

    x_hat = np.einsum('abc,acd->abd', W_h_MMSE, channel_result__for_MMSE)

    return x_hat
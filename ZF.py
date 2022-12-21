import numpy as np
import cupy as cp
def ZF(inp_class):
    channel_H__for_ZF = cp.copy(inp_class.channel_H)
    channel_result__for_ZF = cp.copy(inp_class.channel_result)

    channel_H_hermitian = cp.einsum('ijk->ikj', cp.conj(channel_H__for_ZF))
    H_h__H__inv = cp.linalg.inv(cp.einsum('abc,acd->abd', channel_H_hermitian, inp_class.channel_H))
    W_h_ZF = cp.einsum('abc,acd->abd', H_h__H__inv, channel_H_hermitian)
    x_hat = cp.einsum('abc,acd->abd', W_h_ZF, channel_result__for_ZF)

    return x_hat
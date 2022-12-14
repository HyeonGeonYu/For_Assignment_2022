import numpy as np
import cupy as cp
import cvxpy as cvx
def SDR(inp_class):
    '''
    cvxpy 는 cupy 가속이 되지 않는다.
    '''
    channel_H__for_SDR = cp.asnumpy(inp_class.channel_H)
    channel_result__for_SDR = cp.asnumpy(inp_class.channel_result)
    rootpower_of_symbol__for_SDR = cp.asnumpy(inp_class.rootpower_of_symbol)
    Trans_num = channel_H__for_SDR.shape[0]

    x_hat = np.zeros((Trans_num,inp_class.Tx,1),dtype='complex')
    for Trans_num_idx in range(Trans_num):
        y_Re_Im = np.vstack((channel_result__for_SDR[Trans_num_idx].real,channel_result__for_SDR[Trans_num_idx].imag))\
                  /np.sqrt(rootpower_of_symbol__for_SDR**2/2) # 보낸 sym을 +-1, +-1j 로 normalization
        H_Re_Im = np.vstack(( np.hstack( (channel_H__for_SDR[Trans_num_idx].real,
                                          -channel_H__for_SDR[Trans_num_idx].imag) ),
                   np.hstack( (channel_H__for_SDR[Trans_num_idx].imag, channel_H__for_SDR[Trans_num_idx].real) ) ))
        L =np.vstack( (np.hstack((np.dot(np.transpose(H_Re_Im), H_Re_Im), np.dot(-np.transpose(H_Re_Im), y_Re_Im))) ,
                       np.hstack((np.dot(-np.transpose(y_Re_Im), H_Re_Im), np.dot(np.transpose(y_Re_Im), y_Re_Im))) ))

        X = cvx.Variable((inp_class.Tx * 2+1,inp_class.Tx * 2+1))
        objective_func = cvx.Minimize(cvx.trace(L @ X))
        constaints = [cvx.diag(X)==1,cvx.constraints.psd.PSD(X)]

        prob = cvx.Problem(objective_func,constaints)
        prob.solve()
        val,vec = np.linalg.eig(X.value)
        max_eig = vec[:,0]
        if max_eig[-1] < 0:
            max_eig = -max_eig
        real_arr = max_eig[0:inp_class.Tx]
        imag_arr = 1j*max_eig[inp_class.Tx:2*inp_class.Tx]

        x_hat[Trans_num_idx] = (real_arr +imag_arr).reshape(inp_class.Tx,1)
    x_hat = cp.asarray(x_hat)
    return x_hat

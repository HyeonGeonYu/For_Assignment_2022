import numpy as np
from itertools import product
class communicationsystem:
    def __init__(self, ext,inp_data,mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                 source_coding_type,channel_coding_type,inp_bit_len = None, draw_huffmantree = False,
                 modulation_scheme = None,fading_scheme=None, Tx = None, Rx = None,
                 mu = 0, SNR =0):
        
        self.ext = ext                                                      # 입력 파일 타입. (txt,)
        self.inp_data =inp_data                                             # 입력 데이터. (781600...)
        self.inp_data_unique_arr = inp_data_unique_arr                      # 입력 데이터의 unique arr. (01678...)
        self.inp_data_unique_arr_idx_arr = inp_data_unique_arr_idx_arr      # 입력 데이터의 unique arr의 idx. (01234...), mapped_data와 관련있음.
        self.mapped_data =mapped_data                                       # 입력 데이터를 순서대로 매핑한 arr. (341200...)
        self.count = count                                                  # 각 매핑데이터의 빈도 arr. (21111...)
        self.source_coding_type = source_coding_type                        # 소스코딩 타입. (Huffman,)
        self.channel_coding_type = channel_coding_type                      # 채널코딩 타입. (Repetition,)
        self.inp_bit_len = inp_bit_len                                      # 입력 비트 길이, txt면 inp_data_unique_arr로 결정, png면 8로 고정.
        self.draw_huffmantree = draw_huffmantree                            # tree 결과 그릴지 여부.
        self.mu = mu                                                        # 가우시안 분포의 평균.
        self.SNR = SNR                                                      # SNR dB
        self.rootpower_of_symbol = np.sqrt((10**(SNR/10))/(Tx))             # SNR로 계산한 symbol 한개 power의 root
        self.N0 = 1                                                         # N0
        self.modulation_scheme = modulation_scheme                          # 모듈레이션 타입. (BPSK,)
        self.fading_scheme = fading_scheme                                  # 페이딩 타입. (Rayleight,)
        self.Tx = Tx                                                        # 송신 안테나 갯수. (4)
        self.Rx = Rx                                                        # 수신 안테나 갯수. (4)

        self.mapped_data_bit_num = None                                 # mapped_data가 가진 bit 총 갯수, 소스코딩에따라 달라짐.
        self.code_arr = None                                            # mapped_data 각각이 가진 bit code,inp_data_unique_arr_idx_arr 와 순서 동일, 길이가 다를 경우 2가 포함되어있음.
        self.source_coding_result_np = None                             # 소스코딩 결과.
        self.source_coding_result_bit_num = None                        # 소스코딩 결과의 비트수, 2가 제외되어 계산되어있음.
        self.channel_coding_result_np = None                            # 채널코딩 결과.
        self.channel_coding_result_bit_num = None                       # 채널코딩 결과의 비트수, source_coding_result_bit_num로 계산함.
        self.modulation_result = None                                   # 모듈레이션 결과 2는 nan으로 바뀜.
        self.channel_result = None                                      # 채널겪고난 후 결과
        self.demodulation_result1 = None                                # 필터없음.
        self.demodulation_result2 = None                                # ML.
        self.demodulation_result3 = None                                # ZF.
        self.demodulation_result4 = None                                # MMSE
        self.demodulation_result5 = None                                # ZF_SIC
        self.channel_H = None                                           # 채널 H
        self.noise_N = None                                             # 노이즈 N
        self.channel_decoding_result_np = None                          # 채널 디코딩 결과.
        self.source_decoding_result_np = None                           # 소스 디코딩 결과.
        self.out_data = None                                            # 입력 데이터형태로 변경된 결과물.

def modulation(inp_class):
    if inp_class.modulation_scheme == "QPSK":
        refer_arr  = inp_class.channel_coding_result_np.reshape(-1,2)
        inp_class.modulation_result = np.zeros(refer_arr.shape[0],dtype='complex')

        inp_class.modulation_result = np.where((refer_arr == [0, 0]).all(axis=1),inp_class.rootpower_of_symbol*((1+1j)/np.sqrt(2)),inp_class.modulation_result)
        inp_class.modulation_result = np.where((refer_arr == [1, 0]).all(axis=1),inp_class.rootpower_of_symbol*((-1+1j)/np.sqrt(2)),inp_class.modulation_result)
        inp_class.modulation_result = np.where((refer_arr == [0, 1]).all(axis=1),inp_class.rootpower_of_symbol*((1-1j)/np.sqrt(2)),inp_class.modulation_result)
        inp_class.modulation_result = np.where((refer_arr == [1, 1]).all(axis=1),inp_class.rootpower_of_symbol*((-1-1j)/np.sqrt(2)),inp_class.modulation_result)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_awgn(inp_class):
    mod_size = inp_class.modulation_result.size #symbol 갯수
    Trans_num = int(mod_size/inp_class.Tx)
    channel_shape = (Trans_num,inp_class.Rx,inp_class.Tx) # [Rx x Tx x (symbol/Tx)]
    inp_class.channel_result = np.zeros((Trans_num,inp_class.Rx,1),dtype='complex')
    reshape_mod_result = inp_class.modulation_result.reshape(Trans_num,inp_class.Tx,1)

    if (inp_class.Tx>1) & (inp_class.Rx>1) :
        antenna_scheme = "MIMO"
    if inp_class.modulation_scheme == "QPSK":
        if (inp_class.fading_scheme == "Rayleigh") & (antenna_scheme == "MIMO"):
            inp_class.channel_H = 1 / np.sqrt(2) * np.random.normal(0, 1, (mod_size*inp_class.Rx, 2)).view(np.complex).reshape(channel_shape)
            inp_class.noise_N = 1/np.sqrt(2)*np.random.normal(inp_class.mu, 1, (mod_size, 2)).view(np.complex).reshape(Trans_num,inp_class.Tx,1)
            for i in range(Trans_num):
                inp_class.channel_result[i] = np.dot(inp_class.channel_H[i],reshape_mod_result[i]) + inp_class.noise_N[i]
def demodulation(inp_class):
    if inp_class.modulation_scheme == "QPSK":
        #필터 없는거
        inp_class.demodulation_result1 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1,2)
        real_arr = inp_class.channel_result.reshape(-1,1).real
        imag_arr = inp_class.channel_result.reshape(-1,1).imag
        inp_class.demodulation_result1[np.where((real_arr > 0) & (imag_arr > 0))[0]] = np.array([0,0])
        inp_class.demodulation_result1[np.where((real_arr < 0) & (imag_arr > 0))[0]] = np.array([1,0])
        inp_class.demodulation_result1[np.where((real_arr > 0) & (imag_arr < 0))[0]] = np.array([0,1])
        inp_class.demodulation_result1[np.where((real_arr < 0) & (imag_arr < 0))[0]] = np.array([1,1])
        inp_class.demodulation_result1 = inp_class.demodulation_result1.reshape(inp_class.channel_coding_result_np.shape)

        ####################ML
        inp_class.demodulation_result2 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1, 2)
        QPSK_sym_arr = inp_class.rootpower_of_symbol*np.array([[(1+1j)/np.sqrt(2)],[(-1+1j)/np.sqrt(2)],[(1-1j)/np.sqrt(2)],[(-1-1j)/np.sqrt(2)]],dtype='complex')
        QPSK_sym_arr_mapper = np.array([[0,0],[1,0],[0,1],[1,1]])
        QPSK_sym_perm = np.array([p for p in product(QPSK_sym_arr, repeat=inp_class.Tx)],dtype='complex')
        QPSK_sym_perm = QPSK_sym_perm.reshape(-1,inp_class.Tx,1)
        QPSK_sym_perm_mapper = np.array([p for p in product(QPSK_sym_arr_mapper, repeat=inp_class.Tx)],dtype='int32')
        QPSK_sym_perm_mapper = QPSK_sym_perm_mapper.reshape(-1,inp_class.Tx,2)

        i_num = inp_class.channel_H.shape[0]
        k_num = QPSK_sym_perm.shape[0]

        #import time
        #a = time.time()
        #### 현재 100000개 0.4초걸림
        #np.einsum('rmn,rnd->rmd', inp_class.channel_H, QPSK_sym_perm)
        #hannel_result_shape = inp_class.channel_result.shape
        #repeat_shape = (channel_result_shape[0],QPSK_sym_perm_mapper.shape[0],channel_result_shape[1],channel_result_shape[2])
        #repeat_arr = np.repeat(inp_class.channel_result, QPSK_sym_perm_mapper.shape[0], axis=0).reshape(repeat_shape)
        #test_ = np.einsum('amn,bnd->abmd', inp_class.channel_H, QPSK_sym_perm)
        #test_2 = repeat_arr-test_
        #test_3 = np.argmin(np.einsum('rmna,rmna->rm', np.conj(test_2), test_2).real,axis=1)
        # test_4 = QPSK_sym_perm_mapper[test_3].reshape(-1,2)
        for i in range(i_num) :
            test = np.einsum('mn,rnd->rmd', inp_class.channel_H[i], QPSK_sym_perm)
            test2 = inp_class.channel_result[i] - test
            min_idx = np.argmin(np.sqrt(np.einsum('rmn,rmn->r', np.conj(test2), test2).real))
            inp_class.demodulation_result2[i*inp_class.Tx:i*inp_class.Tx+inp_class.Tx] = QPSK_sym_perm_mapper[min_idx]
        #print("걸린시간 : ", time.time() - a)
        ######

        ####################ZF
        inp_class.demodulation_result3 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1, 2)
        channel_H_hermitian = np.einsum('ijk->ikj', np.conj(inp_class.channel_H))
        H_h__H__inv = np.linalg.inv(np.einsum('abc,acd->abd', channel_H_hermitian,inp_class.channel_H))
        W_h_ZF = np.einsum('abc,acd->abd', H_h__H__inv,channel_H_hermitian)

        x_hat = np.einsum('abc,acd->abd', W_h_ZF,inp_class.channel_result)
        real_arr = x_hat.reshape(-1, 1).real
        imag_arr = x_hat.reshape(-1, 1).imag
        inp_class.demodulation_result3[np.where((real_arr > 0) & (imag_arr > 0))[0]] = np.array([0, 0])
        inp_class.demodulation_result3[np.where((real_arr < 0) & (imag_arr > 0))[0]] = np.array([1, 0])
        inp_class.demodulation_result3[np.where((real_arr > 0) & (imag_arr < 0))[0]] = np.array([0, 1])
        inp_class.demodulation_result3[np.where((real_arr < 0) & (imag_arr < 0))[0]] = np.array([1, 1])
        inp_class.demodulation_result3 = inp_class.demodulation_result3.reshape(
            inp_class.channel_coding_result_np.shape)
        ######

        ####################MMSE
        inp_class.demodulation_result4 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1, 2)
        H_h__H = np.einsum('abc,acd->abd', channel_H_hermitian, inp_class.channel_H)
        H_h__H__N0PNt__inv = np.linalg.inv(H_h__H + inp_class.N0/(inp_class.rootpower_of_symbol**2)*np.eye(inp_class.Tx))
        W_h_MMSE = np.einsum('abc,acd->abd', H_h__H__N0PNt__inv, channel_H_hermitian)
        x_hat = np.einsum('abc,acd->abd', W_h_MMSE, inp_class.channel_result)
        real_arr = x_hat.reshape(-1, 1).real
        imag_arr = x_hat.reshape(-1, 1).imag
        inp_class.demodulation_result4[np.where((real_arr > 0) & (imag_arr > 0))[0]] = np.array([0, 0])
        inp_class.demodulation_result4[np.where((real_arr < 0) & (imag_arr > 0))[0]] = np.array([1, 0])
        inp_class.demodulation_result4[np.where((real_arr > 0) & (imag_arr < 0))[0]] = np.array([0, 1])
        inp_class.demodulation_result4[np.where((real_arr < 0) & (imag_arr < 0))[0]] = np.array([1, 1])
        inp_class.demodulation_result4 = inp_class.demodulation_result4.reshape(
            inp_class.channel_coding_result_np.shape)
        #####

        ####################ZF_SIC
        inp_class.demodulation_result5 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1, 2)
        channel_H_for_ZF_SIC = np.copy(inp_class.channel_H)
        ###테스트코드--------------------------
        H_h__for_ZF_SIC = np.einsum('ijk->ikj', np.conj(channel_H_for_ZF_SIC))
        H_h__H__inv_for_ZF_SIC = np.linalg.pinv(np.einsum('abc,acd->abd', H_h__for_ZF_SIC, channel_H_for_ZF_SIC))
        W_h_ZF = np.einsum('dab,dbc->dac', H_h__H__inv_for_ZF_SIC, H_h__for_ZF_SIC)
        ###테스트코드--------------------------

        SIC_test1 = np.einsum('ij->ji', np.conj(W_h_ZF[0]))
        norm_wi1 = np.einsum('ab,ba->b',SIC_test1, W_h_ZF[0]).real
        min_idx_ZF_SIC1 = np.argmin(norm_wi1)
        x1_hat = np.einsum('ij,ji->i',W_h_ZF[0,min_idx_ZF_SIC1:min_idx_ZF_SIC1+1,:],inp_class.channel_result[0])
        if (x1_hat.real>0) & (x1_hat.imag>0) :
            x1_hat = QPSK_sym_arr[0]
        elif (x1_hat.real<0) & (x1_hat.imag>0) :
            x1_hat = QPSK_sym_arr[1]
        elif (x1_hat.real>0) & (x1_hat.imag<0) :
            x1_hat = QPSK_sym_arr[2]
        elif (x1_hat.real<0) & (x1_hat.imag<0) :
            x1_hat = QPSK_sym_arr[3]
        y1 = inp_class.channel_result[0] - channel_H_for_ZF_SIC[0,:,min_idx_ZF_SIC1:min_idx_ZF_SIC1+1]*x1_hat
        ####################y1

        channel_H_for_ZF_SIC[0, :, min_idx_ZF_SIC1:min_idx_ZF_SIC1 + 1] =np.zeros((channel_H_for_ZF_SIC[0, :, min_idx_ZF_SIC1:min_idx_ZF_SIC1 + 1]).shape)
        H_h__for_ZF_SIC = np.transpose(np.conj(channel_H_for_ZF_SIC[0]))
        H_h__H__inv_for_ZF_SIC = np.linalg.pinv(np.einsum('bc,cd->bd',H_h__for_ZF_SIC,channel_H_for_ZF_SIC[0]))
        W_h_ZF_SIC2 = np.einsum('ab,bc->ac',H_h__H__inv_for_ZF_SIC,H_h__for_ZF_SIC)

        SIC_test2 = np.einsum('ij->ji', np.conj(W_h_ZF_SIC2))
        norm_wi2 = np.einsum('ab,ba->b', SIC_test2, W_h_ZF_SIC2).real
        norm_wi2[min_idx_ZF_SIC1] = 0
        min_idx_ZF_SIC2 = norm_wi2.argsort()[1]
        x2_hat = np.einsum('ij,ji->i', W_h_ZF_SIC2[min_idx_ZF_SIC2:min_idx_ZF_SIC2 + 1, :], y1)

        if (x2_hat.real>0) & (x2_hat.imag>0) :
            x2_hat = QPSK_sym_arr[0]
        elif (x2_hat.real<0) & (x2_hat.imag>0) :
            x2_hat = QPSK_sym_arr[1]
        elif (x2_hat.real>0) & (x2_hat.imag<0) :
            x2_hat = QPSK_sym_arr[2]
        elif (x2_hat.real<0) & (x2_hat.imag<0) :
            x2_hat = QPSK_sym_arr[3]
        y2 = y1 - channel_H_for_ZF_SIC[0, :, min_idx_ZF_SIC2:min_idx_ZF_SIC2 + 1] * x2_hat
        ####################y2

        channel_H_for_ZF_SIC[0, :, min_idx_ZF_SIC2:min_idx_ZF_SIC2 + 1] = np.zeros(
            (channel_H_for_ZF_SIC[0, :, min_idx_ZF_SIC2:min_idx_ZF_SIC2 + 1]).shape)
        H_h__for_ZF_SIC = np.transpose(np.conj(channel_H_for_ZF_SIC[0]))
        H_h__H__inv_for_ZF_SIC = np.linalg.pinv(np.einsum('bc,cd->bd', H_h__for_ZF_SIC, channel_H_for_ZF_SIC[0]))
        W_h_ZF_SIC3 = np.einsum('ab,bc->ac', H_h__H__inv_for_ZF_SIC, H_h__for_ZF_SIC)

        SIC_test3 = np.einsum('ij->ji', np.conj(W_h_ZF_SIC3))
        norm_wi3 = np.einsum('ab,ba->b', SIC_test3, W_h_ZF_SIC3).real
        norm_wi3[[min_idx_ZF_SIC1,min_idx_ZF_SIC2]] = 0
        min_idx_ZF_SIC3 = norm_wi2.argsort()[2]
        x3_hat = np.einsum('ij,ji->i', W_h_ZF_SIC3[min_idx_ZF_SIC3:min_idx_ZF_SIC3 + 1, :], y2)

        if (x3_hat.real > 0) & (x3_hat.imag > 0):
            x3_hat = QPSK_sym_arr[0]
        elif (x3_hat.real < 0) & (x3_hat.imag > 0):
            x3_hat = QPSK_sym_arr[1]
        elif (x3_hat.real > 0) & (x3_hat.imag < 0):
            x3_hat = QPSK_sym_arr[2]
        elif (x3_hat.real < 0) & (x3_hat.imag < 0):
            x3_hat = QPSK_sym_arr[3]
        y3 = y2 - channel_H_for_ZF_SIC[0, :, min_idx_ZF_SIC3:min_idx_ZF_SIC3 + 1] * x3_hat
        ####################y3

    else:
        raise Exception('모듈레이션 scheme 확인필요')

def make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,modulation_scheme,fading_scheme,Tx,Rx,mu,SNR):
    inp_data, mapped_data, inp_data_unique_arr,inp_data_unique_arr_idx_arr, count, inp_bit_len, ext = None, None, None, None, None, None, None

    inp_class = communicationsystem(ext, inp_data, mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                                    source_coding_type,channel_coding_type, inp_bit_len,draw_huffmantree,
                                    modulation_scheme,fading_scheme, Tx, Rx,
                                    mu,SNR)

    inp_class.channel_coding_result_np = np.random.randint(0,2,(1000,90)) #0과1 랜덤하게 1600개 생성
    modulation(inp_class)

    channel_awgn(inp_class)
    demodulation(inp_class)
    return inp_class

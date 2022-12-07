import numpy as np

class communicationsystem:
    def __init__(self, ext,inp_data,mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                 source_coding_type,channel_coding_type,inp_bit_len = None, draw_huffmantree = False,
                 modulation_scheme = None,fading_scheme=None, Tx = None, Rx = None,
                 mu = 0, std =1):
        
        self.ext = ext                                                  # 입력 파일 타입. (txt,)
        self.inp_data =inp_data                                         # 입력 데이터. (781600...)
        self.inp_data_unique_arr = inp_data_unique_arr                  # 입력 데이터의 unique arr. (01678...)
        self.inp_data_unique_arr_idx_arr = inp_data_unique_arr_idx_arr  # 입력 데이터의 unique arr의 idx. (01234...), mapped_data와 관련있음.
        self.mapped_data =mapped_data                                   # 입력 데이터를 순서대로 매핑한 arr. (341200...)
        self.count = count                                              # 각 매핑데이터의 빈도 arr. (21111...)
        self.source_coding_type = source_coding_type                    # 소스코딩 타입. (Huffman,)
        self.channel_coding_type = channel_coding_type                  # 채널코딩 타입. (Repetition,)
        self.inp_bit_len = inp_bit_len                                  # 입력 비트 길이, txt면 inp_data_unique_arr로 결정, png면 8로 고정.
        self.draw_huffmantree = draw_huffmantree                        # tree 결과 그릴지 여부.
        self.mu = mu                                                    # 가우시안 분포의 평균.
        self.std = std                                                  # 가우시안 분포의 표준편차.
        self.modulation_scheme = modulation_scheme                      # 모듈레이션 타입. (BPSK,)
        self.fading_scheme = fading_scheme                              # 페이딩 타입. (Rayleight,)
        self.Tx = Tx                                                    # 송신 안테나 갯수. (4)
        self.Rx = Rx                                                    # 수신 안테나 갯수. (4)

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
        self.demodulation_result3 = None                                # 디모듈레이션 결과. nan이 다시 2로 바뀜
        self.channel_H = None                                           # 채널 H
        self.channel_decoding_result_np = None                          # 채널 디코딩 결과.
        self.source_decoding_result_np = None                           # 소스 디코딩 결과.
        self.out_data = None                                            # 입력 데이터형태로 변경된 결과물.

def modulation(inp_class):

    if inp_class.modulation_scheme == "BPSK":
        inp_class.modulation_result = np.where(inp_class.channel_coding_result_np  == 2, np.nan, inp_class.channel_coding_result_np)
        inp_class.modulation_result = np.where(inp_class.modulation_result == 0, -1, inp_class.modulation_result)
    elif inp_class.modulation_scheme == "QPSK":
        refer_arr  = inp_class.channel_coding_result_np.reshape(-1,2)
        inp_class.modulation_result =np.zeros(refer_arr.shape[0],dtype='complex')
        inp_class.modulation_result = np.where((refer_arr == [0, 0]).all(axis=1),1+1j,inp_class.modulation_result)
        inp_class.modulation_result = np.where((refer_arr == [1, 0]).all(axis=1),-1+1j,inp_class.modulation_result)
        inp_class.modulation_result = np.where((refer_arr == [0, 1]).all(axis=1),1-1j,inp_class.modulation_result)
        inp_class.modulation_result = np.where((refer_arr == [1, 1]).all(axis=1),-1-1j,inp_class.modulation_result)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_awgn(inp_class):
    mod_size = inp_class.modulation_result.size #symbol 갯수
    mod_shape = inp_class.modulation_result.shape
    Trans_num = int(mod_size/inp_class.Tx)
    channel_shape = (Trans_num,inp_class.Rx,inp_class.Tx) # [Rx x Tx x (symbol/matrix_size)]
    Tx_shape = (Trans_num,inp_class.Tx,1)
    inp_class.channel_result = np.zeros((Trans_num,inp_class.Rx,1),dtype='complex')
    reshape_mod_result = inp_class.modulation_result.reshape(Trans_num,inp_class.Tx,1)
    if (inp_class.Tx>1) & (inp_class.Rx>1) :
        antenna_scheme = "MIMO"
    if inp_class.modulation_scheme == "BPSK":
        inp_class.channel_result = inp_class.modulation_result + np.random.normal(inp_class.mu, inp_class.std, inp_class.modulation_result.shape)
    elif inp_class.modulation_scheme == "QPSK":
        if (inp_class.fading_scheme == "Rayleigh") & (antenna_scheme == "MIMO"):
            inp_class.channel_H = 1 / np.sqrt(2) * np.random.normal(0, 1, (mod_size*inp_class.Rx, 2)).view(np.complex).reshape(channel_shape)
            noise_N = np.random.normal(inp_class.mu, inp_class.std, (mod_size, 2)).view(np.complex).reshape(Trans_num,inp_class.Tx,1)
            for i in range(Trans_num):
                inp_class.channel_result[i] = np.dot(inp_class.channel_H[i],reshape_mod_result[i]) + noise_N[i]
def demodulation(inp_class):
    if inp_class.modulation_scheme == "BPSK":
        inp_class.demodulation_result = np.where(inp_class.channel_result < 0, 0,
                                               inp_class.channel_result)
        inp_class.demodulation_result = np.where(inp_class.demodulation_result  > 0, 1,
                                               inp_class.demodulation_result)
        inp_class.demodulation_result = np.where(np.isnan(inp_class.demodulation_result),2,inp_class.demodulation_result).astype('uint8')
    
    elif inp_class.modulation_scheme == "QPSK":
        #필터 없는거
        inp_class.demodulation_result1 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1,2)
        real_arr = inp_class.channel_result.reshape(-1,1).real
        imag_arr = inp_class.channel_result.reshape(-1,1).imag
        inp_class.demodulation_result1[np.where((real_arr > 0) & (imag_arr > 0))[0]] = np.array([0,0])
        inp_class.demodulation_result1[np.where((real_arr < 0) & (imag_arr > 0))[0]] = np.array([1,0])
        inp_class.demodulation_result1[np.where((real_arr > 0) & (imag_arr < 0))[0]] = np.array([0,1])
        inp_class.demodulation_result1[np.where((real_arr < 0) & (imag_arr < 0))[0]] = np.array([1,1])
        inp_class.demodulation_result1 = inp_class.demodulation_result1.reshape(inp_class.channel_coding_result_np.shape)

        #ML
        inp_class.demodulation_result2 = np.zeros_like(inp_class.channel_coding_result_np).reshape(-1, 2)
        QPSK_sym_arr = np.array([[1+1j],[-1+1j],[1-1j],[-1-1j]],dtype='complex')
        from itertools import product
        QPSK_sym_perm = np.array([p for p in product(QPSK_sym_arr, repeat=2)],dtype='complex')
        QPSK_sym_perm = QPSK_sym_perm.reshape(-1,inp_class.Rx,1)
        i_num = inp_class.channel_H.shape[0]
        k_num = QPSK_sym_perm.shape[0]

        for i in range(i_num) :
            for k in range(k_num) :
                if k ==0:
                    min_num = np.linalg.norm(inp_class.channel_result[i]-np.dot(inp_class.channel_H[i],QPSK_sym_perm[k]),2)
                    k_min_idx = k
                else:
                    norm_val = np.linalg.norm(inp_class.channel_result[i]-np.dot(inp_class.channel_H[i],QPSK_sym_perm[k]), 2)
                    if min_num> norm_val:
                        min_num = norm_val
                        k_min_idx = k
            inp_class.demodulation_result2[i*2:i*2+2] = QPSK_sym_perm[k_min_idx]

    else:
        raise Exception('모듈레이션 scheme 확인필요')

def make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,modulation_scheme,fading_scheme,Tx,Rx,mu,std):
    inp_data, mapped_data, inp_data_unique_arr,inp_data_unique_arr_idx_arr, count, inp_bit_len, ext = None, None, None, None, None, None, None

    inp_class = communicationsystem(ext, inp_data, mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                                    source_coding_type,channel_coding_type, inp_bit_len,draw_huffmantree,
                                    modulation_scheme,fading_scheme, Tx, Rx,
                                    mu,std)

    inp_class.channel_coding_result_np = np.random.randint(0,2,(100,16)) #0과1 랜덤하게 1600개 생성

    modulation(inp_class)
    channel_awgn(inp_class)
    demodulation(inp_class)

    return inp_class

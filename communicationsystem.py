import numpy as np
from read_file_func import read_file_func
from sourceencoder import huffman2
import cv2
class communicationsystem:
    def __init__(self, ext,inp_data,mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                 source_coding_type="NoCompression",inp_bit_len = None, draw_huffmantree = False,
                 modulation_scheme = None,
                 mu = 0, std =1):

        ##source코딩에 필요한 파라미터
        self.ext = ext
        self.inp_data =inp_data
        self.mapped_data =mapped_data
        self.count = count
        self.inp_data_unique_arr = inp_data_unique_arr
        self.inp_data_unique_arr_idx_arr = inp_data_unique_arr_idx_arr
        self.code_arr = None
        self.source_coding_type = source_coding_type
        self.inp_bit_len = inp_bit_len
        self.draw_huffmantree = draw_huffmantree
        self.mu = mu
        self.std = std
        self.modulation_scheme = modulation_scheme

        self.source_coding_result_np = None
        self.modulation_result = None
        self.channel_result = None
        self.demodulation_result = None
        self.source_decoding_result_np = None
        self.source_decoding_result_approx_np = None
        self.out_data = None

def source_encoder(inp_class):
    '''
    넘파이 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    ######## source_encoder
    if inp_class.source_coding_type == "Huffman":
        h = huffman2.HuffmanCoding(inp_class.mapped_data,inp_class.count,inp_class.inp_data_unique_arr,inp_class.inp_data_unique_arr_idx_arr , inp_class.draw_huffmantree)
        source_coding_result_np,code_arr = h.compress()
    elif inp_class.source_coding_type == "NoCompression":
        inp_class.mapped_data.reshape(-1,)
        source_coding_result_np = np.unpackbits(inp_class.mapped_data.reshape(-1,1).view('uint8'), axis=1, count=inp_class.inp_bit_len,bitorder='little')  # 데이터를 바이트로 나누고 비트로 변경함
        code_arr = np.unpackbits(inp_class.inp_data_unique_arr_idx_arr.reshape(-1, 1), axis=1, count=inp_class.inp_bit_len,
                      bitorder='little')
    else:
        raise Exception("압축 알고리즘 이름 확인 필요함.")

    inp_class.source_coding_result_np = source_coding_result_np
    inp_class.code_arr = code_arr
def channel_coding(bit_stream):
    '''
        구현해야함
    '''
    return bit_stream
def modulation(inp_class):
    '''
    데이터와 비트 길이, scheme을 입력에 따라 symbol을 반환
    '''
    if inp_class.modulation_scheme == "BPSK":
        inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 2, np.nan, inp_class.source_coding_result_np)
        inp_class.modulation_result = np.where(inp_class.modulation_result == 0, -1, inp_class.modulation_result)
        #inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 1, 1, inp_class.source_coding_result_np)
        #inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 1, 1, inp_class.modulation_result)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_awgn(inp_class):
    inp_class.channel_result = inp_class.modulation_result + np.random.normal(inp_class.mu, inp_class.std, inp_class.modulation_result.shape)
def demodulation(inp_class):
    if inp_class.modulation_scheme == "BPSK":
        inp_class.demodulation_result = np.where(inp_class.channel_result < 0, 0,
                                               inp_class.channel_result)
        inp_class.demodulation_result = np.where(inp_class.demodulation_result  > 0, 1,
                                               inp_class.demodulation_result)
        inp_class.demodulation_result = np.where(np.isnan(inp_class.demodulation_result),2,inp_class.demodulation_result).astype('uint8')

    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_decoding(bit_stream):
    '''
        구현해야함
    '''
    return bit_stream
def source_decoder(inp_class) :

    if inp_class.source_coding_type == "Huffman":

        inp_class.source_decoding_result_np = np.zeros_like(inp_class.mapped_data)

        u1,v1 = np.unique((inp_class.code_arr == 2).sum(axis=1),return_inverse=True) # 2의 갯수 array, code 별 2의 갯수,

        u2,v2 = np.unique((inp_class.demodulation_result == 2).sum(axis=1),return_inverse=True)

        for i in u1: #2의 갯수가 가장작은것 부터 큰것까지 순회하겠음.
            code_idx_arr = np.where(v1 == i)[0]
            code_arr_with_2i = inp_class.code_arr[code_idx_arr].astype('int8') # 2의 갯수가 i개인 코드 어레이들 뭉탱이

            demodul_result_idx_arr = np.where(v2 == i)

            for demodul_result_idx in demodul_result_idx_arr[0]  : # 2의 갯수가 i개인 디모듈 어레이들 뭉탱이
                detection_result = np.argmin(np.power(inp_class.demodulation_result[demodul_result_idx].astype('int8') - code_arr_with_2i.astype('int8'), 2).sum(axis=1)) # bool로 하면 더 빨라질듯
                #inp_class.source_decoding_result_np[demodul_result_idx] = code_arr[detection_result] # 나중에 BER등 결과그래프에서 쓰자
                inp_class.source_decoding_result_np[demodul_result_idx] = code_idx_arr[detection_result] #mapped data 결과


        if inp_class.ext == ".txt":
            inp_class.out_data = "".join(list(inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np]))

        elif inp_class.ext == ".png":
            inp_class.out_data =inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np].reshape(inp_class.inp_data.shape)
            #u, inv = np.unique(inp_class.source_decoding_result_np, return_inverse=True)
            #dec_res_inp_data_np = np.array([inp_class.idx_to_data_dict[idx] for idx in u])[inv].reshape(inp_class.inp_data_np.shape)  # inp_data_np
            #inp_class.out_data = np.copy(dec_res_inp_data_np)  # inp_data


    elif inp_class.source_coding_type == "NoCompression":
        '''
        inp_class.source_decoding_result_np = np.zeros_like(inp_class.mapped_data)
        for demodul_result_idx in range(inp_class.demodulation_result.size):
            detection_result = np.argmin(np.abs(inp_class.demodulation_result[demodul_result_idx].astype('int8') - inp_class.code_arr.astype('int8')).sum(axis=1))
            inp_class.source_decoding_result_np[demodul_result_idx] = detection_result  # mapped data 결과
        '''
        demodulation_result = np.copy(inp_class.demodulation_result)
        if inp_class.mapped_data.dtype == "uint8":
            padding_num = 0
        elif inp_class.mapped_data.dtype == "uint16":
            padding_num = 16 - inp_class.inp_bit_len
        elif inp_class.mapped_data.dtype == "uint32":
            padding_num = 32 - inp_class.inp_bit_len
        else:
            assert False, "mapped_data 자료형 확인필요"

        demodulation_result = np.pad(np.flip(demodulation_result), ((0, 0), (0, padding_num)))
        source_decoding_result_np = np.packbits(demodulation_result, axis=1, bitorder='little').view(inp_class.mapped_data.dtype)
        inp_class.inp_data_unique_arr
        #######################################################여기고쳐라
        inp_class.inp_data_unique_arr_idx_arr
        #여기는 실제로 얻어진 넘파이
        inp_class.source_decoding_result_np = source_decoding_result_np.reshape(
            np.shape(inp_class.mapped_data))  # mapped data
        
        #여기는 인덱스를 근사한 넘파이
        max_idx = max(inp_class.idx_to_data_dict.keys())
        inp_class.source_decoding_result_approx_np = np.where(source_decoding_result_np> max_idx,max_idx,source_decoding_result_np).reshape(
            np.shape(inp_class.mapped_data))   #dictonary 최대 인덱스보다 큰 애들은 최대 인덱스로 매핑하는 근사.

        if inp_class.ext == ".txt":
            inp_class.out_data = "".join(list(inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np]))
        elif inp_class.ext == ".png":
            inp_class.out_data = inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np].reshape(
                inp_class.inp_data.shape)

def inp_with_noise(inp_file_dir,source_coding_type,draw_huffmantree,modulation_scheme,mu,std):
    '''
    디지털통신 시스템에 입력값을 통과시키는 함수
    '''

    inp_data, mapped_data, inp_data_unique_arr,inp_data_unique_arr_idx_arr, count, bit_len, ext = read_file_func(inp_file_dir)

    inp_class = communicationsystem(ext, inp_data, mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                                    source_coding_type,bit_len,draw_huffmantree,
                                    modulation_scheme,
                                    mu,std)

    source_encoder(inp_class)

    modulation(inp_class)

    channel_awgn(inp_class)

    demodulation(inp_class)
    source_decoder(inp_class)

#    cv2.imwrite('Test1.png', inp_class.inp_data)
#   cv2.imwrite('Test2.png', inp_class.out_data)

    return inp_class.out_data


#inp_with_noise(inp,1,8)


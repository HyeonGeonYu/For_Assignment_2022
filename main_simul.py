
import numpy as np
import communicationsystem
import cv2

#inp_file_dir = 'sample.txt'
inp_file_dir = 'Lenna.png'
source_coding_type = "Huffman" # detection 방법이 잘못되서 전체적으로 어두운 결과가 나옴
#source_coding_type = "NoCompression"
draw_huffmantree = False      # huffman이 아니면 True여도 안그림.
modulation_scheme = "BPSK"
mu = 0
''' ## standard deviation 기준으로 SNR 입력
std = np.sqrt(0.5) #SNR = 0dB
Eb_N0 = 1/(2*std**2)
SNR = 10*np.log10(Eb_N0)
'''

## SNR기준으로 standard deviation 입력
SNR = 2
Eb_N0 = 10**(SNR/10)
std = 1/np.sqrt(2*Eb_N0)

result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)


#####압축률 확인
result_class.source_coding_result_bit_num / result_class.mapped_data_bit_num
'''
    수학적 압축률(엔트로피 / mapped_data_bit_len)
    prob_arr = result_class.count/np.sum(result_class.count)
    -np.sum(prob_arr * np.log2(prob_arr))/ prob_arr.size.bit_length()
'''
#####


#####BER 확인
source_coding_result_bit_num = result_class.source_coding_result_bit_num #보내는 비트수
err_bit_num = result_class.source_coding_result_np.size - np.count_nonzero(result_class.source_coding_result_np ==result_class.demodulation_result) #에러난 비트수
BER = err_bit_num/source_coding_result_bit_num
print("%.2fdB, %.4f "%(SNR, BER))
'''
    수학적 BPSK의  BER 
    from scipy import special
    std = 0.5
    0.5 - 0.5*special.erf(1/(std*np.sqrt(2)))
'''
#####

#result_class.inp_data, result_class.out_data
#cv2.imwrite('Test_dir/Test1.png', result_class.inp_data)
#cv2.imwrite('Test_dir/Test2.png', result_class.out_data)
pass



import numpy as np
import communicationsystem
import between_std_SNR
import matplotlib.pyplot as plt


inp_file_dir = None
source_coding_type = None
channel_coding_type = None
draw_huffmantree = None
modulation_scheme = "QPSK"
fading_scheme = "Rayleigh"
Tx = 4
Rx = 4
mu = 0

SNR_arr = np.linspace(0, 30, 11)

SER_result_no = []
SER_result_ML = []
for SNR in SNR_arr: # SNR  #dB
    result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,
                                                   modulation_scheme,fading_scheme, Tx, Rx, mu, SNR)
    #####SER 확인
    Tx_sym = result_class.channel_coding_result_np.reshape(-1,2) #보낸 심볼
    Rx_sym_no = result_class.demodulation_result1.reshape(-1,2) # no
    Rx_sym_ML = result_class.demodulation_result2.reshape(-1, 2)  # ML
    num_sym = Tx_sym.shape[0]
    num_err_no = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_no,axis=1)) #err 갯수
    num_err_ML = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_ML,axis=1)) #err 갯수
    SER_no = num_err_no/num_sym
    SER_ML = num_err_ML/num_sym
    SER_result_no.append(SER_no)
    SER_result_ML.append(SER_ML)
    print("SNR : %.2f dB 완료"%SNR)

plt.xlim([0,30])
plt.yscale('log', base=10)
plt.ylim([10**-3.5,10**-0 ])
plt.yticks(fontname="DejaVu Sans")
plt.grid(True, linestyle = ':',which='both')
plt.plot(SNR_arr,SER_result_no)
plt.plot(SNR_arr,SER_result_ML,'ko-')
plt.show()

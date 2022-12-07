
import numpy as np
import communicationsystem
import between_std_SNR
import matplotlib.pyplot as plt


inp_file_dir = None
source_coding_type = None
channel_coding_type = None
draw_huffmantree = None
modulation_scheme = "QPSK"
mu = 0

SNR_arr = np.arange(0,35,5)
SER_result = []
for SNR in SNR_arr: # SNR  #dB
    std = between_std_SNR.SNR_2_std(SNR)
    result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,
                                                   modulation_scheme, mu, std)
    #####SER 확인
    Tx_sym = result_class.channel_coding_result_np.reshape(-1,2) #보낸 심볼
    Rx_sym = result_class.demodulation_result.reshape(-1,2) # 받은 심볼
    num_sym = Tx_sym.shape[0]
    num_err = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym,axis=1)) #err 갯수
    SER = num_err/num_sym
    SER_result.append(SER)
    print("%.2fdB, %.4f "%(SNR, SER))

plt.yscale('log', base=10)
plt.ylim([10**-4,10**-0 ])
plt.yticks(fontname="DejaVu Sans")
plt.grid(linestyle = ':')
plt.plot(SNR_arr,SER_result)
plt.show()

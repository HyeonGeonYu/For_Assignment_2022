
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
SER_result_ZF = []
SER_result_MMSE = []
SER_result_ZF_SIC = []
SER_result_MMSE_SIC = []
SER_result_LR_ZF  = []
SER_result_SD = []
SER_result_SDR = []
for SNR in SNR_arr: # SNR  #dB
    result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,
                                                   modulation_scheme,fading_scheme, Tx, Rx, mu, SNR)
    #####SER 확인
    Tx_sym = result_class.channel_coding_result_np.reshape(-1,2)                  #보낸 심볼
    Rx_sym_ML = result_class.demodulation_result2.reshape(-1, 2)                   # ML
    Rx_sym_ZF = result_class.demodulation_result3.reshape(-1, 2)                   # ZF
    Rx_sym_MMSE = result_class.demodulation_result4.reshape(-1, 2)             # MMSE
    Rx_sym_ZF_SIC = result_class.demodulation_result5.reshape(-1, 2)       # ZF_SIC
    Rx_sym_MMSE_SIC = result_class.demodulation_result6.reshape(-1, 2) # MMSE_SIC
    Rx_sym_LR_ZF = result_class.demodulation_result7.reshape(-1, 2)          # LR_ZF
    Rx_sym_SD = result_class.demodulation_result8.reshape(-1, 2)          # SD
    Rx_sym_SDR = result_class.demodulation_result9.reshape(-1, 2)                # SDR

    num_sym = Tx_sym.shape[0]

    num_err_ML = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_ML,axis=1))
    num_err_ZF = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_ZF,axis=1))
    num_err_MMSE = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_MMSE,axis=1))
    num_err_ZF_SIC = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_ZF_SIC,axis=1))
    num_err_MMSE_SIC = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_MMSE_SIC,axis=1))
    num_err_ZF_LR = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_LR_ZF,axis=1))
    num_err_SD = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_SD,axis=1))
    num_err_SDR = num_sym - np.count_nonzero(np.all(Tx_sym == Rx_sym_SDR,axis=1))

    SER_ML = num_err_ML/num_sym
    SER_ZF = num_err_ZF/num_sym
    SER_MMSE = num_err_MMSE/num_sym
    SER_ZF_SIC = num_err_ZF_SIC/num_sym
    SER_MMSE_SIC = num_err_MMSE_SIC/num_sym
    SER_LR_ZF = num_err_ZF_LR/num_sym
    SER_SD = num_err_SD/num_sym
    SER_SDR = num_err_SDR/num_sym

    SER_result_ML.append(SER_ML)
    SER_result_ZF.append(SER_ZF)
    SER_result_MMSE.append(SER_MMSE)
    SER_result_ZF_SIC.append(SER_ZF_SIC)
    SER_result_MMSE_SIC.append(SER_MMSE_SIC)
    SER_result_LR_ZF.append(SER_LR_ZF)
    SER_result_SD.append(SER_SD)
    SER_result_SDR.append(SER_SDR)

    print("SNR : %.2f dB 완료"%SNR)

plt.figure(figsize=(5.1,4.06))
plt.xlim([0,30])
plt.yscale('log', base=10)
plt.ylim([10**-3.45,10**0])
plt.yticks(fontname="DejaVu Sans")
plt.grid(True, linestyle = ':',which='both')

plt.plot(SNR_arr,SER_result_ML,color= 'black', label = "ML")
plt.plot(SNR_arr,SER_result_SD,'o-',color= 'red',mfc="None", label = "SD with d^2 = 9")
plt.plot(SNR_arr,SER_result_SDR,'x-',color= 'blue', label = "SDR")

plt.plot(SNR_arr,SER_result_LR_ZF,'v-',color= 'green', mfc="None",label = "LR_ZF")

plt.plot(SNR_arr,SER_result_ZF_SIC,'d-' , color = 'orange', mfc="None",  label = "ZF_SIC")
plt.plot(SNR_arr,SER_result_MMSE_SIC, '-+' , c = 'deepskyblue', label = "MMSE_SIC")
plt.plot(SNR_arr,SER_result_ZF,'>:' , color = 'orange', mfc="None",  label = "ZF")
plt.plot(SNR_arr,SER_result_MMSE,'<:' , color = 'deepskyblue', mfc="None",  label = "MMSE")
plt.legend(loc='upper right')

plt.show()
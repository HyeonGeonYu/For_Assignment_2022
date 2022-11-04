import numpy as np
import cv2
#, max_len
def to_binary(number, max_len):
    bits = "{0:b}".format(number, '#b')
    num_bits = len(bits)
    bit_list = list(map(int,bits))

    bit_list = (max_len - num_bits)*[0] + bit_list
    return bit_list

def deci2bit(input,bit_len=None):
    '''
    넘파이의 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    output_list = []
    row, col = np.shape(input)
    if bit_len ==None:
        bit_len = int(np.max(input)).bit_length()

    for i in range(row):
        for j in range(col):
            bits_ = to_binary(int(input[i][j]), bit_len)
            output_list.append(bits_)
    return np.reshape(np.array(output_list), (-1, 1))
def mod_bpsk(inputs):
    bpsk = np.where(inputs == 0, -1, 1)
    return bpsk

def demod_bpsk(inputs):
    origin_data = (inputs > 0) + np.zeros(np.shape(inputs))
    return origin_data

def channel_awgn(inputs, mu=0, sigma=0.1):
    output = inputs + np.random.normal(mu, sigma, np.shape(inputs))
    return output

'''이미지 테스트 
'''
img_color = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#inp =  np.random.randint(100, 130, size=(9, 10))
inp =  img_gray

a = deci2bit(inp,8)
b =  mod_bpsk(a)
c = (demod_bpsk(b))
d = channel_awgn(b,0,1)
e = (demod_bpsk(d))

np.reshpe(e,np.shape(inp))
pass



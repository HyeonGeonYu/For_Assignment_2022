import cv2
import numpy as np
import communicationsystem

def make_noise(std, img):
    height, width = img.shape
    img_noise = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for a in range(width):
            make_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * make_noise
            img_noise[i][a] = img[i][a] + set_noise
    return img_noise
def nothing(x):
    pass

cv2.namedWindow('with noise')
cv2.createTrackbar('분산= value/10', 'with noise', 1, 50, nothing)
cv2.setTrackbarPos('분산= value/10', 'with noise', 10)

inp_file_dir = 'Lenna.png'

img_color = cv2.imread(inp_file_dir, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', img_gray)

std = 1
#img_noise = make_noise(std, img_gray)
img_noise = communicationsystem.inp_with_noise(inp_file_dir,std)

while(True):
    std = cv2.getTrackbarPos('분산= value/10', 'with noise')/10
    #img_noise = make_noise(std, img_gray)
    #cv2.imshow('with noise', img_noise.astype(np.uint8))
    img_noise = communicationsystem.inp_with_noise(inp_file_dir, std)
    cv2.imshow('with noise', img_noise.astype(np.uint8))

    if cv2.waitKey(1000)&0xFF == 27:
        break

cv2.destroyAllWindows()
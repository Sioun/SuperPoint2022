from numpy import load
import cv2

data = load('logs\\train_pseudogroundtruth_900000\predictions\\train\Frames_B1\FrameBuffer_0682.npz')
data2 = load('logs\\tuned_SuperPoint\predictions\\train\Frames_S1\FrameBuffer_0682.npz')
pic = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\Frames_S1\FrameBuffer_0682.png')
lst = data.files
lst2 = data2.files
for item in lst:
    pst = data[item]
    print(data[item])
# for item in lst2:
    pst2 = data2[item]
    # print(data2[item])
pic2 = cv2.resize(pic,(472, 472))
pic = cv2.resize(pic,(472, 472))
r,r2 = 0,0

print(pst.shape)
print(pst2.shape)
for psti in pst:
    if psti[2] >= 0.03:
        cv2.circle(pic, [psti[0].astype(int),psti[1].astype(int)], 1, (0,225,0))
    else:
        cv2.circle(pic, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 0, 225))
        r += 1
for psti in pst2:
    if psti[2] > 0.03:
        cv2.circle(pic2, [psti[0].astype(int),psti[1].astype(int)], 1, (0,225,0))

    else:
        cv2.circle(pic2, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 0, 225))
        r2+=1
print(r,r2)
cv2.imshow('img',pic)
cv2.imshow('img2',pic2)
cv2.waitKey(0)
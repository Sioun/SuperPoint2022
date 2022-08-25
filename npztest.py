from numpy import load
import numpy as np
import cv2
import matplotlib.pyplot as plt

# data = load('logs\\tuned_not_sp_predict\predictions\\train\Frames_S4\\FrameBuffer_0682.npz')
# data2 = load('logs\\tuned_sp_predict\predictions\\train\Frames_S4\\FrameBuffer_0682.npz')
# pic = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\Frames_S4\FrameBuffer_0682.png')
# lst = data.files
# lst2 = data2.files
# for item in lst:
#     pst = data[item]
#     # print(data[item])
# for item in lst2:
#     pst2 = data2[item]
#     # print(data2[item])
# pic2 = cv2.resize(pic,(472, 472))
# pic = cv2.resize(pic,(472, 472))
# r,r2 = 0,0
#
# print(pst.shape)
# # print(pst2.shape)
# for psti in pst:
#     if psti[2] >= 0.03:
#         cv2.circle(pic, [psti[0].astype(int),psti[1].astype(int)], 1, (0,225,0))
#     else:
#         cv2.circle(pic, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 0, 225))
#         r += 1
# for psti in pst2:
#     if psti[2] >= 0.03:
#         cv2.circle(pic2, [psti[0].astype(int),psti[1].astype(int)], 1, (0,225,0))
#
#     else:
#         cv2.circle(pic2, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 0, 225))
#         r2+=1
# print(r,r2)
# cv2.imshow('img',pic)
# cv2.imshow('img2',pic2)
# cv2.waitKey(0)

#-------- read confidence -------------
dir_original = 'logs\\tuned_not_sp_predict\predictions\\train\Frames_S4\\'
dir_tuned = 'logs\\tuned_sp_predict\predictions\\train\Frames_S4\\'
avg_confi_o = np.zeros(1201)
avg_confi_t = np.zeros(1201)
r_list_ratio = []
r2_list_ratio = []
for i in range(0,1201):
    frame = str(i).zfill(4)
    # -- orginal --
    data_o = load(dir_original+'FrameBuffer_'+frame +'.npz')
    for item in data_o:
        pst_o = data_o[item]
    # print(pst_o[:,2])
    avg_confi_o[i] = np.mean(pst_o[:,2])
    # -- tuned --
    data_t = load(dir_tuned + 'FrameBuffer_' + frame + '.npz')
    for item in data_t:
        pst_t = data_t[item]
    avg_confi_t[i] = np.mean(pst_t[:,2])
    # if i % 50 == 0:
    if True:
        r = 0
        r2 = 0
        # print('image'+frame)
        # pic_o = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\Frames_S4\FrameBuffer_'+frame+'.png')
        # pic_t = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\Frames_S4\FrameBuffer_'+frame+'.png')
        total_r = len(pst_o)
        total_r2 = len(pst_t)
        for psti in pst_o:
            if psti[2] >= 0.03:
                # cv2.circle(pic_o, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 225, 0))
                a = 0
            else:
                # cv2.circle(pic_o, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 0, 225))
                r += 1
        for psti in pst_t:
            if psti[2] >= 0.03:
                # cv2.circle(pic_t, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 225, 0))
                a= 0
            else:
                # cv2.circle(pic_t, [psti[0].astype(int), psti[1].astype(int)], 1, (0, 0, 225))
                r2 += 1
        r_list_ratio.append((total_r-r)/total_r)
        r2_list_ratio.append((total_r2 - r2) / total_r2)
        # cv2.putText(pic_o,text='Frame_S4_'+frame,fontScale=0.3,org=(340,10),color=(255,0,0),thickness =1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.putText(pic_o, text='Confidence >= 0.03 : '+str(total_r-r), fontScale=0.3, org=(340, 20), color=(0, 255, 0), thickness=1,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.putText(pic_o, text='Confidence < 0.03 : ' +str(r), fontScale=0.3, org=(340, 30), color=(0, 0, 255), thickness=1,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.putText(pic_t, text='Frame_S4_' + frame, fontScale=0.3, org=(340, 10), color=(255, 0, 0), thickness=1,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.putText(pic_t, text='Confidence >= 0.03 : ' + str(total_r2 - r2), fontScale=0.3, org=(340, 20), color=(0, 255, 0),
        #             thickness=1,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.putText(pic_t, text='Confidence < 0.03 : ' + str(r2), fontScale=0.3, org=(340, 30), color=(0, 0, 255),
        #             thickness=1,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        # cv2.imwrite('E:\\2021-2022 Msc\Dissertation\Result\SP\O\\FrameBuffer_'+frame+'.png', pic_o)
        # cv2.imwrite('E:\\2021-2022 Msc\Dissertation\Result\SP\T\\FrameBuffer_'+frame+'.png', pic_t)

print(avg_confi_o.shape)
print(avg_confi_o)
print('max', max(avg_confi_o), 'min', min(avg_confi_o), 'avg', np.mean(avg_confi_o))
print(avg_confi_t.shape)
print(avg_confi_t)
print('max', max(avg_confi_t), 'min', min(avg_confi_t), 'avg', np.mean(avg_confi_t))


axis = []
for i in range(0,1201):
    axis.append(i)


plt.subplot(211)
plt.title('Ratio of Kpts with Conf>0.03 to all Kpts')
plt.plot(axis,r_list_ratio,label="Pre-trained SuperPoint",zorder = 2)
plt.hlines(np.mean(r_list_ratio), 0, 1200, label= 'avg', colors= 'orange',zorder = 5)
plt.ylabel('ratio')
plt.grid()
leg0 = plt.legend(loc='lower right')

plt.subplot(212)
plt.plot(axis,r2_list_ratio,label="Tuned SuperPoint",zorder = 2)
plt.hlines(np.mean(r2_list_ratio), 0, 1200, label= 'avg', colors= 'orange',zorder = 5)
plt.ylabel('ratio')
plt.grid()
plt.xlabel('frame time')
leg1 = plt.legend(loc='lower right')

print(np.mean(r_list_ratio), np.mean(r2_list_ratio))
plt.show()

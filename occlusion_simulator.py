import cv2
import numpy as np




import os
import shutil
#image = cv2.imread('/home/ws2/Downloads/PETA/PETAdataset/CUHK/archive/0733.png')
#print(image)
import random
i=0
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_diffiden_v1'):

    '''if f_name.endswith('_occ.jpg'):
        print(f_name)
        shutil.copy('/home/ws2/Downloads/AiC/crops/'+f_name, '/home/ws2/Downloads/AiC/crops_occ')
    else:
        shutil.copy('/home/ws2/Downloads/AiC/crops/'+f_name, '/home/ws2/Downloads/AiC/crops_nonocc')'''
    print(f_name)
    image = cv2.imread('/home/ws2/Downloads/Market1501_train_test/nonocc_train_diffiden_v1/'+str(f_name))
    #print(image)
    #assert not isinstance(image, type(None)), 'image not found'
    h, w, _ = image.shape
    x1 = random.randrange(w)
    if(x1>w//2):
        x1=w//2
    y1 = random.randrange(h)
    if (y1 > 2*h // 4):
        y1 = 2*h//4
    ad_w=random.randrange(w)
    '''if (ad_w < w // 2 ):
        ad_w = w // 2'''
    if (ad_w > w // 2 ):
        ad_w = w // 2
    x2 = x1 + ad_w
    ad_h = random.randrange(h)
    if (ad_h < h // 2):
        ad_h = h // 2
    if (ad_h > h // 2):
        ad_h = h // 2
    y2 = y1 + ad_h

    '''w0 = w // 4
    h0 = h // 4
    w1 = 3 * w // 4
    h1 = 3 * h // 4'''
    #out_image = cv2.rectangle(image, (w0, h0), (w1, h1), (0, 0, 0), cv2.FILLED)
    out_image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    i=i+1
    '''if(i>1):
        break'''
    #cv2.imshow("image", out_image)
    #cv2.waitKey(0)
    print(str(f_name[:-4]))
    cv2.imwrite('/home/ws2/Downloads/Market1501_train_test/crops_occ_diffiden_train/'+str(f_name[:-4])+'.jpg', out_image)

print(i)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


''''import cv2
import numpy as np
image = cv2.imread('/home/ws2/Downloads/Market1501_train_test/ave/0002_22.jpg')
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
#heatmap = cv2.applyColorMap(cv2.GaussianBlur(saliencyMap.astype(np.uint8) * 255, (3, 3), 0), cv2.COLORMAP_JET)
 
# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
#threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# show the images
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
#cv2.imshow("Output_heat", heatmap)
#cv2.imshow("Thresh", threshMap)
#cv2.waitKey(0)''''


# In[1]:


''''import cv2
#for fname in os.
image = cv2.imread('/home/ws2/Downloads/Market1501_train_test/ave/0002_22.jpg')

#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", image)
cv2.imshow("Output", saliencyMap)
#cv2.waitKey(0)''''


# In[4]:


import cv2
import os
for f_name in os.listdir('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1'):
    image = cv2.imread('/home/ws2/Downloads/Market1501_train_test/nonocc_train_v1/'+str(f_name))
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    cv2.imwrite('/home/ws2/Downloads/Market1501_train_test/target_nonocc_pose_train/'+str(f_name),saliencyMap)


# In[ ]:






# coding: utf-8

# In[29]:


import cv2
import numpy as np
import pylab as pl
from PIL import Image
from pytesseract import image_to_string


# In[30]:


filename='sample_image5.jpg'
input_image = cv2.imread(filename)
input_image_gray =cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)


# In[31]:


#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', 1000,1000)
#cv2.imshow('image',thresholded_roi)
#cv2.waitKey(0)


# In[32]:


#for i in range(1,3):
 #   input_image_gray= cv2.bilateralFilter(input_image_gray,9,50,50)
 #   thresholded_roi = cv2.adaptiveThreshold(input_image_gray,255,1,1,11,15);
            


# In[33]:


#input_image_gray=cv2.medianBlur(input_image_gray,3)
#input_image_gray=cv2.GaussianBlur(input_image_gray,(3,3),0)
input_image_gray= cv2.bilateralFilter(input_image_gray,5,3,3)
canny_image = cv2.Canny(input_image_gray,50,150)


# In[34]:


contours, hierarchy = cv2.findContours(canny_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# In[35]:


max_size_rect=0
copy_input_img = input_image.copy()
for cnt in contours:
    rect = cv2.boundingRect(cnt)
    x, y, w, h = rect
    if w*h>max_size_rect:
        biggest_rectangle = rect
        max_size_rect=w*h
x, y, w, h = biggest_rectangle
cv2.rectangle(copy_input_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
#cv2.drawContours(copy_input_img, [cnt], 0, (0,0,255), 3)


# In[36]:


roi = input_image[y:y+h, x:x+w]


# In[37]:


cv2.imwrite("ROI.jpg", roi)


# In[38]:


roi=cv2.imread('r.jpg')


# In[39]:


roi_gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)


# In[40]:


for i in range(1,3):
    roi_gray=cv2.bilateralFilter(roi_gray,7,50,50)
thresholded_roi = cv2.adaptiveThreshold(roi_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,            cv2.THRESH_BINARY,11,2)


# In[41]:


cv2.imwrite("threshold.jpg",thresholded_roi)


# In[42]:


kernel = np.ones((3,3), np.uint8)
input_image_gray_dilation = cv2.erode(thresholded_roi, kernel, iterations=1)
cv2.imwrite("eroded_img.jpg", input_image_gray_dilation)


# In[43]:


contours_roi, hierarchy_roi = cv2.findContours(input_image_gray_dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# In[44]:


copy_roi = roi.copy()
mask = np.zeros(copy_roi.shape,np.uint8)
for cnt in contours_roi:
    rect = cv2.boundingRect(cnt)
    x, y, w, h = rect
    if w*h>1000:
        biggest_rectangle = rect
        max_size_rect=w*h
        x, y, w, h = biggest_rectangle
        if w>h:
            cv2.rectangle(mask, (x, y+5), (x+w-5, y+h-7), (255, 255, 255), -1)
        else:
            cv2.rectangle(mask, (x+3, y), (x+w-5, y+h), (255, 255, 255), -1)
cv2.imwrite("mask.jpg", mask)


# In[45]:


mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
inverse_threshold=255-thresholded_roi


# In[49]:


final_img = cv2.bitwise_and(inverse_threshold,inverse_threshold, mask=mask)
inverse_final_img=255-final_img
cv2.imwrite("final.jpg", inverse_final_img)


# In[50]:


print image_to_string(Image.open('final.jpg'))


# In[47]:





# In[345]:





# In[ ]:





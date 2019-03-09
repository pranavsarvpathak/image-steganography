#!/usr/bin/env python
# coding: utf-8

# In[1]:


# STEGANOGRAPHY.py
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


file_name = input("Enter name of larger image or absolute directory: ")
img = cv2.imread(file_name)
img = np.array(img)
img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0].copy()
plt.imshow(img)


# In[3]:


file_name = input("Enter name of smaller image or absolute directory: ")
img2 = cv2.imread(file_name)
img2 = np.array(img2)
img2[:, :, 0], img2[:, :, 2] = img2[:, :, 2], img2[:, :, 0].copy()
plt.imshow(img2)


# In[4]:


# Function to save new image
def save_image(img):
    z = img.copy()
    z[:, :, 0], z[:, :, 2] = z[:, :, 2], z[:, :, 0].copy()
    file_name = input("To Save, enter filename only: ")
    cv2.imwrite(file_name + ".png", z, [0])  # for preserving quality in png


# In[5]:


print(img.shape)
print(img2.shape)


# In[6]:


# function for steganography
def steganography(img, img2):
    for k in range(3):
        for i in range(img2.shape[0]):
            col = 0
            for j in range(img2.shape[1]):
                c = 0
                while c != 8:
                    if img[i][col][k] % 2:
                        img[i][col][k] -= 1
                    img[i][col][k] += img2[i][j][k] % 2
                    img2[i][j][k] //= 2
                    col = col + 1
                    c = c + 1
    return img


# In[7]:


steg = img.copy()
to_hide = img2.copy()
steg = steganography(steg, to_hide)


# In[8]:


plt.imshow(steg)
save_image(steg)


# In[9]:


# function for extracting image
def extraction(steg, shape):
    ext = np.zeros(shape, dtype=int)
    for k in range(3):
        for i in range(shape[0]):
            for j in range(shape[1] * 8):
                if j % 8 == 0:
                    pro = 1
                else:
                    pro *= 2
                ext[i][j // 8][k] += steg[i][j][k] % 2 * pro
    return ext


# In[10]:


file_name = input("Enter name of steganographed image or absolute directory: ")
steg_img = cv2.imread(file_name)
steg_img = np.array(steg_img)
steg_img[:, :, 0], steg_img[:, :, 2] = steg_img[:, :, 2], steg_img[:, :, 0].copy()
plt.imshow(steg_img)


# In[11]:


# Extracting from steganographed image
extract = extraction(steg_img, img2.shape)


# In[12]:


plt.imshow(extract)
save_image(extract)


# In[13]:


# To check whether larger image is tampered or not
diff = img - steg
plt.imshow(diff)


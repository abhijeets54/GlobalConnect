#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


# In[2]:


name="tortoise"


# In[3]:


path=f"/home/skumar/Desktop/Trainee/Swaraj/test_images/{name}.jpg"


# In[4]:


global shi
global exp
shi=0
exp=0


# In[5]:


def normalized_correlation_coefficient(image1, image2):
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    std1 = np.std(image1)
    std2 = np.std(image2)
    correlation = np.sum((image1 - mean1) * (image2 - mean2)) / (image1.size * std1 * std2)
    return correlation


# In[6]:


def PSNR(arr1,arr2):
    mse=np.mean((arr1-arr2)**2)
    if (mse==0):
        return np.inf;
    psnr=20*np.log10(255/np.sqrt(mse))
    return psnr


# In[7]:


def ssim(img1, img2, dynamic_range=255.0):
    mu_x = np.mean(img1)
    mu_y = np.mean(img2)
    sigma_x_sq = np.var(img1)
    sigma_y_sq = np.var(img2)
    sigma_xy = np.cov(img1.flatten(), img2.flatten())[0, 1]

    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2

    l = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    c = (2 * np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq) + C2) / (sigma_x_sq + sigma_y_sq + C2)
    s = (sigma_xy + C2 / 2) / (np.sqrt(sigma_x_sq) * np.sqrt(sigma_y_sq) + C2 / 2)
    ssim_index = l * c * s

    return ssim_index


# In[8]:


def create_hash_map(arr):
    hash_map={}
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if(arr[i][j] in hash_map):
                hash_map[arr[i][j]]+=1
            else:
                hash_map[arr[i][j]]=1
    for i in range(256):
        if i not in hash_map:
            hash_map[i]=0
    hash_map=dict(sorted(hash_map.items()))
    keymax=max(zip(hash_map.values(),hash_map.keys()))
    return hash_map,keymax


# In[9]:


def find_delta(img, min_val, max_val):
    H = img.astype(dtype=np.float64)
    h, w = img.shape
    mx_capacity = 0

    for delta in range(min_val, max_val):
        map = np.zeros(w, dtype=int)
        flag = np.zeros(w, dtype=int)
        map[0] = 0  # Initialize with zero-based index
        flag[0] = 1

        for i in range(1, len(map)):
            map[i] = (map[i - 1] + delta) % w  # Adjust index using modulo
            if flag[map[i]] == 1:
                map[i] = (map[i] + 1) % w
            flag[map[i]] = 1

        diff = np.zeros((h, w // 2), dtype=np.float64)
        for i in range(w // 2):
            x = 2 * i
            y = x + 1
            diff[:, i] = H[:, map[x]] - H[:, map[y]]

        u_val, counts = np.unique(diff, return_counts=True)
        if len(u_val) == 1:
            peak = u_val[0]
            capacity = w * h // 2
        else:
            ind = np.argmax(counts)
            capacity = counts[ind]
            peak = u_val[ind]

        if capacity > mx_capacity:
            mx_capacity = capacity
            mx_delta = delta
            mx_map = map
            mx_peak = peak
            diff_img=diff
    return mx_delta, mx_capacity, mx_map, mx_peak,diff_img


# In[10]:


def histogram_shifting(arr1,peak):
    for i in range(len(arr1)):
        for j in range(len(arr1[0])):
            if(arr1[i][j]>peak):
                arr1[i][j]=arr1[i][j]+1
    return arr1


# In[11]:


def embedding(arr,peak,data):
    ind=0
    for i in range(len(arr)):
        if(ind==len(data)):
            break
        for j in range(len(arr[0])):
            if(arr[i][j]==peak):
                if(data[ind]=='1'):
                    arr[i][j]=arr[i][j]+1
                    ind+=1
                    if(ind==len(data)):
                        break
                else:
                    ind+=1
                    if(ind==len(data)):
                        break
    return arr


# In[12]:


def generate_transformed_embedded(transformed_image, embedded_diff_image,embedd=False):
    n = transformed_image.shape[1]
    half_n = n // 2
    transformed_with_embedded = np.copy(transformed_image)
    for i in range(half_n):
        old_value=np.copy(transformed_with_embedded[:, 2 * i])
        transformed_with_embedded[:, 2 * i] = transformed_image[:, 2 * i + 1] + embedded_diff_image[:, i]
        if(embedd==True):
            global shi
            shi+=embedded_diff_image.shape[0]
    return transformed_with_embedded


# In[1]:


def construct_image(transformed_with_embedded,map):
    sorted_columns = np.argsort(map)
    marked_image = transformed_with_embedded[:, sorted_columns]
    return marked_image


# In[14]:


def difference_image_for_single_delta(arr,map):
   H=arr.astype(dtype=np.float64)
   h,w=arr.shape
   diff = np.zeros((h, w // 2), dtype=np.float64)
   for i in range(w // 2):
            x = 2 * i
            y = x + 1
            diff[:, i] = H[:, map[x]] - H[:, map[y]]
   return diff


# In[15]:


def deshift(arr,peak):
  for i in range(len(arr)):
    for j in range(len(arr[0])):
      if(arr[i][j]>peak):
        arr[i][j]-=1
  return arr


# In[16]:


def extract_data(arr,peak,length):
  data=""
  for i in range(len(arr)):
    for j in range(len(arr[0])):
      if(arr[i][j]==peak+1):
        data=data+"1"
        arr[i][j]-=1
        if(len(data)==length):
          return arr,data
      elif(arr[i][j]==peak):
        data=data+"0"
        if(len(data)==length):
          return arr,data


# In[17]:


def posp(shi,exp):
    return shi/(shi+exp)


# In[18]:



# image = np.array([
#     [82, 81, 72, 72, 81, 80, 71, 72],
#     [84, 85, 78, 77, 83, 84, 77, 77],
#     [92, 96, 45, 80, 91, 93, 44, 78],
#     [102, 99, 103, 88, 101, 97, 102, 88],
#     [78, 79, 77, 30, 77, 77, 76, 30],
#     [90, 90, 75, 77, 89, 86, 74, 77],
#     [46, 44, 36, 45, 45, 44, 34, 47],
#     [44, 78, 82, 66, 43, 77, 81, 68]
# ])


# In[19]:


img=Image.open(path)
grayscale_image=img.convert("L")
image=np.array(grayscale_image)


# In[20]:


mx_delta, mx_capacity, mx_map, mx_peak,difference_image=find_delta(image, 1, image.shape[1])


# In[21]:


print(f"Max capacity: {mx_capacity}, Delta: {mx_delta}, Peak: {mx_peak}")


# In[22]:


transformed_image = image[:, mx_map]
print(transformed_image)


# In[23]:


copy_difference_image=np.copy(difference_image)


# In[24]:


inter_diff_image= histogram_shifting(copy_difference_image,mx_peak)


# In[25]:


difference_image[0]


# In[26]:


inter_diff_image[0]


# In[27]:


copy_inter_diff_img=np.copy(inter_diff_image)


# In[28]:


# data="111100111111111010"


# In[29]:


data="1010100000010101010101"
for i in range(len(data),mx_capacity):
    data+="1"


# In[30]:


len(data)


# In[31]:


marked_diff_image=embedding(copy_inter_diff_img,mx_peak,data)


# In[32]:


marked_diff_image[0]


# In[33]:


copy_marked_diff_image=np.copy(marked_diff_image)


# In[34]:


test1=transformed_image[0]


# In[35]:


transformed_with_embedded= generate_transformed_embedded(transformed_image, copy_marked_diff_image,embedd=True)


# In[36]:


print(shi)


# In[37]:


exp=mx_capacity


# In[38]:


ps=posp(shi,exp)


# In[39]:


transformed_with_embedded


# In[40]:


mx_map


# In[41]:


transformed_with_embedded[0]


# In[42]:


marked_image=construct_image(transformed_with_embedded,mx_map)


# In[43]:


copy_marked_image=np.copy(marked_image)


# In[44]:


marked_image[0]


# In[45]:


psnr=PSNR(image,marked_image)
ss=ssim(image,marked_image)
total_capacity=mx_capacity
corr= normalized_correlation_coefficient(image,marked_image)


# # extraction

# In[46]:


rec_transformed_image= copy_marked_image[:,mx_map]
print(rec_transformed_image[0])


# In[47]:


rec_diff_image= difference_image_for_single_delta(copy_marked_image,mx_map)


# In[48]:


rec_diff_image[0]


# In[49]:


copy_rec_diff_image=np.copy(rec_diff_image)


# In[50]:


marked_image[0][38],marked_image[0][39]


# In[51]:


intermed_marked_image,extracted_data=extract_data(rec_diff_image,mx_peak,len(data))


# In[52]:


len(data)


# In[53]:


len(extracted_data)


# In[54]:


for i in range(len(copy_rec_diff_image[0])):
    if(copy_rec_diff_image[0][i]==0):
        print(i)


# In[55]:


copy_rec_diff_image[0]


# In[56]:


intermed_marked_image[0]


# In[57]:


len(data)


# In[58]:


extracted_data==data


# In[59]:






recovered_diff_arr=deshift(intermed_marked_image,mx_peak)


# In[60]:


recovered_diff_arr


# In[61]:


rec_transformed_image


# In[62]:


recreated_transformed_image=generate_transformed_embedded(rec_transformed_image,recovered_diff_arr)


# In[63]:


ti=recreated_transformed_image[:]


# In[64]:


recovered_image=construct_image(ti,mx_map)


# In[65]:


recovered_image


# In[66]:


h,kx=create_hash_map(image)
h1,k1=create_hash_map(recovered_image)


# In[67]:


h==h1


# In[68]:


kx==k1


# In[69]:


extracted_data==data


# In[70]:


print(total_capacity,f"{psnr:.2f}",f"{ss:.5f}",f"{ps:.5f}",f"{corr:.5f}")


# In[71]:


print(shi)


# In[72]:


mx_capacity


# In[73]:


512*512


# In[ ]:





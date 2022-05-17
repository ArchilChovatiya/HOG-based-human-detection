#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle as pkl
import numpy as np
import cv2
import math
import glob
from PIL import Image as im


# In[2]:


# This function is used to load Masks from .pkl files
def loadMask(path,lable):  
    with open(path, 'rb') as f:
        mask = pkl.load(f)
        return mask


# In[3]:


def display_Img(img,lable="Img"): # function to Display an Image 
    img=np.uint8(img)
    cv2.imshow(lable,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #data = im.fromarray(img) # uncomment to the save Images
    #data.save(lable+'.bmp')  # uncomment to the save Images


# In[4]:


# This function loads image and converts 3 channel image to single channel grayscale image
def load_img(path):  
    img=cv2.imread(path)
    h,w = img.shape[:2]
    gray = np.zeros((h,w), np.uint8)
    for i in range(h):
        for j in range(w):
            gray[i,j] = np.clip(0.114 * img[i,j,0]  + 0.587 * img[i,j,1] + 0.299 * img[i,j,2], 0, 255)
    return gray


# In[5]:


# This function is used to apply operators like prewitt,Robert on an Image
def apply_mask(oldImg,mask,pedding=0): 
    ( h , w ) = oldImg.shape
    newImg = np.zeros( ( h , w ), dtype = np.int32 )
    P = ( mask[0].size//2 ) 
    for j in range( (P + pedding) , h - (P + pedding) ):
        for i in range( (P + pedding) , w - (P + pedding) ):
            value = 0
            for k in range( -P , P+1 ):
                for l in range( -P , P+1 ):
                    value += mask[k+P][l+P] * oldImg[j+k][i+l]
            newImg[j][i] = value
    return ( newImg , P + pedding )


# In[6]:


# In Normalization, pixel values are rescale to 0-255 range
def Normalization(img): 
    img=np.absolute(img)
    img=img/img.max()*255
    return img


# In[7]:


# Function to calculate gradient magnitude from horizontal and vertical gradient
def gradient_magnitude(gx,gy): 
    ( h , w ) = gx.shape
    grad_mag = np.zeros( ( h , w ), dtype = np.uint32 )
    for j in range(h):
        for i in range(w):
            grad_mag[j][i]= abs(gx[j][i])+abs(gy[j][i])
    return grad_mag


# In[8]:


# Function to calculate gradient angle from horizontal and vertical gradient
def gradient_angle(gx,gy): 
    ( h , w ) = gx.shape
    grad_ang = np.zeros( ( h , w ), dtype = np.float32 )
    for j in range(h):
        for i in range(w):
            if gx[j][i] != 0:
                grad_ang[j][i]= math.degrees( math.atan(gy[j][i]/gx[j][i]) )
    return grad_ang


# In[9]:


def Gradient_Operation( Image , pedding=0 ):
    # loading Prewitt X derivative from .pkl file
    Prewitt_X = loadMask('Pickle/prewitt-x.pkl','Prewitt X derivative') 
    # loading Prewitt Y derivative from .pkl file
    Prewitt_Y = loadMask('Pickle/prewitt-y.pkl','Prewitt Y derivative') 
    (Gx , _ ) = apply_mask(Image,Prewitt_X,pedding) # Calculating horizontal gradient
    Gx=Normalization(Gx)  # Normalizing horizontal gradient image
    (Gy , pedding ) = apply_mask(Image,Prewitt_Y,pedding) #Calculating vertical gradient
    Gy=Normalization(Gy)  # Normalizing vertical gradient image
    Grad_Mag=gradient_magnitude(Gx,Gy) # Calculating gradient magnitude 
    Grad_Mag=Normalization(Grad_Mag) # Normalizing gradient magnitude image
    Grad_Ang = gradient_angle(Gx,Gy) # Calculating gradient angle
    return ( Grad_Mag , Grad_Ang )


# In[10]:


def extract_features(training_image_path):
    list_vec=np.array([])
    list_lable=np.array([])
    lable=0
    for dir in glob.glob(training_image_path): # iterating class directory 
        for file_name in glob.glob(dir+"\\*"): # iterating image in the diractory
            Img=load_img(file_name) #Loading Image
            (Grad_Mag,Grad_Ang)=Gradient_Operation(Img) #Gradient operation on image
            vec=np.array([])
            h,w = Grad_Mag.shape[:2]
            for i in range(0,h-8,8):  
                for j in range(0,w-8,8):
                    for k in range(i,i+16,8):
                        for l in range(j,j+16,8):
                            Bin = np.zeros(9)
                            for m in range (k,k+8):
                                for n in range(l,l+8):   
                                    ang=Grad_Ang[m][n]%180 # if angle is not in [0,180) then converting it to [0,180) 
                                    bin_index1=int(ang/20)%9 # calculating index1 of bin
                                    bin_index2=(bin_index1+1)%9 # calculating index2 of bin
                                    Bin[bin_index1]=Bin[bin_index1]+Grad_Mag[m][n]*(20-(ang%20))/20 # adding gradient value to bin
                                    Bin[bin_index2]=Bin[bin_index2]+Grad_Mag[m][n]*(ang%20)/20 # adding gradient to bin
                            vec=np.append(vec,Bin) # appending current 9 histogram channels to previous channels 
            vec=vec/math.sqrt(sum(p*p for p in vec)) # Normalization
            list_lable=np.append(list_lable,[lable]) # labeling according to the class
            
            '''a_file = open(file_name[:-3]+"txt", "w")
            for row in vec:
                np.savetxt(a_file, [row],fmt="%f")
            a_file.close()''' #uncomment this to create .txt files of the image feature vector
            
            if list_vec.size == 0: # creating list of features of each image
                list_vec=np.array([vec])
            else:
                list_vec=np.concatenate((list_vec,[vec]),axis=0)
        lable=lable+1
    list_lable=np.uint8(list_lable)
    return list_vec,list_lable


# In[11]:


list_vec,list_lable=extract_features("Image Data\\Training\\*")


# In[12]:


NN_lables=["1st NN: ","2nd NN: ","3rd NN: "]
is_person=["No Human","Human"]
def detect_Image(testing_image_path,list_vec,list_lable):
    for dir in glob.glob(testing_image_path): # iterating class directory
        for file_name in glob.glob(dir+"\\*"): # iterating image in the diractory
            Img=load_img(file_name) #Loading Image
            (Grad_Mag,Grad_Ang)=Gradient_Operation(Img) #Gradient operation on image
            #data = im.fromarray(np.uint8(Grad_Mag)) # uncomment to the save Images
            #data.save("Gred "+file_name[:-4]+"_Gred.bmp")  # uncomment to the save Images
            input_image_vec=np.array([])
            h,w = Grad_Mag.shape[:2]
            for i in range(0,h-8,8):
                for j in range(0,w-8,8):
                    for k in range(i,i+16,8):
                        for l in range(j,j+16,8):
                            Bin = np.zeros(9)
                            for m in range (k,k+8):
                                for n in range(l,l+8):   
                                    ang=Grad_Ang[m][n]%180 # if angle is not in [0,180) then converting it to [0,180) 
                                    bin_index1=int(ang/20)%9 # calculating index1 of bin
                                    bin_index2=(bin_index1+1)%9 # calculating index2 of bin
                                    Bin[bin_index1]=Bin[bin_index1]+Grad_Mag[m][n]*(20-(ang%20))/20 # adding gradient value to bin
                                    Bin[bin_index2]=Bin[bin_index2]+Grad_Mag[m][n]*(ang%20)/20 # adding gradient value to bin
                            input_image_vec=np.append(input_image_vec,Bin) # appending current 9 histogram channels to previous channels
            input_image_vec=input_image_vec/math.sqrt(sum(p*p for p in input_image_vec)) # Normalizing
            
            '''a_file = open(file_name[:-3]+"txt", "w")
            for row in input_image_vec:
                np.savetxt(a_file, [row],fmt="%f")
            a_file.close()''' #uncomment this to create .txt files of the image feature vector
            
            diff_vec_list=np.array([])
            for training_image_vec in list_vec:
                diff_vec=sum(np.minimum(input_image_vec,training_image_vec))/sum(training_image_vec) # Histogram intersection formula
                if list_vec.size == 0:
                    diff_vec_list=np.array([diff_vec])
                else:
                    diff_vec_list=np.concatenate((diff_vec_list,[diff_vec]),axis=0)
            print(file_name+": ")
            r=0
            s=0
            for min_value in np.flip(sorted(diff_vec_list)[-3:]): # Finding 3NNs
                index=list_lable[np.where(diff_vec_list==min_value)][0]
                print(NN_lables[r]+is_person[index]+", Distance: "+str(diff_vec_list[np.where(diff_vec_list==min_value)][0]))
                s=s+index
                r=r+1
            if s>1:
                print("Final class: Human")
            else:
                print("Final class: No Human")
            print("\n")


# In[13]:


detect_Image("Image Data\\Test\\*",list_vec,list_lable)


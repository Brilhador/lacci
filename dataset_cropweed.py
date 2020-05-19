import os
import yaml
import cv2 #open cv
import numpy as np
import random  
import seaborn as sns # Seaborn is a Python data visualization library based on matplotlib

from scipy.ndimage import label

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

dir_data = "dataset/cropweed/"

dir_seg = "dataset/cropweed/annotations/data/"
dir_img = "dataset/cropweed/images/data/"

## data augmentation
#dir_seg = "dataset/cropweed/annotations_aug_10/data/"
#dir_img = "dataset/cropweed/images_aug_10/data/"

## crop augmentation
#dir_seg = "dataset/cropweed/crop/annotations_aug/data/"
#dir_img = "dataset/cropweed/crop/images_aug/data/"


n_classes = 3

def give_color_to_seg_img(seg, n_classes):

    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)

def getImageArrResize(path, width, height):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1 ## normalization by sub_and_divide
        return img

def getImageArr(path):
        img = cv2.imread(path, 1)
        img = np.float32(img) / 127.5 - 1 ## normalization by sub_and_divide
        return img

def getSegmentationArrResize(path, nClasses, width, height):

    seg_labels = np.zeros((height ,width, 3))

    img = cv2.imread(path)  ## opencv use BGR
    img = cv2.resize(img, (width, height))
    
    mask_0 = (img == [0,0,0]).all(axis=2) # background
    mask_1 = (img == [0,0,255]).all(axis=2) # weed 
    mask_2 = (img == [0,255,0]).all(axis=2) # crop
    
    seg_labels[mask_0, 0] = 1
    seg_labels[mask_1, 1] = 1
    seg_labels[mask_2, 2] = 1
    
    return seg_labels

def getSegmentationArr(path, nClasses):

    img = cv2.imread(path)  ## opencv use BGR

    height, width, channels = img.shape
    seg_labels = np.zeros((height ,width, 3))

    mask_0 = (img == [0,0,0]).all(axis=2) # background
    mask_1 = (img == [0,0,255]).all(axis=2) # weed 
    mask_2 = (img == [0,255,0]).all(axis=2) # crop
    
    seg_labels[mask_0, 0] = 1
    seg_labels[mask_1, 1] = 1
    seg_labels[mask_2, 2] = 1
    
    return seg_labels

def getSegmentationArrSVM(path, nClasses, width, height):

    seg_labels = np.zeros((height ,width))

    img = cv2.imread(path)  ## opencv use BGR
    img = cv2.resize(img, (width, height))
    
    mask_0 = (img == [0,0,0]).all(axis=2) # background
    mask_1 = (img == [0,0,255]).all(axis=2) # weed 
    mask_2 = (img == [0,255,0]).all(axis=2) # crop
    
    seg_labels[mask_0] = 0
    seg_labels[mask_1] = 1
    seg_labels[mask_2] = 2
    
    return seg_labels

def getImageLabels(path, width, height):
    
    img_labels = np.zeros((height ,width))

    img = cv2.imread(path)  ## opencv use BGR
    img = cv2.resize(img, (width, height))
    
    mask_0 = (img == [0,0,0]).all(axis=2) # background
    mask_1 = (img == [0,0,255]).all(axis=2) # weed 
    mask_2 = (img == [0,255,0]).all(axis=2) # crop
    
    img_labels[mask_0] = 0
    img_labels[mask_1] = 1
    img_labels[mask_2] = 2
    
    return img_labels

def visualizeDataset():

    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)

    # first images name
    fni = rawimg[0]
    fns = imgseg[0]
    print(fni, fns)

    # read in the raw image (original) 
    img = cv2.imread(dir_img + fni)
    height, width, channels = img.shape

    # read in the annotatios labels (segmentated)
    # background, crop and weed
    #seg = cv2.imread(dir_seg + fns, cv2.IMREAD_COLOR)
    #seg = cv2.normalize(seg, None, 0, 2, cv2.NORM_MINMAX) 
    seg = getImageLabels(dir_seg + fns, width, height)

    # shape images
    print("seg.shape{}, img.shape{}".format(seg.shape, img.shape))

    # check the number of labels
    mi, ma = np.min(seg).astype(int), np.max(seg).astype(int)
    n_classes = ma - mi + 1
    print("minimum seg = {}, maximum seg = {}, Total number of segmentation classes = {}".format(mi, ma, n_classes))

    # display the original image
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img)
    ax.set_title("Original Image")
    plt.show()

    # display the annotations labels
    fig = plt.figure(figsize=(5,5))
    for k in range(mi, ma+1):
        ax = fig.add_subplot(3, n_classes/3, k+1)
        ax.imshow((seg == k) * 1.0)
        ax.set_title({0: 'background', 1: 'weed', 2: 'crop'}[k])
    plt.tight_layout()
    plt.show()

    print("Visualize Dataset")
    
def displayResizeData(num_show, width, height):

    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)

    # 
    for i in range(1, num_show):

        img = cv2.imread(dir_img + rawimg[i])
        img_height, img_width, channels = img.shape
        seg = getImageLabels(dir_seg + imgseg[i], img_width, img_height)
        seg_img = give_color_to_seg_img(seg, n_classes)

        fig = plt.figure(figsize=(20,40))
        ax = fig.add_subplot(1,4,1)
        ax.imshow(seg_img)
        
        ax = fig.add_subplot(1,4,2)
        ax.imshow(img/255.0)
        ax.set_title("original image {}".format(img.shape[:2]))
        
        ax = fig.add_subplot(1,4,3)
        ax.imshow(cv2.resize(seg_img,(height , width)))
        
        ax = fig.add_subplot(1,4,4)
        ax.imshow(cv2.resize(img,(height , width))/255.0)
        ax.set_title("resized to {}".format((height , width)))
        plt.show()

    print("Display Resized Data")

def getData(train_rate, width, height):

    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)
        
    # return array images
    # X = raw images
    # Y = annotations images    
    X = []
    Y = []

    for im, seg in zip(rawimg, imgseg):
        X.append( getImageArrResize(dir_img + im, width , height))
        Y.append( getSegmentationArrResize( dir_seg + seg, n_classes, width, height))

    X, Y = np.array(X) , np.array(Y)

    # split between training and testing data
    index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
    index_test  = list(set(range(X.shape[0])) - set(index_train))
                                
    X, Y = shuffle(X, Y)
    X_train, y_train = X[index_train], Y[index_train]
    X_test, y_test = X[index_test], Y[index_test]

    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test

### https://www.kaggle.com/stefanie04736/simple-keras-model-with-k-fold-cross-validation
### https://github.com/keras-team/keras/issues/1711
def load_data_resize(width, height):
    
    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)
        
    # return array images
    # X_train = raw images
    # y_train = annotations images    
    X_train = []
    y_train = []

    for im, seg in zip(rawimg, imgseg):
        X_train.append( getImageArrResize(dir_img + im, width , height))
        y_train.append( getSegmentationArrResize( dir_seg + seg, n_classes, width, height))

    X_train, y_train = np.array(X_train), np.array(y_train)

    print(X_train.shape, y_train.shape)

    return X_train, y_train

def load_data():

    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)
        
    # return array images
    # X_train = raw images
    # y_train = annotations images    
    X_train = []
    y_train = []

    for im, seg in zip(rawimg, imgseg):
        X_train.append( getImageArr(dir_img + im))
        y_train.append( getSegmentationArr( dir_seg + seg, n_classes))

    X_train, y_train = np.array(X_train), np.array(y_train)

    print(X_train.shape, y_train.shape)

    return X_train, y_train

def load_paths():

    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)

    return rawimg, imgseg
        
def load_images_from_paths(rawimg, imgseg):
    # return array images
    # X_train = raw images
    # y_train = annotations images    
    X_train = []
    y_train = []

    for im, seg in zip(rawimg, imgseg):
        X_train.append( getImageArr(dir_img + im))
        y_train.append( getSegmentationArr( dir_seg + seg, n_classes))

    X_train, y_train = np.array(X_train), np.array(y_train)

    print(X_train.shape, y_train.shape)

    return X_train, y_train

def getDataSVM(train_rate, width, height):

    # raw images
    rawimg = np.array([f for f in os.listdir(dir_img) if(f.endswith('.png'))])
    # annotations images
    imgseg = np.array([f for f in os.listdir(dir_seg) if(f.endswith('.png'))])

    # sort images to capture first image 
    rawimg = np.sort(rawimg)
    imgseg = np.sort(imgseg)
        
    # return array images
    # X = raw images
    # Y = annotations images    
    X = []
    Y = []

    for im, seg in zip(rawimg, imgseg):
        X.append( getImageArr(dir_img + im, width , height))
        Y.append( getSegmentationArrSVM( dir_seg + seg, n_classes, width, height))

    X, Y = np.array(X) , np.array(Y)

    # split between training and testing data
    index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
    index_test  = list(set(range(X.shape[0])) - set(index_train))
                                
    X, Y = shuffle(X, Y)
    
    X_train, y_train = X[index_train], Y[index_train]
    X_test, y_test = X[index_test], Y[index_test]

    ###
    samples, height, width, num_class = X_train.shape
    X_train = X_train.reshape(samples * height * width, num_class)

    samples, height, width, num_class = X_test.shape
    X_test = X_test.reshape(samples * height * width, num_class)

    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)
    
    return X_train, y_train, X_test, y_test


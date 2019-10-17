import numpy as np
import cv2
import os
import glob

def pad(image, height, width):
    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return constant


def get_imagelist(input_folder):
        filetype = 'png' # ajustar para seu tipo        
        files = sorted(glob.glob(input_folder+'*.'+filetype))
        return files

def load_images(imagelist):
        images = []
        for i in imagelist:
                img = cv2.imread(i,1)
                images.append(img)

        images = np.array(images)
        return images


def save_crops(crops, filename):
        for i,c in enumerate(crops):
                print('saving',filename[:-4]+'_crop_'+str(i)+filename[-4:])
                cv2.imwrite(filename[:-4]+'_crop_'+str(i)+filename[-4:],c)

def crop_image(image, crop_size):
        # print(image.shape)
        crops = []
        size_x = crop_size[0]
        size_y = crop_size[1]

        n_crops_x = np.ceil(image.shape[0] / float(size_x)) 
        n_crops_y = np.ceil(image.shape[1] / float(size_y)) 

        required_x = n_crops_x * size_x
        required_y = n_crops_y * size_y

        # print(n_crops_x, n_crops_y, required_x, required_y)
        required_x = required_x - image.shape[0]
        required_y = required_y - image.shape[1]

        # print(n_crops_x, n_crops_y, required_x, required_y)
        required_x /= 2.
        required_y /= 2.

        required_x = required_x
        required_y = required_y

        # print(n_crops_x, n_crops_y, required_x, required_y)

        new_image = pad(image, required_x, required_y)
        bordersize=10
        border=cv2.copyMakeBorder(image, top=int(np.ceil(required_x)), bottom=int(np.floor(required_x)), left= int(np.ceil(required_y)), right=int(np.floor(required_y)), borderType= cv2.BORDER_CONSTANT)

        image = border
        x_start = 0
        y_start = 0
        for nx in range(int(n_crops_x)):
                # print(nx)
                for ny in range(int(n_crops_y)):
                        crop = image[x_start:x_start+size_x, y_start:y_start+size_y].copy()
                        crops.append(crop)
                        # print(crop.shape, x_start, y_start)
                        y_start += size_y
                y_start = 0
                x_start += size_x

        return crops

def crop_generator_online(images, crop_size):

        data = np.array([])
        flag = True

        for i, img in enumerate(images):
                crops = crop_image(img, crop_size)
                # print(np.array(crops).shape)

                if flag:
                        data = crops
                        flag = False
                else:
                        data = np.append(data, crops, axis=0)

        return data


print("Start")
dir_data = "dataset/cropweed"
dir_seg = "dataset/cropweed/annotations/data/"
dir_img = "dataset/cropweed/images/data/"

dir_seg_aug = "dataset/cropweed/crop/annotations_aug/data/"
dir_img_aug = "dataset/cropweed/crop/images_aug/data/"

## save img annotations
input_folder = dir_seg
output_folder = dir_seg_aug
crop_size = (224,224)

try:
        os.makedirs(output_folder)
except:
        pass

imagelist = get_imagelist(input_folder)
images = load_images(imagelist)

data = np.array([])
flag = True


for img, imgname in zip(images, imagelist):
        crops = crop_image(img, crop_size)
        print(np.array(crops).shape)
        filename = imgname.split('/')[-1]

        if flag:
                data = crops
                flag = False
        else:
                data = np.append(data, crops, axis=0)

        save_crops(crops, output_folder+filename)


## save img 
input_folder = dir_img
output_folder = dir_img_aug
crop_size = (224,224)

try:
        os.makedirs(output_folder)
except:
        pass

imagelist = get_imagelist(input_folder)
images = load_images(imagelist)

data = np.array([])
flag = True


for img, imgname in zip(images, imagelist):
        crops = crop_image(img, crop_size)
        print(np.array(crops).shape)
        filename = imgname.split('/')[-1]

        if flag:
                data = crops
                flag = False
        else:
                data = np.append(data, crops, axis=0)

        save_crops(crops, output_folder+filename)



print(np.array(data).shape)

import os
import numpy as np 

## data augmentation
from keras_preprocessing.image import ImageDataGenerator


## manual generate data augmentation online
def data_generator_online(data_gen_args, X, y, batch_size):

    ## value based to retun predict_generator
    point_break = len(X)
    batches = 0

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    ## generate data
    ## for each loop generate one new batch (size_fold, width, heigth, channel)
    for x_batch, y_batch in zip(image_datagen.flow(X, None, batch_size=batch_size, shuffle=False, seed=seed), mask_datagen.flow(y, None, batch_size=batch_size, shuffle=False, seed=seed)):
        
        if batches == 0:
            X_batch_test = x_batch
            y_batch_test = y_batch
        else:
            X_batch_test = np.append(X_batch_test, x_batch, axis=0)
            y_batch_test = np.append(y_batch_test, y_batch, axis=0)

        batches += 1

        ## we need to break the loop by hand because
        ## the generator loops indefinitely
        if batches >= point_break:
            break

    return X_batch_test, y_batch_test

def data_generator_offline(data_gen_args, dir_img, dir_seg, batch_size, target_size):

    ## value based to retun predict_generator
    rawimg = np.array([f for f in os.listdir(dir_img + "data/") if(f.endswith('.png'))])
    point_break = len(rawimg)
    batches = 0

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen = image_datagen.flow_from_directory(
        directory=dir_img, 
        batch_size=batch_size, 
        shuffle=False, seed=seed, 
        target_size=target_size)
        
    mask_datagen = mask_datagen.flow_from_directory(
        directory=dir_seg, 
        batch_size=batch_size, 
        shuffle=False, 
        seed=seed, 
        target_size=target_size)

    ## generate data
    ## for each loop generate one new batch (size_fold, width, heigth, channel)
    while 1:
        
        x_batch, _ = next(image_datagen)
        y_batch, _ = next(mask_datagen)

        if batches == 0:
            X_batch_test = x_batch
            y_batch_test = y_batch
        else:
            X_batch_test = np.append(X_batch_test, x_batch, axis=0)
            y_batch_test = np.append(y_batch_test, y_batch, axis=0)

        batches += 1

        ## we need to break the loop by hand because
        ## the generator loops indefinitely
        if batches >= point_break:
            break


    ## generate data
    ## for each loop generate one new batch (size_fold, width, heigth, channel)
    # for x_batch, y_batch in zip(next(image_datagen), next(mask_datagen)):
        
    #     x_batch = x_batch[0]
    #     y_batch = y_batch[0]

    #     if batches == 0:
    #         X_batch_test = x_batch
    #         y_batch_test = y_batch
    #     else:
    #         X_batch_test = np.append(X_batch_test, x_batch, axis=0)
    #         y_batch_test = np.append(y_batch_test, y_batch, axis=0)

    #     batches += 1

    #     ## we need to break the loop by hand because
    #     ## the generator loops indefinitely
    #     if batches >= point_break:
    #         break


    return X_batch_test, y_batch_test
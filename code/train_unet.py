###############################
## imports
###############################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0";  

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

tf.Session(config=config)

### ignore warnings
import warnings

### garbage collector
import gc

###
## Links
## https://github.com/danielelic/deep-segmentation
## https://www.microsoft.com/developerblog/2018/07/18/semantic-segmentation-small-data-using-keras-azure-deep-learning-virtual-machine/

## Modelos - Image Segmentation Keras
## https://github.com/divamgupta/image-segmentation-keras
##
## Binary Segmentation
## https://github.com/danielelic/deep-segmentation
####

import numpy as np  

from skimage import io 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Seaborn is a Python data visualization library based on matplotlib
import seaborn as sns 

###############################
## define parameters
###############################

rede = "unet"

### camvid - 32 classes
### cropweed - 3 classes
### sugarbeets - 3 classes
### synthia - 12 classes
### cornspacing - 2 classes

from dataset_cropweed import load_data_resize, load_data
dataset = "cropweed_path_all_448"
n_classes = 3
target_names = ["Soil", "Weed", "Crop"]

#yes_train_rate = True ## not use
#train_rate = 0.8

width_image = height_image = 448
 
epochs = 500
batch_size = 32 #batch menor 

##############################################
## categorial_crossentropy to semantic segmentation multiclass
## softmax to semantic segmentation multiclass
##############################################
## binary_crossentropy to semantic segmentation binary class
## sigmoid function to classification pixel-wise
##############################################
## https://stats.stackexchange.com/questions/246287/regarding-the-output-format-for-semantic-segmentation
##############################################

loss = "categorical_crossentropy"
#optimizer = "adam"
optimizer = "adadelta"
#optimizer = "sgd"
#optimizer = "rmsprop"

## data augmentation    
crop_augmentation = True
data_augmentation = True


## report 
metrics = "jaccard_distance"

##############################################
## create directory to save results
##############################################

import time 
import os
import errno

cwd = os.getcwd()
print(cwd)

millis = int(round(time.time() * 1000))
dir_result = cwd + "/resultados/"+ dataset + "/" + str(millis) + "/"

def create_dir(path):
    ## criando o diretorio
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

create_dir(dir_result)
print(dir_result)

################################
## write parameters
################################

with open(dir_result + "parameters.txt", "w") as text_file:
        text_file.write("n_classes: " + str(n_classes))
        #text_file.write("\ntrain_rate: " + str(train_rate))
        text_file.write("\nwidth_image: " + str(width_image))
        text_file.write("\nheight_image: " + str(height_image))
        text_file.write("\nrede: " + str(rede))
        text_file.write("\nepochs: " + str(epochs))
        text_file.write("\nbatch_size: " + str(batch_size))
        text_file.write("\nloss: " + str(loss))
        text_file.write("\noptimizer: " + str(optimizer))
        text_file.write("\nmetrics: " + str(metrics))


################################
## load dataset
################################

print("Carregando dados ....")

#X_train, y_train = load_data_resize(width_image, height_image)
X_train, y_train = load_data()

print(X_train.shape)

################################
## load model CNN
################################

#from model_segunet import SEGUNET
#model = SEGUNET(n_classes, width_image, height_image)

from model_unet import UNET_MULTI_CLASS

## lib for load weights and networks
from keras.models import load_model

## data augmentation
from keras_preprocessing.image import ImageDataGenerator

## load networks
model = UNET_MULTI_CLASS(n_classes, width_image, height_image)

## show layers
print(model.summary)

################################
## training starts here
################################

from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping 

from keras_train_functions import data_generator_online
from crop import crop_generator_online

model.compile(loss=loss,
              optimizer=optimizer)      ####### change ######### 

#################################
## https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#################################

### cross validation kfolds
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False)
k = 0

## maximum number of epoch without gain
patience = 10

from report_functions import plot_loss_history, save_loss_history
from report_functions import save_confusion_matrix, save_json_file_report_classification, save_matrix_img_seg
from report_functions import save_balanced_accuracy, save_jaccard_index
from report_metrics import save_metrics

#############################
## Data Augmentation
## https://machinelearningmastery.com/image-augmentation-deep-learning-keras/ 
#############################   

## configuration to data augmentation
#datagen = ImageDataGenerator(
#                        horizontal_flip=True,
#                        vertical_flip=True,
#                        rotation_range=90,
#                        width_shift_range=0.1,
#                        height_shift_range=0.1,
#                        shear_range=0.2,
#                        zoom_range=0.2,
#                        fill_mode="constant",
#                        cval=0
#                    )

# we create two instances with the same arguments
data_gen_args = dict(
                    horizontal_flip=True,
                    vertical_flip=True,
                    rotation_range=90,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    fill_mode="constant",
                    cval=0
                    )

### trainning cnn in k folds
for train_index, test_index in kf.split(X_train):

    print('\nFold ', k)
    k = k + 1

    # split folds
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    # directory for save files the trainning
    dir_fold = dir_result + "model_kfold_" + str(k) + "/"
    create_dir(dir_fold)
    create_dir(dir_fold + "images/")
    print(dir_fold)

    # load model
    model = UNET_MULTI_CLASS(n_classes, width_image, height_image)

    # compile model
    model.compile(loss=loss, optimizer=optimizer) 

    if crop_augmentation:

        X_train_fold = crop_generator_online(X_train_fold, (width_image, height_image))
        y_train_fold = crop_generator_online(y_train_fold, (width_image, height_image))

        X_test_fold = crop_generator_online(X_test_fold, (width_image, height_image))
        y_test_fold = crop_generator_online(y_test_fold, (width_image, height_image))

    if data_augmentation: 

        X_train_fold, y_train_fold = data_generator_online(data_gen_args, X_train_fold, y_train_fold, batch_size)
        #X_test_fold, y_test_fold = data_generator_online(data_gen_args, X_test_fold, y_test_fold, batch_size)

    print(X_train_fold.shape, X_test_fold.shape)

    # name for networks pre trained save
    model_pre_trained = dir_fold + '/best_model_kfold_' + str(k) + "_" + str(rede) + '_' +  str(width_image) + '_' + str(dataset) + '_' + str(epochs) + '.h5'

    # callbacks
    checkpointer = ModelCheckpoint(filepath=model_pre_trained, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

    # fit the model
    hist = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_test_fold, y_test_fold),
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                callbacks=[checkpointer, earlystopper]
                )

    ###############################################
    ## Save history CNN training loss
    ###############################################

    save_loss_history(dir_fold + rede + "_loss_history.txt", hist)

    ###############################################
    ## Plot or Save the change in loss over epochs
    ###############################################

    plot_loss_history(hist, dir_fold + rede + "_history.png")

    ###############################################
    ## Prepare preditions to report
    ###############################################

    ### predict over test_fold to extract report
    y_pred = model.predict(X_test_fold)
    y_predi = np.argmax(y_pred, axis=3)
    y_testi = np.argmax(y_test_fold, axis=3)

    ## number images
    size_test = len(y_pred) 
    
    ### preditions for calcule value geral trainning
    if k == 1:
        preditions = y_pred
        targets = y_test_fold
    else:
        preditions = np.append(preditions, y_pred, axis=0)
        targets = np.append(targets, y_test_fold, axis=0)

    ### reshap for unidimensional
    y_predicted = np.reshape(y_predi, width_image * height_image * size_test)
    y_target = np.reshape(y_testi, width_image * height_image * size_test)

    ###############################################
    ## Report Classification
    ###############################################

    save_json_file_report_classification(dir_fold, y_target, y_predicted, target_names)

    ###############################################
    ## Plot Confusion Matrix
    ###############################################

    save_confusion_matrix(dir_fold + rede + "_confusion_matrix.png", y_target, y_predicted, target_names)

    ###############################################
    ## Visualize the model performance
    ###############################################

    save_matrix_img_seg(dir_fold, X_test_fold, y_testi, y_predi, n_classes)

    ###############################################
    ## Garbage Collection
    ###############################################
    del(y_pred)
    del(y_predi) 
    del(y_testi)
    del(X_train_fold)
    del(y_train_fold)
    del(X_test_fold)       
    del(y_test_fold) 
    gc.collect()

###############################################
## Prepare preditions to report
###############################################

preditions = np.argmax(preditions, axis=3)
labels = np.argmax(targets, axis=3)
size_total = len(targets)

preditions = np.reshape(preditions, width_image * height_image * size_total)
labels = np.reshape(labels, width_image * height_image * size_total)

###############################################
## Report Classification
###############################################

save_json_file_report_classification(dir_result, labels, preditions, target_names)

###############################################
## Plot Confusion Matrix
###############################################

save_confusion_matrix(dir_result + rede + "_confusion_matrix.png", labels, preditions, target_names)

###############################################
## Extract Extra Metrics
###############################################

save_balanced_accuracy(dir_result + rede + "_extra_metrics.txt", labels, preditions)
save_jaccard_index(dir_result + rede + "_extra_metrics.txt", labels, preditions)


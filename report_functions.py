import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

###############################################
## Save history CNN training loss
###############################################

def save_loss_history(file_path, hist):

    loss_history = hist.history["loss"]
    np.savetxt(file_path, np.array(loss_history), delimiter=",")

    total_epochs = len(loss_history)

    with open(file_path, "a") as f:
        f.write("\nTotal epochs: {}".format(total_epochs))

###############################################
## Plot and Save history CNN training loss
## https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
## https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
## https://chrisalbon.com/deep_learning/keras/visualize_loss_history/
###############################################

def plot_loss_history(history, name_file):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    #acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    #val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'r--', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'b-', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    #plt.figure(2)
    #for l in acc_list:
    #    plt.plot(epochs, history.history[l], 'b', label='Training jaccard index (' + str(format(history.history[l][-1],'.5f'))+')')
    #for l in val_acc_list:    
    #    plt.plot(epochs, history.history[l], 'g', label='Validation jaccard index (' + str(format(history.history[l][-1],'.5f'))+')')

    #plt.title('Jaccard Index')
    #plt.xlabel('Epochs')
    #plt.ylabel('Jaccard Index')
    #plt.legend()
    plt.savefig(name_file)
    plt.clf()

def plot_jaccard_history(history, name_file):

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(acc_list) == 0:
        print('Jaccard index is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1, len(history.history[acc_list[0]]) + 1)
    
    ## Jaccard Index
    plt.figure(1)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'r--', label='Training jaccard index (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'b-', label='Validation jaccard index (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Model jaccard index')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard index')
    plt.legend()
    
    plt.savefig(name_file)
    plt.clf()


###############################################
## Plot Confusion Matrix
## http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/
## https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
###############################################

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

def save_confusion_matrix(file_path, y_target, y_predicted, target_names=None, binary=False):
    
    cm = confusion_matrix(y_target, y_predicted, binary)

    fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=False, show_normed=True)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    plt.savefig(file_path)

###############################################
## Balanced Accuracy
## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score
## best value is 1
###############################################

from sklearn.metrics import balanced_accuracy_score

def save_balanced_accuracy(file_path, y_target, y_predicted):

    balanced_accuracy = balanced_accuracy_score(y_target, y_predicted)
    balanced_accuracy = round(balanced_accuracy * 100, 2)

    with open(file_path, "a") as f:
        f.write("\nBalanced Accuracy: {}".format(balanced_accuracy))

###############################################
## Jaccard Index
## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score
## best value is 1 with nomalize=True
###############################################

from sklearn.metrics import jaccard_score

def save_jaccard_index(file_path, y_target, y_predicted):

    jaccard_index = jaccard_score(y_target, y_predicted, normalize=True)
    jaccard_index = round(jaccard_index * 100, 2)

    with open(file_path, "a") as f:
        f.write("\nJaccard Index: {}".format(jaccard_index))    

###############################################
## Report Classification
###############################################

import json

from sklearn.metrics import classification_report

def save_json_file_report_classification(dir_path, y_true, y_pred, target_names):

    c_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names, output_dict=True)

    with open(dir_path + "c_report.json", "w") as f:
        json.dump(c_report, f)

##############################################
## convert index for label
##############################################

def convert_index_to_label(index, dict_label):

    return False

###############################################
## Visualize the model performance
###############################################

import random

# Seaborn is a Python data visualization library based on matplotlib
import seaborn as sns

## 
import warnings
from skimage import io 

## removendo a grade branca do seaborn
sns.set_style("whitegrid", { "axes.grid": False})

def give_color_to_seg_img(seg, n_classes):

    if len(seg.shape) == 3:
        seg = seg[:,:,0]
    seg_img = np.zeros( (seg.shape[0],seg.shape[1], 3) ).astype('float')
    
    if n_classes == 2:
        flatui = ["#FFFFFF", "#000000"] # white and black # binary segmentation
        sns.set_palette(flatui)
        colors = sns.color_palette()
    else:
        #colors = sns.color_palette("hls", n_classes)
        flatui = ["#000000", "#FE0203", "#03FE06"] #black #red #green #soil #weed #crop 
        sns.set_palette(flatui)
        colors = sns.color_palette()
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc * ( colors[c][0] ))
        seg_img[:,:,1] += (segc * ( colors[c][1] ))
        seg_img[:,:,2] += (segc * ( colors[c][2] ))

    return(seg_img)

def save_matrix_img_seg(dir_path, X_test, y_true, y_pred, n_classes):

        size = len(X_test)

        ### ignore warning in loop to save images
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for i in range(0, size):
                img_is  = ((X_test[i] + 1) * (255.0 / 2)) ## sub_and_divide
                seg = y_pred[i]
                segtest = y_true[i]

                ##########
                ## save imagens resultados separadamente
                ##########
                io.imsave(dir_path + "images/original_image" + str(i + 1) + ".png", (img_is / 255.0))
                io.imsave(dir_path + "images/pred_image" + str(i + 1) + ".png", give_color_to_seg_img(seg, n_classes))
                io.imsave(dir_path + "images/mask_image" + str(i + 1) + ".png", give_color_to_seg_img(segtest, n_classes))

                fig = plt.figure(figsize=(15,5)) 

                ax = fig.add_subplot(1,3,1)
                ax.imshow(img_is/255.0)
                ax.set_title("original")
                
                ax = fig.add_subplot(1,3,2)
                ax.imshow(give_color_to_seg_img(seg, n_classes))
                ax.set_title("predicted class")
                
                ax = fig.add_subplot(1,3,3)
                ax.imshow(give_color_to_seg_img(segtest, n_classes))
                ax.set_title("true class")
            
                plt.tight_layout()
                plt.savefig(dir_path + "result_" + str(i) + ".png")

def save_imgs(dir_path, images):

        for i in range(0, len(images)):
            io.imsave(dir_path + "img" + str(i + 1) + ".png", images[i])


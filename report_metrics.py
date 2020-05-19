import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

###############################################
## SEN, SPE, DIC, JAC, ACC
## https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
###############################################

from sklearn.metrics import confusion_matrix

def get_metrics(y_true, y_pred):

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    print("TP ", TP, "FP ", FP, "TN ", TN, "FN ", FN)

    SEN = TP / ( TP + FN )
    SPE = TN / ( TN + FP )
    DIC = 2 * TP / ( (2 * TP) + FP + FN )
    JAC = TP / ( TP + FN + FP )
    ACC = ( TP + TN ) / ( TP + FN + TN + FP )
    
    return SEN, SPE, DIC, JAC, ACC

def save_metrics(file_path, y_target, y_predicted):

    SEN, SPE, DIC, JAC, ACC = get_metrics(y_target, y_predicted)

    with open(file_path, "a") as f:
        f.write("\n ==================")
        f.write("\nSen: {}".format(SEN))
        f.write("\nSpe: {}".format(SPE)) 
        f.write("\nDic: {}".format(DIC)) 
        f.write("\nJac: {}".format(JAC)) 
        f.write("\nAcc: {}".format(ACC))  
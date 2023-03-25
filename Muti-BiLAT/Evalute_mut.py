# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:27:44 2021

@author: Administrator
"""
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import cohen_kappa_score,brier_score_loss

import os, sys
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix





def Classification_result(CDKs_Name,y_true,y_pred):
    acc = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred, average='macro')
    recall = recall_score(y_true,y_pred)
    mcc = matthews_corrcoef(y_true,y_pred)
    kappa = cohen_kappa_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred, average='macro')
    print('{}:准确度:{}, 精确度：{},\n 召回率：{}, 马修斯系数：{},\n Kappa:{}, F1 score:{} '.
      format(CDKs_Name,acc,precision,recall, mcc, kappa,f1))
    print('-------------------------------------')
    return None




def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
　　 cm:混淆矩阵值
　　 classes:分类标签
　　 """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    class_names = ['disactivated', 'activated']
    class_names = np.array(class_names) 
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
    np.set_printoptions(precision=2)
    plt.figure()
    plt.show()
    
    
    
def plot_roc_curve(fpr, tpr,title):
    plt.figure()
    lw = 2
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, color='darkorange',lw=lw, 
              label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()  


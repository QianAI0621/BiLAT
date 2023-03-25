# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:30:08 2021

@author: Administrator
"""
import os, sys
import tensorflow as tf
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import cohen_kappa_score,brier_score_loss
from sklearn.metrics import confusion_matrix
from Evalute_mut import Classification_result, plot_confusion_matrix, plot_roc_curve



tf.compat.v1.disable_eager_execution()
# load data =========================================
##读取特征值——摩根指纹
print('start loading data')
dataX = np.load('CDKs_2048.npy',allow_pickle = True)
##读取标签值
dataY  = np.load('CDKs_Y.npy',allow_pickle = True)
# print(dataY)
print('loading data is done!')
# ===================================================

# from  tensorflow.keras.utils import to_categorical

#划分数据集：
train_x_concat, te_x_concat, train_y_concat, te_y_concat = [], [], [], []

for k in range(len(dataX)):
    train_x, test_x, train_y, test_y = train_test_split(dataX[k], dataY[k], test_size=0.4,random_state=212)
    
    train_x_concat.append(train_x)
    te_x_concat.append(test_x)
    train_y_concat.append(train_y)
    te_y_concat.append(test_y)
    # print(train_x.shape, train_y.shape)
    # print(test_x.shape, test_y.shape) 


test_x_concat ,val_x_concat,test_y_concat , val_y_concat = [],[],[],[]
for k in range(len(te_x_concat)):
    test_x, val_x, test_y, val_y = train_test_split(te_x_concat[k], te_y_concat[k], test_size=0.5,random_state=212)
    
    test_x_concat.append(test_x)
    val_x_concat.append(val_x)

    test_y_concat.append(test_y)
    val_y_concat.append(val_y)
    print(test_x.shape, test_x.shape)
    print(val_x.shape, val_y.shape) 
    
    



####超参数优化
Max_length = 2048  #DNN 特征数目
lr = 0.001   #学习率

#数据占位
# X = tf.placeholder(tf.int32, [None, 1024])
# Y = tf.placeholder(tf.float32, [None, 2])

X = tf.compat.v1.placeholder(tf.float32, [None, Max_length])
Y = tf.compat.v1.placeholder(tf.int32, [None,1])

#2...构建模型
#初始化 权重
# def init_weights(shape):
#     return tf.Variable(tf.random_normal(shape, stddev = 0.01))
def init_weights(shape):
    return tf.Variable(tf.compat.v1.random.normal(shape, stddev = 0.01))
#初始化 偏置
def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

###层数建立
#############共享层1
w_a = init_weights([2048,1024])
b_a =  bias_variable([1024])
py_x_a = tf.sigmoid(tf.matmul(X, w_a) + b_a)

# py_x1 = tf.matmul(X, w1) + b1
# py_x2 = tf.matmul(X, w2) + b2
#############共享层2
w_b = init_weights([1024,560])
b_b = bias_variable([560])
py_x_b = tf.sigmoid(tf.matmul(py_x_a, w_b) + b_b)

#############共享层3
w_c = init_weights([560,250])
b_c = bias_variable([250])
py_x_c = tf.sigmoid(tf.matmul(py_x_b, w_c) + b_c)

#############共享层4
w_d = init_weights([250,64])
b_d = bias_variable([64])
py_x_d = tf.sigmoid(tf.matmul(py_x_c, w_d) + b_d)

##############输出层
w1 = init_weights([64,1])
w2 = init_weights([64,1])
w3 = init_weights([64,1])
w4 = init_weights([64,1])
w5 = init_weights([64,1])
w6 = init_weights([64,1])
w7 = init_weights([64,1])
w8 = init_weights([64,1])

b1 = bias_variable([1])
b2 = bias_variable([1])
b3 = bias_variable([1])
b4 = bias_variable([1])
b5 = bias_variable([1])
b6 = bias_variable([1])
b7 = bias_variable([1])
b8 = bias_variable([1])

py_x1 = tf.sigmoid(tf.matmul(py_x_d, w1) + b1)
py_x2 = tf.sigmoid(tf.matmul(py_x_d, w2) + b2)
py_x3 = tf.sigmoid(tf.matmul(py_x_d, w3) + b3)
py_x4 = tf.sigmoid(tf.matmul(py_x_d, w4) + b4)
py_x5 = tf.sigmoid(tf.matmul(py_x_d, w5) + b5)
py_x6 = tf.sigmoid(tf.matmul(py_x_d, w6) + b6)
py_x7 = tf.sigmoid(tf.matmul(py_x_d, w7) + b7)
py_x8 = tf.sigmoid(tf.matmul(py_x_d, w8) + b8)



#3..构建损失
# cost1 = tf.losses.log_loss(labels=Y, predictions=py_x1)
# cost2 = tf.losses.log_loss(labels=Y, predictions=py_x2)
    
cost1 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x1)
cost2 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x2)
cost3 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x3)
cost4 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x4)
cost5 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x5)
cost6 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x6)
cost7 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x7)
cost8 = tf.compat.v1.losses.log_loss(labels=Y, predictions=py_x8)


#4..优化器
# train_op1 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost1)
# train_op2 = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost2)

train_op1 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost1)
train_op2 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost2)
train_op3 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost3)
train_op4 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost4)
train_op5 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost5)
train_op6 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost6)
train_op7 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost7)
train_op8 = tf.compat.v1.train.AdamOptimizer(learning_rate = lr).minimize(cost8)

prediction_error1 = cost1
prediction_error2 = cost2
prediction_error3 = cost3
prediction_error4 = cost4
prediction_error5 = cost5
prediction_error6 = cost6
prediction_error7 = cost7
prediction_error8 = cost8

#初始化变量
# int = tf.global_variables_initializer()

init = tf.compat.v1.global_variables_initializer()


#模型保存
SAVER_DIR = "model_CDKs_2048"
saver = tf.compat.v1.train.Saver()
ckpt_path = os.path.join(SAVER_DIR,"model_CDKs_2048")  #SAVER_DIR,
ckpt = tf.train.get_checkpoint_state(ckpt_path)
 
#开启会话
with tf.compat.v1.Session() as sess:
    
    sess.run(init)
    best_auc = 0
    best_idx = 0
    for i in range(60):
        training_batch = zip(range(0, len(train_x_concat[0]), 28),
                             range(28, len(train_x_concat[0])+1, 28))
        for start, end in training_batch:
            sess.run([train_op1,cost1], feed_dict={X: train_x_concat[0][start:end], Y: train_y_concat[0][start:end]})
            sess.run([train_op2,cost2], feed_dict={X: train_x_concat[1][start:end], Y: train_y_concat[1][start:end]})
            sess.run([train_op3,cost3], feed_dict={X: train_x_concat[2][start:end], Y: train_y_concat[2][start:end]})
            sess.run([train_op4,cost4], feed_dict={X: train_x_concat[3][start:end], Y: train_y_concat[3][start:end]})
            sess.run([train_op5,cost5], feed_dict={X: train_x_concat[4][start:end], Y: train_y_concat[4][start:end]})
            sess.run([train_op6,cost6], feed_dict={X: train_x_concat[5][start:end], Y: train_y_concat[5][start:end]})
            sess.run([train_op7,cost7], feed_dict={X: train_x_concat[6][start:end], Y: train_y_concat[6][start:end]})
            sess.run([train_op8,cost8], feed_dict={X: train_x_concat[7][start:end], Y: train_y_concat[7][start:end]})

    # print test loss   
        merr = sess.run(prediction_error1, feed_dict={X: test_x_concat[0], Y: test_y_concat[0]})
        print(i, merr,  end = ' ')
        merr = sess.run(prediction_error2, feed_dict={X: test_x_concat[1], Y: test_y_concat[1]})
        print(merr, end = ' ')
        merr = sess.run(prediction_error3, feed_dict={X: test_x_concat[2], Y: test_y_concat[2]})
        print(merr, end = ' ')
        merr = sess.run(prediction_error4, feed_dict={X: test_x_concat[3], Y: test_y_concat[3]})
        print(merr, end = ' ')
        merr = sess.run(prediction_error5, feed_dict={X: test_x_concat[4], Y: test_y_concat[4]})
        print(merr, end = ' ')
        merr = sess.run(prediction_error6, feed_dict={X: test_x_concat[5], Y: test_y_concat[5]})
        print(merr, end = ' ')
        
        merr = sess.run(prediction_error7, feed_dict={X: test_x_concat[6], Y: test_y_concat[6]})
        print(merr, end = ' ')
        merr = sess.run(prediction_error8, feed_dict={X: test_x_concat[7], Y: test_y_concat[7]})
        print(merr, end = ' ')
    
    # calculate auc 
        test_preds1 = sess.run(py_x1, feed_dict={X: test_x_concat[0]})
        y_preds1=[round(test_preds1[i][0]) for i in range(len(test_preds1))]
        
        test_preds2 = sess.run(py_x2, feed_dict={X: test_x_concat[1]})
        y_preds2=[round(test_preds2[i][0]) for i in range(len(test_preds2))]
        
        test_preds3 = sess.run(py_x3, feed_dict={X: test_x_concat[2]})
        y_preds3=[round(test_preds3[i][0]) for i in range(len(test_preds3))]
        
        test_preds4 = sess.run(py_x4, feed_dict={X: test_x_concat[3]})
        y_preds4=[round(test_preds4[i][0]) for i in range(len(test_preds4))]
        
        test_preds5 = sess.run(py_x5, feed_dict={X: test_x_concat[4]})
        y_preds5=[round(test_preds5[i][0]) for i in range(len(test_preds5))]
        
        test_preds6 = sess.run(py_x6, feed_dict={X: test_x_concat[5]})
        y_preds6=[round(test_preds6[i][0]) for i in range(len(test_preds6))]
        
        test_preds7 = sess.run(py_x7, feed_dict={X: test_x_concat[6]})
        y_preds7=[round(test_preds7[i][0]) for i in range(len(test_preds7))]
        
        test_preds8 = sess.run(py_x8, feed_dict={X: test_x_concat[7]})
        y_preds8=[round(test_preds8[i][0]) for i in range(len(test_preds8))]
        
        
        
        test_aucs1 = roc_auc_score(test_y_concat[0], test_preds1)
        test_aucs2 = roc_auc_score(test_y_concat[1], test_preds2)
        test_aucs3 = roc_auc_score(test_y_concat[2], test_preds3)
        test_aucs4 = roc_auc_score(test_y_concat[3], test_preds4)
        test_aucs5 = roc_auc_score(test_y_concat[4], test_preds5)
        test_aucs6 = roc_auc_score(test_y_concat[5], test_preds6)
        test_aucs7 = roc_auc_score(test_y_concat[6], test_preds7)
        test_aucs8 = roc_auc_score(test_y_concat[7], test_preds8)
        
        test_aucs = [test_aucs1, test_aucs2,test_aucs3,test_aucs4,test_aucs5,test_aucs6,test_aucs7,test_aucs8]
                                        
        
        
        print('mean test auc: ', end = ' ')
        print(np.mean(test_aucs))

        if best_auc < np.mean(test_aucs):
            auc1 = test_aucs1
            auc2 = test_aucs2
            auc3 = test_aucs3
            auc4 = test_aucs4
            auc5 = test_aucs5
            auc6 = test_aucs6
            auc7 = test_aucs7
            auc8 = test_aucs8
            

            best_auc = np.mean(test_aucs)
            best_idx = i
            save_path = saver.save(sess, ckpt_path, global_step = best_idx)
            print('model saved!')
            print()

print('best epoch index: '+str(best_idx))
print('best test auc total: '+str(best_auc))
print('best test auc CDK1: '+str(auc1))
print('best test auc CDK2: '+str(auc2))
print('best test auc CDK4: '+str(auc3))
print('best test auc CDK5: '+str(auc4))
print('best test auc CDK6: '+str(auc5))
print('best test auc CDK7: '+str(auc6))
print('best test auc CDK8: '+str(auc7))
print('best test auc CDK9: '+str(auc8))





########## 分类结果

CDKs_Name = ['CDK1','CDK2','CDK4','CDK5','CDK6',"CDK7","CDK8",'CDK9']
y_preds_list = [y_preds1,y_preds2,y_preds3,y_preds4,y_preds5,y_preds6,y_preds7,y_preds8]
test_preds_list = [test_preds1, test_preds2,test_preds3,test_preds4,test_preds5,test_preds6,test_preds7,test_preds8]


for i in range (len(dataX)):
    Classification_result (CDKs_Name[i],test_y_concat[i],y_preds_list[i])
    ####混淆矩阵
    cnf_matrix = confusion_matrix(test_y_concat[i], y_preds_list[i])
    plot_confusion_matrix(cnf_matrix, title = str(CDKs_Name[i]))
    ####ROC曲线
    fpr, tpr, _ = roc_curve(test_y_concat[i], test_preds_list[i])
    plot_roc_curve(fpr, tpr,str(CDKs_Name[i]))


'''
####################################################################
#=========================== test part ============================#
####################################################################

def get_XY(file):   
    trfile = open(file , 'r')
    line = trfile.readline() #读取文件第一行，指针导向下一行，略过标题行 ！
    dataX_cdk = []
    dataY_cdk = []
    for i, line in enumerate(trfile):
        line = line.rstrip().split(',') #去除尾部空值
        smiles = str(line[1])
        # mol = Chem.MolFromSmiles(smiles)
        # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        mol = Chem.MolFromSmiles(smiles)
        fp= AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        npfp = np.array(list(fp.ToBitString())).astype('int8')
        dataX_cdk.append(npfp) 
    
        # label = float(line[2])
        # dataY_cdk.append(label)    
    
    dataX_cdk=np.array(dataX_cdk) 
    dataY_cdk=np.array(dataY_cdk)
    trfile.close()

    return(dataX_cdk ,dataY_cdk)

file = './M_Similarity_3s.csv'
XY = get_XY(file)
x,y = XY[0],XY[1]


print(x.shape)

saver = tf.compat.v1.train.Saver()
ckpt_path = os.path.join(SAVER_DIR, "model_CDKs_2048")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("model loaded successfully!")
    

    preds1 = sess.run(py_x1, feed_dict={X:x})
    preds2 = sess.run(py_x2, feed_dict={X:x})
    preds3 = sess.run(py_x3, feed_dict={X:x})
    preds4 = sess.run(py_x4, feed_dict={X:x})
    preds5 = sess.run(py_x5, feed_dict={X:x})
    preds6 = sess.run(py_x6, feed_dict={X:x})
    
    pred_cdk1_int=[round(preds1[i][0]) for i in range(len(preds1))]
    pred_cdk2_int=[round(preds2[i][0]) for i in range(len(preds2))]
    pred_cdk4_int=[round(preds3[i][0]) for i in range(len(preds3))]
    pred_cdk5_int=[round(preds4[i][0]) for i in range(len(preds4))]
    pred_cdk6_int=[round(preds5[i][0]) for i in range(len(preds5))]
    pred_cdk9_int=[round(preds6[i][0]) for i in range(len(preds6))]
    
    


CDKs_Name_test = [pred_cdk1_int,pred_cdk2_int,pred_cdk4_int,pred_cdk5_int,pred_cdk6_int,pred_cdk9_int]

print("总预测样本数：{}".format(len(pred_cdk1_int)))

for i in range(len(CDKs_Name_test)):
    pre = CDKs_Name_test[i]
    totle = []
    same = []  
    dif = []
    for smi in range(len(x)): 
        totle.append(smi)
        if pre[smi] == 0:
            # print(i)
            same.append(smi)
        else:
            dif.append(smi)
    
    print("模型预测对的样本数量：{}".format(len(same)))
    print("准确度为：{:.2f}".format(len(same)/len(totle)))
'''
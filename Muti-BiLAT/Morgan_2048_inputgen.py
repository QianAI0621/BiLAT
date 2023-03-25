# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:17:46 2021

@author: YaoDaBaoanChuZhang
"""

import tensorflow as tf
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem


tf.compat.v1.set_random_seed(2)
tf.compat.v1.reset_default_graph()


tf.compat.v1.disable_eager_execution()

#1..准备数据

##########将每一个数据集形成一个array形式并嵌套在列表中。
def get_XY(file):
    dataX_concat = []
    dataY_concat = []
    for i in range(len(file)):
        
        trfile = open(file[i] , 'r')
        line = trfile.readline() #读取文件第一行，指针导向下一行，略过标题行 ！
        dataX_cdk = []
        dataY_cdk = []
        for i, line in enumerate(trfile):
            line = line.rstrip().split(',') #去除尾部空值
            smiles = str(line[2])
            # mol = Chem.MolFromSmiles(smiles)
            # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            mol = Chem.MolFromSmiles(smiles)
            fp= AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            npfp = np.array(list(fp.ToBitString())).astype('int8')
            dataX_cdk.append(npfp) 
        
            label = float(line[1])
            dataY_cdk.append(label)    
        
        dataX_cdk=np.array(dataX_cdk) 
        dataY_cdk=np.array(dataY_cdk)
        trfile.close()
        dataX_concat.append(dataX_cdk)   
        dataY_concat.append(dataY_cdk)
    return(dataX_concat ,dataY_concat)


# file = './CDK1.csv','./CDK2.csv','./CDK4.csv','./CDK5.csv','./CDK6.csv','./CDK9.csv'

file =  './CDK9.csv','./CDK2.csv','./CDK4.csv','./CDK5.csv','./CDK6.csv','./CDK7.csv','./CDK8.csv' ,'./CDK9.csv' 
##"./CDK7.CSV","./CDK8.CSV",
XY = get_XY(file)
dataX,dataY = XY[0],XY[1]

from  tensorflow.keras.utils import to_categorical

# Y = []
# for i in range(len(dataY)):
#     Y_=np.vstack(dataY[i]).reshape(-1,1)
#     # data_Y = to_categorical(Y_)
#     # data_Y = data_Y[:,1:]
#     Y.append(data_Y)    

dataX = np.array(dataX)
np.save('CDKs_1024.npy', dataX)
dataY = np.array(dataY)
np.save('CDKs_Y.npy', dataY)

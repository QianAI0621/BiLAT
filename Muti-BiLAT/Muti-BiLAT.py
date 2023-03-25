from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time
import numpy as np
import os
import argparse
#from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import itertools
from preprocessing import randomize_smile
import pandas as pd

import pandas as pd


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score

log_path =  "./logs/"
try:
    os.rmdir(log_path)
except OSError as e:
    print("Error: %s : %s" % (log_path, e.strerror))
    TC = tf.keras.callbacks.TensorBoard("logs", 2, write_graph=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
print("current pid:", os.getpid())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("should be ok...right?")
    except RuntimeError as e:
        print(e)
else:
    print("gpu unlimited?")


tasks = ["CDK1","CDK2","CDK4","CDK5","CDK6","CDK7","CDK8","CDK9"]
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--batchsize', '-b', type=int, default=200, help='Number of moleculars in each mini-batch')
parser.add_argument('--epochs', '-e', type=int, default=40, help='Number of sweeps over the dataset to train')
parser.add_argument('--input', '-i', default='./CDK', help='Input SDFs Dataset')
parser.add_argument('--num_layers', type=int, default=2, help='No. of hidden perceptron')
parser.add_argument('--d_model', type=int, default=128, help='No. of hidden perceptron')  # default 512
parser.add_argument('--dff', type=int, default=1024, help='No. of hidden perceptron')
parser.add_argument('--num_heads', type=int, default= 1 , help='No. of hidden perceptron')
parser.add_argument('--dropout_rate', '-d', type=float, default=0.5, help='No. of hidden perceptron')
parser.add_argument('--lr', '-l', type=float, default=0.0005, help='No. of hidden perceptron')
parser.add_argument('--max_vocab_size', type=int, default=1025, help='')
parser.add_argument('--atomsize', '-c', type=int, default=200, help='max length of smiles')
parser.add_argument('--seq_size', '-s', type=int, default=150, help='seq length of smiles fp2vec')
parser.add_argument('--pickle_load', type=bool, default=False, help='pickle embedding')
parser.add_argument('--weight_load', type=bool, default=False, help='pickle embedding')
parser.add_argument('--_test', type=bool, default=False, help='pickle embedding')#True
parser.add_argument('--current_num', default="for3", help='name says it')
parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class)')
parser.add_argument('--num_randoms', type=int, default=5, help='No. of output perceptron (class)')
parser.add_argument('--random_state', type=int, default=69, help='No. random_state split')



def score_re( y_val_list,y_pred_list,tasks):

    eval_roc_auc_score = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    
    eval_accuracy_score =  [accuracy_score(y_val_list[i],
                                      (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    
    eval_precision_score = [precision_score(y_val_list[i],
                                      (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    
    #eval_prc = [auc(precision_recall_curve(y_val_list[i], y_pred_list[i])[1],precision_recall_curve(y_val_list[i], y_pred_list[i])[0]) for i in range(len(tasks))]
    eval_f1_score = [f1_score(y_val_list[i],
                                (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    
    
    eval_recall_score = [recall_score(y_val_list[i],
                                (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    
    eval_matthews_corrcoef =  [matthews_corrcoef(y_val_list[i],
                                      (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    
    eval_cohen_kappa_score =  [cohen_kappa_score(y_val_list[i],
                                      (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    Score = pd.DataFrame(columns=["AUC","ACC","Pre","F1","Recall","Mcc","Kapppa"],index = tasks)
    Score["AUC"] = eval_roc_auc_score
    Score["ACC"] = eval_accuracy_score
    Score["Pre"] = eval_precision_score
    Score["F1"] = eval_f1_score
    Score["Recall"] = eval_recall_score
    Score["Mcc"] = eval_matthews_corrcoef
    Score["Kapppa"] = eval_cohen_kappa_score
    
    return Score


def posNegNums(ydata):
    cntP = 0
    cntN = 0
    for ele in ydata:
        if ele == 1:
            cntP += 1
        else:
            cntN += 1
    return cntP, cntN

# detaset function definition
def random_list(x, seed=0):
    np.random.seed(seed)
    np.random.shuffle(x)


def char2indices(listStr, dicC2I):
    listIndices = [0] * args.seq_size
    charlist = listStr
    size = len(listStr)
    twoChars = {"Al": 1, "Au": 1, "Ag": 1, "As": 1, "Ba": 1, "Be": 1, "Bi": 1, "Br": 1, "Ca": 1, "Cd": 1, "Cl": 1,
                "Co": 1, "Cr": 1, "Cu": 1, "Dy": 1, "Fe": 1, "Gd": 1, "Ge": 1, "In": 1, "Li": 1, "Mg": 1, "Mn": 1,
                "Mo": 1, "Na": 1, "Ni": 1, "Nd": 1, "Pb": 1, "Pt": 1, "Pd": 1, "Ru": 1, "Sb": 1, "Se": 1, "se": 1,
                "Si": 1, "Sn": 1, "Sr": 1, "Ti": 1, "Tl": 1, "Yb": 1, "Zn": 1, "Zr": 1}
    prevTwoCharsFlag = False
    indexForList = 0
    for i, c in enumerate(charlist):
        if prevTwoCharsFlag:
            prevTwoCharsFlag = False
            continue
        
        if i != size - 1 and "".join(charlist[i:i+2]) in twoChars:
            two = "".join(charlist[i:i+2])
            if two not in dicC2I:
                dicC2I[two] = len(dicC2I) + 1
                listIndices[indexForList] = dicC2I[two]
                indexForList += 1
            else:
                listIndices[indexForList] = dicC2I[two]
                indexForList += 1
            prevTwoCharsFlag = True
        else:    
            if c not in dicC2I:
                dicC2I[c] = len(dicC2I) + 1
                listIndices[indexForList] = dicC2I[c]
                indexForList += 1
            else:
                listIndices[indexForList] = dicC2I[c]
                indexForList += 1
    return listIndices


def makeDataForSmilesOnly(proteinName, dicC2I):
    print(proteinName)
    listX, listY = [], []
    afile = args.input + '/' + proteinName + '.csv'
    
    file = pd.read_csv(afile)
    
    X = file["Smiles"]
    
    Y = file["Label"]
    
    train,val_ = train_test_split(file[["Smiles","Label"]],random_state=args.random_state,test_size=0.2,stratify=(file["Label"]))
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y,random_state=args.random_state,test_size=0.5,stratify=(Y))
    
    val_,test_= train_test_split(val_,random_state=args.random_state,test_size=0.5,stratify=(val_["Label"]))
    
    # valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y,random_state=212,test_size=0.5,stratify=(valid_y))
    
    if args.aug_num > 1:
        train_temp = pd.concat([train] * (args.aug_num - 1), axis=0)
        train_temp["Smiles"]= train_temp["Smiles"].map(lambda x: randomize_smile(x))
        train = pd.concat([train_temp, train], ignore_index=True)
        train_x = [char2indices(i, dicC2I) for i in train["Smiles"] ]
        train_y = [float(i) for i in train["Label"] ]
        
    else:
        train_x = [char2indices(i, dicC2I) for i in train_x] # length can vary
        train_y = [float(i) for i in train_y ]
        
    if args.aug_num > 1:
            val_temp = pd.concat([val_] * (args.aug_num - 1), axis=0)
            val_temp["Smiles"]= val_temp["Smiles"].map(lambda x: randomize_smile(x))
            val_ = pd.concat([val_temp, val_], ignore_index=True)
            valid_x = [char2indices(i, dicC2I) for i in val_["Smiles"] ]
            valid_y = [float(i) for i in val_["Label"] ]
            
    else:
            valid_x = [char2indices(i, dicC2I) for i in valid_x] 
            valid_y = [float(i) for i in valid_y ]

    test_x = [char2indices(i, dicC2I) for i in test_["Smiles"].values]
    


    test_y = [float(i) for i in test_["Label"].values ]
        
    pos_num, neg_num = posNegNums(train_y)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
    print(proteinName+"done")
    return train_tf, valid_tf, test_tf, pos_num, neg_num, train_x, valid_x, test_x#testlistX


if args.pickle_load:
    embeddings_, dicC2I = pickle.load(open(args.current_num+"saved_emb.pkl", "rb"))
else:
    dicC2I = {}
pos_num, neg_num = 0, 0


if not args._test:
    train_tf1, valid_tf1, test_tf1, pos1, neg1, _, _, _ = makeDataForSmilesOnly("CDK1", dicC2I)
    train_tf2, valid_tf2, test_tf2, pos2, neg2, _,_,_,_ = makeDataForSmilesOnly("CDK2", dicC2I)
    train_tf3, valid_tf3, test_tf3, pos3, neg3, _, _, _ = makeDataForSmilesOnly("CDK4", dicC2I )
    train_tf4, valid_tf4, test_tf4, pos4, neg4, _, _, _ = makeDataForSmilesOnly("CDK5", dicC2I)
    train_tf5, valid_tf5, test_tf5, pos5, neg5, _, _, _ = makeDataForSmilesOnly("CDK6", dicC2I )
    train_tf6, valid_tf6, test_tf6, pos6, neg6, _, _, _ = makeDataForSmilesOnly("CDK7", dicC2I)
    train_tf7, valid_tf7, test_tf7, pos7, neg7, _, _, _ = makeDataForSmilesOnly("CDK8", dicC2I )
    train_tf8, valid_tf8, test_tf8, pos8, neg8, _, _, _ = makeDataForSmilesOnly("CDK9", dicC2I)
    
    pos_num = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7 + pos8 
    neg_num = neg1 + neg2 + neg3 + neg4 + neg5 + neg6 + neg7 + neg8
print("pos/neg:", pos_num, neg_num)


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_, num_heads_):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads_
        self.d_model = d_model_
        assert d_model_ % self.num_heads == 0
        self.depth = d_model_ // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model_)
        self.wk = tf.keras.layers.Dense(d_model_)
        self.wv = tf.keras.layers.Dense(d_model_)
        self.dense = tf.keras.layers.Dense(d_model_)

    def split_heads(self, x_, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x_ = tf.reshape(x_, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x_, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q) 
        k = self.wk(k) 
        v = self.wv(v)  
        q = self.split_heads(q, batch_size)  )
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention) 
        return output, attention_weights


def point_wise_feed_forward_network(d_model_, dff_):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff_, activation="relu", bias_initializer='glorot_uniform', use_bias=True),  
        tf.keras.layers.Dense(d_model_, bias_initializer='glorot_uniform', use_bias=True) 
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_, num_heads_, dff_, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model_, num_heads_)
        self.ffn = point_wise_feed_forward_network(d_model_, dff_)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask) 
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  
        return out2


class CustomFC(tf.keras.layers.Layer):
    def __init__(self, output_bias, d_model_):
        super(CustomFC, self).__init__()
        self.d_model = d_model_
        self.finalFC1 = tf.keras.layers.Dense(args.n_out, bias_initializer=output_bias)
        self.finalFC2 = tf.keras.layers.Dense(args.n_out, bias_initializer=output_bias)

    def call(self, inputs, seq_len):
        out = self.finalFC1(inputs)

        out = tf.keras.layers.Reshape([seq_len])(out)
        out = self.finalFC2(out)
        return out
    

class CustomHot(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomHot, self).__init__()

    def call(self, inputs):
        return tf.one_hot(inputs, 8)
                                        
class CustomRSum(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomRSum, self).__init__()
    def call(self, inputs, dWhich):
        return tf.math.reduce_sum(inputs * dWhich, axis=1) 


class Encoder(tf.keras.Model):
    def __init__(self, num_layers_, d_model_, num_heads_, dff_, output_bias, rate=0.1, seq_size=None):  # input_vocab and max_vocab are the same
        super(Encoder, self).__init__()
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.seq_size = seq_size
        self.d_model = d_model_
        self.num_layers = num_layers_
        self.enc_layers = [EncoderLayer(d_model_, num_heads_, dff_, rate) for _ in range(num_layers_)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.pads1 = tf.constant([[0, 0], [0, 7 - 1], [0, 0]])
        self.conv1 = tf.keras.layers.Conv2D(d_model_, [7, d_model_], strides=1)

        self.lstm = tf.keras.layers.LSTM(d_model_,return_sequences=True, return_state=True) 
        self.FC1 = CustomFC(output_bias, self.d_model)
        self.FC2 = CustomFC(output_bias, self.d_model)
        self.FC3 = CustomFC(output_bias, self.d_model)
        self.FC4 = CustomFC(output_bias, self.d_model)
        self.FC5 = CustomFC(output_bias, self.d_model)
        self.FC6 = CustomFC(output_bias, self.d_model)
        self.FC7 = CustomFC(output_bias, self.d_model)
        self.FC8 = CustomFC(output_bias, self.d_model)

    def call(self, x_, whichClass, training, mask_att, justmask):
        x_ ,_ ,_= self.lstm(x_)

        for i in range(self.num_layers):
            x_ = self.enc_layers[i](x_, training, mask_att)
            
        # print(x_.shape,"out")
        out = self.dropout(x_, training=training)
        for i in range(self.num_layers):
            x_ = self.enc_layers[i](x_, training, mask_att)
        out = self.dropout(x_, training=training)
        cl1 = self.FC1(out, self.seq_size)
        cl2 = self.FC2(out, self.seq_size)
        cl3 = self.FC3(out, self.seq_size)
        cl4 = self.FC4(out, self.seq_size)
        cl5 = self.FC5(out, self.seq_size)
        cl6 = self.FC6(out, self.seq_size)
        cl7 = self.FC7(out, self.seq_size)
        cl8 = self.FC8(out, self.seq_size)

        x_out = tf.keras.layers.concatenate([cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8])#default axis -1 ==> batch, 12
        decideWhich = CustomHot()(whichClass)
        pred_logit = CustomRSum()(x_out, decideWhich)
        return pred_logit, tf.math.sigmoid(pred_logit)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model_, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model_
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred_logit, sampleW=None):
    cross_ent = tf.nn.weighted_cross_entropy_with_logits(logits=pred_logit, labels=real, pos_weight=sampleW)
    return tf.reduce_sum(cross_ent)


def create_padding_mask_fp2vec(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :], seq  

def create_padding_mask_scfp(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.cast(tf.math.argmin(seq, axis=-1), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :], seq 


learning_rate = args.lr 
optimizer = tf.keras.optimizers.Adam(learning_rate)  

train_loss = tf.keras.metrics.Mean(name='train_loss')
AUCFunc = tf.keras.metrics.AUC()
accFunc = tf.keras.metrics.BinaryAccuracy()
precFunc = tf.keras.metrics.Precision(name='precFunc')
recallFunc = tf.keras.metrics.Recall(name='recallFunc')
initial_bias = np.log([pos_num / neg_num])

weight_for_1 = tf.convert_to_tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=tf.float32)

encoder = Encoder(args.num_layers, args.d_model, args.num_heads, args.dff, output_bias=initial_bias, rate=args.dropout_rate, seq_size=args.seq_size)
TC.set_model(encoder)
checkpoint_dir = "trS"+args.current_num+"/cp.ckpt"
if args.pickle_load == False:
    bit_size = len(dicC2I) 
    emb = tf.Variable(tf.random.uniform([bit_size, args.d_model], -1, 1), dtype=tf.float32)
    pads = tf.constant([[1,0], [0,0]])
    embeddings_ = tf.pad(emb, pads) 
    encoder.save_weights(checkpoint_dir)
else:
    print("embedding loaded already")
    if args.weight_load == True:
        encoder.load_weights(checkpoint_dir)

        
def train_step(inp_, real, whichClass): 
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    with tf.GradientTape() as tape:
        pred_logit, pred = encoder(tf.nn.embedding_lookup(embeddings_, inp_), whichClass, True, inp_padding_mask, justmask)
        loss = loss_function(real, pred_logit, sampleW=weight_for_1)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    train_loss(loss)


def eval_step(inp_, real, whichClass):
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    _, pred = encoder(tf.nn.embedding_lookup(embeddings_, inp_), whichClass, False, inp_padding_mask, justmask)

    precFunc.update_state(y_true=real, y_pred=pred)
    recallFunc.update_state(y_true=real, y_pred=pred)
    AUCFunc.update_state(y_true=real, y_pred=pred)
    accFunc.update_state(y_true=real, y_pred=pred)
 
def test_step(inp_, real, whichClass):
    inp_padding_mask, justmask = create_padding_mask_fp2vec(inp_)
    logit, pred = encoder(tf.nn.embedding_lookup(embeddings_, inp_), whichClass, False, inp_padding_mask, justmask)
    
    precFunc.update_state(y_true=real, y_pred=pred)
    recallFunc.update_state(y_true=real, y_pred=pred)
    AUCFunc.update_state(y_true=real, y_pred=pred)
    accFunc.update_state(y_true=real, y_pred=pred)
    print(len(logit), len(real))
    return pred,real

def indices2chars(listx, dicC2I):
    dicI2C = {}
    for key in dicC2I:
        dicI2C[dicC2I[key]] = key

    retList = []
    for instance in listx:
        tmplist = []
        for index in instance:
            if index == 0:
                break
            tmplist.append(dicI2C[index])
        retList.append(tmplist)
    return retList
f2w = open("performanceS", "w")
bestEpoch = 0
bestAUC = 0
bestACC = 0
for epoch in range(args.epochs):
    start = time.time()
    train_loss.reset_states()
    precFunc.reset_states()
    recallFunc.reset_states()
    AUCFunc.reset_states()
    accFunc.reset_states()
    for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8,  in itertools.zip_longest(train_tf1, train_tf2, train_tf3, train_tf4, train_tf5, train_tf6, train_tf7,
                                                     train_tf8):
        if tf1:
            train_step(tf1[0], tf1[1], 0)
        if tf2:
            train_step(tf2[0], tf2[1], 1)
        if tf3:
            train_step(tf3[0], tf3[1], 2)
        if tf4:
            train_step(tf4[0], tf4[1], 3)
        if tf5:
            train_step(tf5[0], tf5[1], 4)
        if tf6:
            train_step(tf6[0], tf6[1], 5)
        if tf7:
            train_step(tf7[0], tf7[1], 6)
        if tf8:
            train_step(tf8[0], tf8[1], 7)


        
    print('Train Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    f2w.write('Train Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    f2w.write("\n")
    if epoch % 1 == 0:
        for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8, in itertools.zip_longest(valid_tf1, valid_tf2, valid_tf3, valid_tf4, valid_tf5, valid_tf6,
                                             valid_tf7, valid_tf8):
            if tf1:
                eval_step(tf1[0], tf1[1], 0)
            if tf2:
                eval_step(tf2[0], tf2[1], 1)
            if tf3:
                eval_step(tf3[0], tf3[1], 2)
            if tf4:
                eval_step(tf4[0], tf4[1], 3)
            if tf5:
                eval_step(tf5[0], tf5[1], 4)
            if tf6:
                eval_step(tf6[0], tf6[1], 5)
            if tf7:
                eval_step(tf7[0], tf7[1], 6)
            if tf8:
                eval_step(tf8[0], tf8[1], 7)

        if bestAUC < AUCFunc.result():
            bestEpoch = epoch + 1
            bestAUC = AUCFunc.result()
            encoder.save_weights(checkpoint_dir)
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_dir))
        print('Valid prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                              AUCFunc.result(), accFunc.result()))
        logs = {'auc': AUCFunc.result(), 'loss': train_loss.result()}
        TC.on_epoch_end(epoch, logs)
        
        f2w.write('Valid prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                                  AUCFunc.result(), accFunc.result()))
        f2w.write("\n")
        f2w.flush()
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
f2w.close()


######testing phase
precFunc.reset_states()
recallFunc.reset_states()
AUCFunc.reset_states()
accFunc.reset_states()


def listg(a):

    newlist = []
    for i in a:
        for j in i:
            newlist.append(j)
    return newlist
real = []

pre = []
encoder.load_weights(checkpoint_dir)

print("weights loaded from the epoch:", bestEpoch)



if not args._test:
    for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8 in itertools.zip_longest(test_tf1, test_tf2, test_tf3, test_tf4, test_tf5, test_tf6,
                                                                          test_tf7, test_tf8):
    
                                                 
        if tf1:
            
            pr,tr = test_step(tf1[0], tf1[1], 0)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

                        
        if tf2:
            pr,tr = test_step(tf2[0], tf2[1], 1)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

        if tf3:
            pr,tr = test_step(tf3[0], tf3[1], 2)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

        if tf4:
            pr,tr = test_step(tf4[0], tf4[1], 3)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

                        
        if tf5:
            pr,tr = test_step(tf5[0], tf5[1], 4)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

        if tf6:
           pr,tr = test_step(tf6[0], tf6[1], 5)
           pr = (pr.cpu().numpy()).tolist()
           tr = (tr.cpu().numpy()).tolist()

                        
        if tf7:
           pr,tr = test_step(tf7[0], tf7[1], 6)
           pr = (pr.cpu().numpy()).tolist()
           tr = (tr.cpu().numpy()).tolist()

                    
        if tf8:
           pr,tr = test_step(tf8[0], tf8[1], 7)
           pr = (pr.cpu().numpy()).tolist()
           tr = (tr.cpu().numpy()).tolist()

                        

    real = listg(real)
    pre = listg(pre)
    print('Test prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                     AUCFunc.result(), accFunc.result()))
    

TC.on_train_end('_')

if not args.pickle_load:
    pickle.dump((embeddings_, dicC2I), open(args.current_num+"saved_emb.pkl", "wb"))      
      

'''
all_score  = pd.DataFrame()
for i in range (20):
seed = i
def makeDataForSmilesOnly(proteinName, dicC2I):
    print(proteinName)
    listX, listY = [], []
    afile = args.input + '/' + proteinName + '.csv'
    
    file = pd.read_csv(afile)
    
    X = file["Smiles"]
    Y = file["Label"]
    
    train,val_ = train_test_split(file[["Smiles","Label"]],random_state=seed,test_size=0.2,stratify=(file["Label"]))
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y,random_state=seed,test_size=0.5,stratify=(Y))
    
    val_,test_= train_test_split(val_,random_state=212,test_size=0.5,stratify=(val_["Label"]))
    
    # valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y,random_state=212,test_size=0.5,stratify=(valid_y))
    
    if args.aug_num > 1:
        train_temp = pd.concat([train] * (args.aug_num - 1), axis=0)
        train_temp["Smiles"]= train_temp["Smiles"].map(lambda x: randomize_smile(x))
        train = pd.concat([train_temp, train], ignore_index=True)
        train_x = [char2indices(i, dicC2I) for i in train["Smiles"] ]
        train_y = [float(i) for i in train["Label"] ]
        
    else:
        train_x = [char2indices(i, dicC2I) for i in train_x] # length can vary
        train_y = [float(i) for i in train_y ]
        
    if args.aug_num > 1:
            val_temp = pd.concat([val_] * (args.aug_num - 1), axis=0)
            val_temp["Smiles"]= val_temp["Smiles"].map(lambda x: randomize_smile(x))
            val_ = pd.concat([val_temp, val_], ignore_index=True)
            valid_x = [char2indices(i, dicC2I) for i in val_["Smiles"] ]
            valid_y = [float(i) for i in val_["Label"] ]
            
    else:
            valid_x = [char2indices(i, dicC2I) for i in valid_x] # length can vary
            valid_y = [float(i) for i in valid_y ]
    # valid_x = [char2indices(i, dicC2I) for i in valid_x]
    test_x = [char2indices(i, dicC2I) for i in test_["Smiles"].values]
    

    # valid_y = [float(i) for i in valid_y ]
    test_y = [float(i) for i in test_["Label"].values ]
        
    pos_num, neg_num = posNegNums(train_y)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batchsize)
    valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(args.batchsize)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batchsize)
    print(proteinName+"done")
    return train_tf, valid_tf, test_tf, pos_num, neg_num, train_x, valid_x, test_x#testlistX


embeddings_, dicC2I = pickle.load(open(args.current_num+"saved_emb.pkl", "rb"))



if not args._test:
    train_tf1, valid_tf1, test_tf1, pos1, neg1, _, _, _ = makeDataForSmilesOnly("CDK1", dicC2I)
    train_tf2, valid_tf2, test_tf2, pos2, neg2, train2, valid2, test2 = makeDataForSmilesOnly("CDK2", dicC2I)
    train_tf3, valid_tf3, test_tf3, pos3, neg3, _, _, _ = makeDataForSmilesOnly("CDK4", dicC2I )
    train_tf4, valid_tf4, test_tf4, pos4, neg4, _, _, _ = makeDataForSmilesOnly("CDK5", dicC2I)
    train_tf5, valid_tf5, test_tf5, pos5, neg5, _, _, _ = makeDataForSmilesOnly("CDK6", dicC2I )
    train_tf6, valid_tf6, test_tf6, pos6, neg6, _, _, _ = makeDataForSmilesOnly("CDK7", dicC2I)
    train_tf7, valid_tf7, test_tf7, pos7, neg7, _, _, _ = makeDataForSmilesOnly("CDK8", dicC2I )
    train_tf8, valid_tf8, test_tf8, pos8, neg8, _, _, _ = makeDataForSmilesOnly("CDK9", dicC2I)



print("weights loaded from the epoch:", bestEpoch)



if not args._test:
    for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8 in itertools.zip_longest(test_tf1, test_tf2, test_tf3, test_tf4, test_tf5, test_tf6,
                                                                          test_tf7, test_tf8):
        
    # for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8 in itertools.zip_longest(valid_tf1, valid_tf2, valid_tf3, valid_tf4, valid_tf5, valid_tf6,
    #                                           valid_tf7, valid_tf8):
                                                 
        if tf1:
            
            pr,tr = test_step(tf1[0], tf1[1], 0)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

                        
        if tf2:
            pr,tr = test_step(tf2[0], tf2[1], 1)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

        if tf3:
            pr,tr = test_step(tf3[0], tf3[1], 2)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

        if tf4:
            pr,tr = test_step(tf4[0], tf4[1], 3)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

        if tf5:
            pr,tr = test_step(tf5[0], tf5[1], 4)
            pr = (pr.cpu().numpy()).tolist()
            tr = (tr.cpu().numpy()).tolist()

                        
        if tf6:
           pr,tr = test_step(tf6[0], tf6[1], 5)
           pr = (pr.cpu().numpy()).tolist()
           tr = (tr.cpu().numpy()).tolist()
           pred_re[5] = pr + pred_re[5]
           true[5] = tr + true[5]
                        
        if tf7:
           pr,tr = test_step(tf7[0], tf7[1], 6)
           pr = (pr.cpu().numpy()).tolist()
           tr = (tr.cpu().numpy()).tolist()

        if tf8:
           pr,tr = test_step(tf8[0], tf8[1], 7)
           pr = (pr.cpu().numpy()).tolist()
           tr = (tr.cpu().numpy()).tolist()

                        

    real = listg(real)
    pre = listg(pre)
    print('Test prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format(precFunc.result(), recallFunc.result(),
                                                                     AUCFunc.result(), accFunc.result()))
    
    # from sklearn.metrics import accuracy_score
    # eval_acc = accuracy_score(real,
    #                            (np.array(pre) > 0.5))
    Score = score_re(true,pred_re,tasks)
    all_score = pd.concat([all_score,Score],axis = 0)
    print(all_score)
    # Score.to_csv("./res_aug_test_0.001.csv",index = False)
TC.on_train_end('_')

if not args.pickle_load:
    pickle.dump((embeddings_, dicC2I), open(args.current_num+"saved_emb.pkl", "wb"))     
                                 

            '''
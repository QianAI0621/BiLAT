import pandas as pd
import numpy as np
from preprocessing import randomize_smile
import re
from copy import deepcopy



regex_pattern=r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
class Dataset(object):
    def __init__(self, filename,
                 smile_field,
                 label_field,
                 max_len=150,
                 train_augment_times=1,
                 test_augment_times=1,
                 random_state=0):

        df = filename
        df['length'] = df[smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y')))
        self.df = deepcopy(df[df.length <= max_len])
        self.smile_field = smile_field
        self.label_field = label_field
        self.max_len = max_len
        self.train_augment_times = train_augment_times
        self.test_augment_times = test_augment_times
        self.random_state = random_state
        vocab = np.load('./new_vocab.npy', allow_pickle=True)
        self.vocab =  vocab.item()

    def numerical_smiles(self, data):
        x = np.zeros((len(data), (self.max_len + 2)), dtype='int32')
        y = np.array(data[self.label_field]).astype('float32')
        for i,smiles in enumerate(data[self.smile_field].tolist()):
            smiles = self._char_to_idx(seq = smiles)
            smiles = self._pad_start_end_token(smiles)
            x[i,:len(smiles)] = np.array(smiles)
        return x, y

    def _pad_start_end_token(self,seq):
        seq.insert(0, self.vocab['<s>'])
        seq.append(self.vocab['<end>'])

        return seq


    def _char_to_idx(self,seq):
        char_list = re.findall(regex_pattern, seq)
        return [self.vocab[char_list[j]] for j in range(len(char_list))]
    def get_data(self):
        data = self.df
        length_count = data.length.value_counts()
        train_idx = []
        for k, v in length_count.items():
            if v >= 3:
                idx = data[data.length == k].sample(frac=0.9, random_state=self.random_state).index
            else:
                idx = data[data.length == k].sample(n=1, random_state=self.random_state).index
            train_idx.extend(idx)

        X_train = deepcopy(data[data.index.isin(train_idx)])
        X_test = deepcopy(data[~data.index.isin(train_idx)])

        if self.train_augment_times>1:
            train_temp = pd.concat([X_train] * (self.train_augment_times - 1), axis=0)
            train_temp[self.smile_field] = train_temp[self.smile_field].map(lambda x: randomize_smile(x))
            train_set = pd.concat([train_temp, X_train], ignore_index=True)
        else:
            train_set = X_train
        train_set.dropna(inplace=True)
        train_set = deepcopy(train_set)
        train_set['length'] = train_set[self.smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y')))
        train_set = train_set[train_set.length <= self.max_len]

        if self.test_augment_times>1:
            test_temp = pd.concat([X_test] * (self.test_augment_times - 1), axis=0)
            test_temp[self.smile_field] = test_temp[self.smile_field].map(lambda x: randomize_smile(x))
            test_set = pd.concat([test_temp, X_test], ignore_index=True)
            # test_set['length'] = test_set[self.smile_field].map(lambda x: len(x.replace('Cl', 'X').replace('Br', 'Y')))
            # test_set = test_set[test_set.length <= self.max_len]
        else:
            test_set = X_test
        test_set = deepcopy(test_set)


        x_train,y_train = self.numerical_smiles(train_set)
        x_test, y_test = self.numerical_smiles(test_set)
        print(len(X_train)/len(X_train[self.smile_field].unique()))
        print(x_test.shape)
        x_train = x_train.tolist()
        y_train = y_train.tolist()
        x_test = x_test.tolist()
        y_test = y_test.tolist()
        
        return x_train , y_train,x_test,y_test




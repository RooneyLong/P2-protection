# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def data_reader(data_name = "adult"):
    if(data_name == "adult"):
        #load data
        file_path = "./dataset/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)
        data2 = pd.read_csv(file_path + 'adult.test', header=None)
        data2 = data2.replace(' <=50K.', ' <=50K')    
        data2 = data2.replace(' >50K.', ' >50K')
        
        data = pd.concat([data1,data2])

        #data transform: str->int
        data = np.array(data, dtype=str)
        labels = data[:,14]
        le= LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:,:-1]
        
        categorical_features = [1,3,5,6,7,8,9,13]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)
        
        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
            data[:,feature] = sacled_data.reshape(-1)
        
        #OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
    

    elif(data_name == "bank"):
        #load data
        file_path = "./data/bank/"
        data = pd.read_csv(file_path + 'bank-full.csv',sep=';')
        #data transform
        data = np.array(data, dtype=str)
        labels = data[:,-1]
        le= LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:,:-1]
        
        categorical_features = [1,2,3,4,6,7,8,10,15]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)
        
        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:,feature].reshape(-1,1))
            data[:,feature] = sacled_data.reshape(-1)
        #OneHotLabel
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
    elif(data_name == "mnist"):
        file_path = "./data/mnist/"
        data = pd.read_csv(file_path + 'mnist_train.csv', header=None)
        data = np.array(data)
        labels = data[:,0]
        data = data[:,1:]
        
        categorical_features = []
        data = data/data.max()
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
    else:
        
        str_list = data_name.split('_')
        file_path = "./data/purchase/"
        data = pd.read_csv(file_path+'dataset_purchase')
        data = np.array(data)
        data = data[:,1:]
        
        label_file = './data/purchase/label'+ str_list[1] + '.npy'
        
        labels = np.load(label_file)
        
        categorical_features = []
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
            remainder='passthrough' )
        oh_data = oh_encoder.fit_transform(data)
        
        X_train, _, y_train, _ = train_test_split(oh_data, labels,test_size = 0.75)
        oh_data = X_train
        labels = y_train
        
    #randomly select 10000 records as training data
    train_idx = np.random.choice(len(labels), 30000, replace = False)
    idx = range(len(labels))
    idx = np.array(idx)
    test_idx = list(set(idx).difference(set(train_idx)))
    test_idx = np.array(test_idx)
    
    assert test_idx.sum() + train_idx.sum() == idx.sum()
    
    X_train = data[train_idx,:]
    Y_train = labels[train_idx]
    
    X_test = data[test_idx,:]
    Y_test = labels[test_idx]
    
    orig_dataset = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}
    
    X_train = oh_data[train_idx,:]
    
    X_test = oh_data[test_idx,:]
    
    oh_dataset = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}

    return orig_dataset, oh_dataset, oh_encoder


def fn_Feature_Range_Counter(dataset):
    '''
    Function: counting the value range for each feature of the dataset
    '''
    data = np.vstack([dataset['X_train'], dataset['X_test']])
    feature_num = data.shape[1]
    
    feature_range_dict = {}
    for ii in range(feature_num):
        cnt = Counter(data[:,ii])
        values = list(cnt.keys())
        values = np.array(values)
        feature_range_dict[ii] = values
        
        
    # labels = np.hstack([dataset['Y_train'], dataset['Y_test']]) 
    # n_class = len(Counter(labels))
    return feature_range_dict
    


if(__name__ == '__main__'):
    file_path = "./data/purchase/"
    data = pd.read_csv(file_path+'dataset_purchase')
    data = np.array(data)
    label_100 = data[:,0]
    data = data[:,1:]
    
    label_2 = KMeans(n_clusters=2, random_state=9).fit_predict(data)
    np.save("./data/purchase/label2", label_2)
    print("label_2 finished!")
    
    label_10 = KMeans(n_clusters=10, random_state=9).fit_predict(data)
    np.save("./data/purchase/label10", label_10)
    print("label_10 finished!")
    
    label_20 = KMeans(n_clusters=20, random_state=9).fit_predict(data)
    np.save("./data/purchase/label20", label_20)
    print("label_20 finished!")
    
    label_50 = KMeans(n_clusters=50, random_state=9).fit_predict(data)
    np.save("./data/purchase/label50", label_50)
    print("label_50 finished!")

    np.save("./data/purchase/label100", label_100)
    print("label_100 finished!")
        #OneHotLabel
    
    # ct = ColumnTransformer(
        # [('oh_enc', OneHotEncoder(sparse=False), categorical_features),], 
        # remainder='passthrough' )
    # ct.fit(data)










# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from data_preprocessing import data_reader


orig_dataset, oh_dataset, oh_encoder = data_reader("adult")


x_train_dataset_org = orig_dataset['X_train']
y_train_dataset_org= orig_dataset['Y_train']

x_test_dataset_org = orig_dataset['X_test'][0:18000]
y_test_dataset_org = orig_dataset['Y_test'][0:18000]



x_dataset = np.vstack((x_train_dataset_org,x_test_dataset_org))
y_dataset = np.hstack((y_train_dataset_org,y_test_dataset_org))

# ###################

num_x_feature = np.size(x_dataset,1)
num_x_record  = np.size(x_dataset,0)


x_name = np.ones((num_x_feature,), dtype=np.int)


x_data_all = pd.DataFrame(columns=(x_name), data=x_dataset)
x_data_all.to_csv('./dataset/adult/dataset_adult.csv',index = False,header=True) 


y_name = ['0']

y_data_all = pd.DataFrame(columns=(y_name), data=y_dataset)
y_data_all.to_csv('./dataset/adult/dataset_adult_label.csv',index = False,header=True) 

x_dataset = pd.read_csv('./dataset/adult/dataset_adult.csv')


lables = pd.read_csv('./dataset/adult/dataset_adult_label.csv')


data_size = 24000
x_train_dataset = x_dataset.iloc[:data_size]
y_train_dataset = lables.iloc[:data_size]

x_test_dataset = x_dataset.iloc[data_size:2*data_size]
y_test_dataset = lables.iloc[data_size:2*data_size]


x_train_dataset.to_csv('./dataset/adult/x_train_adult_all.csv',index = False,header=True)   
y_train_dataset.to_csv('./dataset/adult/y_train_adult_all.csv',index = False,header=True)   

x_test_dataset.to_csv('./dataset/adult/x_test_adult_all.csv',index = False,header=True)  
y_test_dataset.to_csv('./dataset/adult/y_test_adult_all.csv',index = False,header=True) 


num_partition = 10
num_data_size = int(data_size/num_partition)


for i in range(num_partition):
    num_index = i
    print(num_index)
    partition_start = num_index
    partition_end = num_index + 1
    
    x_train_dataset_save = x_train_dataset.iloc[num_data_size*partition_start:num_data_size*partition_end]
    y_train_dataset_save = y_train_dataset.iloc[num_data_size*partition_start:num_data_size*partition_end]
        
    x_test_dataset_save = x_test_dataset.iloc[num_data_size*partition_start:num_data_size*partition_end]
    y_test_dataset_save = y_test_dataset.iloc[num_data_size*partition_start:num_data_size*partition_end]
            
 
    x_train_dataset_save.to_csv('./dataset/adult/x_train_adult_{}.csv'.format(num_index),index = False,header=True)   
    y_train_dataset_save.to_csv('./dataset/adult/y_train_adult_{}.csv'.format(num_index),index = False,header=True)   
    
    x_test_dataset_save.to_csv('./dataset/adult/x_test_adult_{}.csv'.format(num_index),index = False,header=True)   
    y_test_dataset_save.to_csv('./dataset/adult/y_test_adult_{}.csv'.format(num_index),index = False,header=True) 

print('this is adult data partition for {} clients'.format(num_partition))
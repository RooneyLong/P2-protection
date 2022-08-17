# -*- coding: utf-8 -*-


import member_inf
import torch
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import attack_data


## Model to train attack model on. Should be same as the one trained.
class Net(torch.nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature,500)
        self.fc2 = nn.Linear(500,300)
        self.fc3 = nn.Linear(300,2)
        
        self.grad_list = []
        self.layer_list = []        
        self.layer_output_list = []
        

    def forward(self, x):      
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)         
        return x

    def eval_layer_output(self, x):
        x = F.relu(self.fc1(x))
        self.layer_output_list.append(x)
        x = F.relu(self.fc2(x))
        self.layer_output_list.append(x)
        x = self.fc3(x)
        self.layer_output_list.append(x)
        return self.layer_output_list, x
    

    def get_grad(self, list=[]):        
        return self.grad_list


x_train_all = pd.read_csv('../dataset/adult/x_train_adult_all.csv')
y_train_all = pd.read_csv('../dataset/adult/y_train_adult_all.csv')

x_test_all = pd.read_csv('../dataset/adult/x_test_adult_all.csv')
y_test_all = pd.read_csv('../dataset/adult/y_test_adult_all.csv')


sample_size = 2000
train_data = pd.concat([x_train_all,y_train_all],axis=1)
test_data = pd.concat([x_test_all,y_test_all],axis=1)


train_data_sample = train_data.sample(n=sample_size,random_state=None)
test_data_sample = test_data.sample(n=sample_size,random_state=None)


x_train_attack = train_data_sample.iloc[:, 0:14]
y_train_attack = train_data_sample.iloc[:, 14:15]
x_test_attack = test_data_sample.iloc[:, 0:14]
y_test_attack = test_data_sample.iloc[:, 14:15]

x_train_attack.to_csv('../dataset/adult/x_train_adult_attack.csv',index = False,header=True)
y_train_attack.to_csv('../dataset/adult/y_train_adult_attack.csv',index = False,header=True)
x_test_attack.to_csv('../dataset/adult/x_test_adult_attack.csv',index = False,header=True)
y_test_attack.to_csv('../dataset/adult/y_test_adult_attack.csv',index = False,header=True)


cprefixA = 'global_model_g_count_9.pkl'
cmodelA = torch.load('../model/adult/{}'.format(cprefixA))


train_feature_path = '../dataset/adult/x_train_adult_{}.csv'.format('attack')
train_lable_path = '../dataset/adult/y_train_adult_{}.csv'.format('attack')
test_feature_path = '../dataset/adult/x_test_adult_{}.csv'.format('attack')
test_lable_path = '../dataset/adult/y_test_adult_{}.csv'.format('attack')

save_list = []



input_features = 14


datahandleA = attack_data.attack_data(train_feature_path,
                                      train_lable_path,
                                      test_feature_path,
                                      test_lable_path,
                                      batch_size=64,
                                      attack_percentage=50,
                                      input_shape=(input_features,))


attackobj = member_inf.initialize(
        target_train_model = cmodelA,
        target_attack_model = cmodelA,
        train_datahandler = datahandleA,
        attack_datahandler = datahandleA,

        gradients_to_exploit=[1,2,3],
        learning_rate=0.001,
        optimizer='SGD',
        epochs=100,
        model_name='adult_dataset_14x2y_classification'
        )


attackobj.train_attack()


best_attack_accuracy = max(attackobj.last20_epochs)

str_save = str('model {} before protecting: {} '.format(cprefixA, best_attack_accuracy))
save_list.append(str_save)


best_attack_epoch = attackobj.last20_epochs.index(best_attack_accuracy) + (100 - 1 - 20)




cprefixB = 'protected_global_model_g_count_9_layers12345.pkl'
cmodelB = torch.load('../adult/adult_defend/global/{}'.format(cprefixB))


protected_attackobj = member_inf.initialize(
        target_train_model = cmodelB,
        target_attack_model = cmodelB,

        train_datahandler = datahandleA,
        attack_datahandler = datahandleA,
        gradients_to_exploit=[1,2,3],
        learning_rate=0.001,
        optimizer='SGD',
        epochs=best_attack_epoch,
        model_name='adult_dataset_14x2y_classification'
        )

protected_attackobj.train_attack()
protected_attack_accuracy = protected_attackobj.last20_epochs[-1]
str_save = str('model{} after protecting: {} '.format(cprefixB,protected_attack_accuracy))
save_list.append(str_save)

info_save = '\n'.join(save_list)
with open('test_info_save.txt', 'w') as f:     
    f.write(info_save) 





# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# difine the network
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

net_g = Net(n_feature=14)

num_client = 10

g_iteration_count = 10
num_model_layer = 3

str_name = 'layers{}'.format('123')

cur_all_client = []

for client_i in range(num_client):    

    cur_net = torch.load('../model/adult_defend/protected_client_{}_g_count_{}_{}.pkl'.format(client_i, g_iteration_count-1,str_name))
    cur_all_client.append(cur_net)




item_name_arr = []
for i in range(num_model_layer):
    item_name_arr.append('fc{}.weight'.format(i+1))
for i in range(num_model_layer):
    item_name_arr.append('fc{}.bias'.format(i+1))
 

model_dict  = net_g.state_dict()

for i in range(len(item_name_arr)):
    cur_name = item_name_arr[i]
   
    for j in range(len(cur_all_client)):
        net_cur_ele = cur_all_client[j]        
        if j == 0 :
            sum_item = net_cur_ele.state_dict()[cur_name]
        else:
            sum_item = sum_item + net_cur_ele.state_dict()[cur_name]

    model_dict[cur_name] = sum_item/(num_client*1.0)


net_g.load_state_dict(model_dict)
print('##### finish and save protected global_model_once_in_g_count_{}'.format(g_iteration_count-1)) 

torch.save(net_g, '../model/adult_defend/global/protected_global_model_g_count_{}_{}.pkl'.format(g_iteration_count-1,str_name))  # save entire net
torch.save(net_g.state_dict(),'../model/adult_defend/global/protected_global_model_g_count_{}_params_{}.pkl'.format(g_iteration_count-1,str_name))         
    
x_test_all = pd.read_csv('../dataset/adult/x_test_adult_{}.csv'.format('all'))
y_test_all = pd.read_csv('../dataset/adult/y_test_adult_{}.csv'.format('all'))

x_test_g = np.array(x_test_all)
x_test = torch.FloatTensor(x_test_g)
y_test_g = np.array(y_test_all)
y_test = torch.LongTensor(y_test_g).squeeze()    

out_test = net_g(x_test)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("################# test_result_protected_net_g_model = ",accuracy)      
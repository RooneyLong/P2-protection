# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy

# the number of classes
class_num = 2


# model structure
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

    # get membership information
    def get_grad(self, list=[]):        
        return self.grad_list


####################################################################
# client numbers
num_client = 10
# The number of aggregation
g_iteration_count = 10


cur_num_client = 9
# disturbation intensity
eps_setting = 0.1
# protected layers
str_name = 'layers{}'.format('123')

net_la =torch.load('../model/adult/client_{}_g_count_{}.pkl'.format(cur_num_client, g_iteration_count-1))


x_train = pd.read_csv('../dataset/adult/x_train_adult_{}.csv'.format(cur_num_client))
y_train = pd.read_csv('../dataset/adult/y_train_adult_{}.csv'.format(cur_num_client))

x_test = pd.read_csv('../dataset/adult/x_test_adult_{}.csv'.format(cur_num_client))
y_test = pd.read_csv('../dataset/adult/y_test_adult_{}.csv'.format(cur_num_client))
 

x_train = np.array(x_train)
x_train = torch.FloatTensor(x_train)

y_train = np.array(y_train)
y_train = torch.LongTensor(y_train)


x_a = x_train

y_a = y_train.squeeze()


net_w_b = copy.deepcopy(net_la)


net_w_b.fc1.weight.requires_grad = True
net_w_b.fc2.weight.requires_grad = True
net_w_b.fc3.weight.requires_grad = True

net_w_b.fc1.bias.requires_grad = True
net_w_b.fc2.bias.requires_grad = True
net_w_b.fc3.bias.requires_grad = True

    

net_w_b_opt = optim.Adam(filter(lambda p: p.requires_grad, net_w_b.parameters()), 
                         lr=0.0003)

x_test = np.array(x_test)
x_test = torch.FloatTensor(x_test)

y_test = np.array(y_test)
y_test = torch.LongTensor(y_test).squeeze() 


out_test = net_w_b(x_test)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print('test_result_adult_model before protection =', accuracy)




num_protect_layer = 2

input_dataset = x_train
input_dataset_y = y_train
len_example = input_dataset.size()[0]  # len_example = 2400


num_classification = class_num

def createOHE(num_output_classes):
        """
        creates one hot encoding matrix of all the vectors
        in a given range of 0 to number of output classes.
        """
        return F.one_hot(torch.arange(0, num_output_classes),
                         num_output_classes)
    
    

def createLabel(num_output_classes):
    res = []
    for i in range(num_output_classes):
        res.append([i])
    return res

y_ext_list_prob = createOHE(num_classification)
y_ext_list_prob = y_ext_list_prob.float()    # tensor([[1., 0.],[0., 1.]])

y_ext_list_label = torch.LongTensor(createLabel(num_classification))  # tensor([[0],[1]])


def get_grad_direction(example_input, net, protect_number_layer, y_ext_label):

    model_input = example_input     

    model_output = net(model_input)

    
    model_output = torch.unsqueeze(model_output,0)   # (1,2)
    print(model_output.size())
    

    loss_func = torch.nn.CrossEntropyLoss()
    loss_a = loss_func(model_output, y_ext_label)
    print(model_output.shape, y_ext_label.shape)
    
    cur_arr = []
    

    net_w_b_opt.zero_grad()
    

    with torch.no_grad():
        loss_a.backward()
        for name, parms in net.named_parameters():
 
            if parms.grad != None:
                cur_arr.append(parms.grad)  

    return copy.deepcopy(cur_arr)   


def get_point_via_bisection(org_y, graddif_y, eps, seg_point_start, seg_point_end):

    seg_point = (seg_point_start + seg_point_end)/2

    cur_y = (1-seg_point)*org_y + seg_point*graddif_y
    

    org_label = torch.nonzero(torch.eq(org_y, max(org_y))).squeeze(1)

    if len(org_label) > 1:
        org_label = org_label[0]
    
    cur_label = torch.nonzero(torch.eq(cur_y, max(cur_y))).squeeze(1)

    if len(cur_label) > 1:
        cur_label = cur_label[0]
    

    dif_stop = 1e-3

    # 分类标签的准确度约束，eps
    if torch.abs(torch.max(org_y) - torch.max(cur_y)) <= eps:
        if org_label != cur_label :
            next_seg_start = seg_point_start
            next_seg_end = seg_point        
            seg_point = get_point_via_bisection(org_y, graddif_y, eps, next_seg_start,next_seg_end)
        else:
            if seg_point_end - seg_point_start < dif_stop:
                return seg_point            
            else:
              
                next_seg_start = seg_point
                next_seg_end = seg_point_end        
                seg_point = get_point_via_bisection(org_y, graddif_y, eps, next_seg_start,next_seg_end)
        
    else:
        next_seg_start = seg_point_start
        next_seg_end = seg_point        
        seg_point = get_point_via_bisection(org_y, graddif_y, eps, next_seg_start,next_seg_end)
    
    return seg_point


def get_modified_y_list(org_y_list,max_graddif_y_list,eps):
    len_data = len(org_y_list)
    modified_y_list = []
    
    for i in range(len_data):
        seg_point = get_point_via_bisection(org_y_list[i],max_graddif_y_list[i],eps,0,1)
        new_cur_y = (1-seg_point)*org_y_list[i] + seg_point*max_graddif_y_list[i]
        modified_y_list.append(new_cur_y)
        
    return modified_y_list


y_ext_list = []
for i in range(len_example):
    example_y_org = input_dataset_y[i]
    grad_dir_org = get_grad_direction(input_dataset[i],net_w_b, num_protect_layer,example_y_org)[0]
    
    y_star_max = torch.tensor(-1)
    max_norm_square = torch.tensor(-1)
    
    for j in range(len(y_ext_list_label)):
        grad_dir_ext = get_grad_direction(input_dataset[i],net_w_b, num_protect_layer,y_ext_list_label[j])[0]
        cur_dif = torch.sub(torch.div(grad_dir_ext,torch.norm(grad_dir_ext,p=2)), torch.div(grad_dir_org,torch.norm(grad_dir_org,p=2)))
        cur_norm_square = torch.pow(torch.norm(cur_dif,p=2),2)
        if cur_norm_square > max_norm_square:
            max_norm_square = cur_norm_square
            y_star_max = y_ext_list_label[j]
    
    y_ext_list.append(torch.squeeze(y_ext_list_prob[y_star_max]))
    
net_w_b.layer_output_list = []
layer_output,_ =  net_la.eval_layer_output(x_train)
cur_layer_output = layer_output[num_protect_layer]


modified_y_list = get_modified_y_list(org_y_list_prob, y_ext_list, eps_setting)
print('done modified_y_list') 

class MyLoss(nn.Module):     
    def __init__(self):
       super(MyLoss, self).__init__()
       
    def forward(self, org_y_prob_list, modified_y_list):
         sum_loss = torch.tensor(0,dtype=float,requires_grad=True)
         for i in range(len(org_y_prob_list)):
            cur_loss = torch.norm(torch.sub(org_y_prob_list[i],modified_y_list[i]),p=2)
            sum_loss = torch.add(sum_loss,cur_loss)
         return sum_loss

lossfunc = MyLoss()

iteration_count = 200

for i in range(len(modified_y_list)):
    modified_y_list[i] = modified_y_list[i].detach().numpy()

modified_y_list = torch.tensor(modified_y_list)



torch_dataset = torch.utils.data.TensorDataset(input_dataset, modified_y_list)
train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=128, 
                                          shuffle=False)




for epoch in range(iteration_count):  

     if epoch == 1:
         torch_dataset = torch.utils.data.TensorDataset(input_dataset, modified_y_list)
         train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=128, 
                                          shuffle=False)
    
     for step, data in enumerate(train_loader, start=0):
        input_dataset, modified_y_list = data # 解构出特征和标签
        
        output =  net_w_b(input_dataset)
        y_prob_list = F.softmax(output,dim=1)    
    
     
        loss_prob = lossfunc(y_prob_list,modified_y_list)
        print(loss_prob)
        
        net_w_b_opt.zero_grad()   # clear gradients for next train
        loss_prob.backward(retain_graph=True)         # backpropagation, compute gradients
        net_w_b_opt.step()  
        

net_w_b.fc1.weight.requires_grad = True
net_w_b.fc2.weight.requires_grad = True
net_w_b.fc3.weight.requires_grad = True


net_w_b.fc1.bias.requires_grad = True
net_w_b.fc2.bias.requires_grad = True
net_w_b.fc3.bias.requires_grad = True


torch.save(net_w_b, '../model/adult_defend/protected_client_{}_g_count_{}_{}.pkl'.format(cur_num_client, g_iteration_count-1,str_name)) # save entire net


out_test = net_w_b(x_test)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test.data.numpy()
accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print('test_result_adult_model after protection =', accuracy)




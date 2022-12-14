# -*- coding: utf-8 -*-

import datetime

from logger import get_logger
from attack_utils import attack_utils, sanity_check
from create_fcn import fcn_module
from create_cnn import cnn_for_fcn_gradients
from create_encoder import encoder
import torch
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score
import numpy as np
from attack_model_init import attack_model_init
from losses import CrossEntropyLoss,CrossEntropyLoss_exampleloss,mse
import copy
from optimizers import optimizer_op
import os
import json
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import random

class initialize(object):
    def __init__(self,
                 target_train_model,
                 target_attack_model,
                 train_datahandler,
                 attack_datahandler,
                 optimizer="Adam",            
                 layers_to_exploit=None,
                 gradients_to_exploit=None,
                 exploit_loss=True,
                 exploit_label=True,
                 learning_rate=0.001,
                 epochs=100,
                 model_name='sample',
                 if_withDP = [False, 1, 1],
                 ):
       
        time_stamp = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        self.attack_utils = attack_utils()
        self.logger = get_logger(self.attack_utils.root_dir, "attack",
                               "meminf", "info", time_stamp)  
        self.target_train_model = target_train_model
        self.target_attack_model = target_attack_model
        self.train_datahandler = train_datahandler
        self.attack_datahandler = attack_datahandler        
        self.optimizer = optimizer_op(optimizer)
        self.layers_to_exploit = layers_to_exploit
        self.gradients_to_exploit = gradients_to_exploit
        self.exploit_loss = exploit_loss
        self.exploit_label = exploit_label
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_name = model_name
        self.output_size = int(len(list(target_train_model.parameters())[-1]))
        self.ohencoding = self.attack_utils.createOHE(self.output_size)
        self.last20_epochs = []
        self.if_withDP = if_withDP
        
        # Create input containers for attack & encoder model.
        self.create_input_containers()
        # torch???????????????????????????????????????????????????weight???bias
        self.layers = self.fix_layers()
        
        # basic sanity checks
        sanity_check(self.layers, layers_to_exploit)
        sanity_check(self.layers, gradients_to_exploit)
        
        # Create individual attack components
        self.create_attack_components(self.layers)
        
        # Initialize the attack model
        self.initialize_attack_model()

        # Log info
        self.log_info()
    
    def log_info(self):
        """
        Logs vital information pertaining to training the attack model.
        Log files will be stored in `/ml_privacy_meter/logs/attack_logs/`.
        """
        self.logger.info("`exploit_loss` set to: {}".format(self.exploit_loss))
        self.logger.info(
            "`exploit_label` set to: {}".format(self.exploit_label))
        self.logger.info("`layers_to_exploit` set to: {}".format(
            self.layers_to_exploit))
        self.logger.info("`gradients_to_exploit` set to: {}".format(
            self.gradients_to_exploit))
        self.logger.info("Number of Epochs: {}".format(self.epochs))
        self.logger.info("Learning Rate: {}".format(self.learning_rate))
        self.logger.info("Optimizer: {}".format(self.optimizer))       
    
    def create_input_containers(self):
        """
        Creates arrays for inputs to the attack and 
        encoder model. 
        (NOTE: Although the encoder is a part of the attack model, 
        two sets of containers are required for connecting 
        the TensorFlow graph).
        ???torch????????????????????????????????????????????????????????????????????????????????????????????????????????????xx_real???
        """        
        self.attackinputs = []
        self.encoderinputs = []
        self.encoderinputs_size = 0
        self.attackinputs_size = []

        
    def fix_layers(self):
        fix_result = []
        wb_params = list(self.target_train_model.parameters())
        for i in range(len(wb_params)):
            if i%2 == 0:
                fix_result.append((wb_params[i],wb_params[i+1]))
        return fix_result
    
    def create_attack_components(self, layers):
        """
        Creates FCN and CNN modules constituting the attack model. 
        
        """
        model = self.target_train_model
        
        # for layer outputs
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.create_layer_components(layers)
            
        # for one hot encoded labels
        if self.exploit_label:
            self.create_label_component(self.output_size)
            
        # for loss
        if self.exploit_loss:
            self.create_loss_component()
            
        # for gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.create_gradient_components(model, layers)


            
    def create_layer_components(self, layers):
        """
        Creates CNN or FCN components for layers to exploit
        """
        for l in self.layers_to_exploit:
            # For each layer to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[l-1]  
            input_shape = len(layer[0])      
            module = fcn_module(input_shape, 100)
            self.attackinputs.append(module)
            self.encoderinputs_size += 64
            self.encoderinputs.append(module)  
            self.attackinputs_size.append(input_shape)
   
    def create_label_component(self, output_size):
        """
        Creates component if OHE label is to be exploited
        """
        module = fcn_module(output_size)
        self.attackinputs.append(module)
        self.encoderinputs_size += 64
        self.encoderinputs.append(module)
        self.attackinputs_size.append(output_size)         
            
    def create_loss_component(self):
        """
        Creates component if loss value is to be exploited
        """
        module = fcn_module(1, 100)
        self.attackinputs.append(module)
        self.encoderinputs_size += 64
        self.encoderinputs.append(module) 
        self.attackinputs_size.append(1)    
    
    def create_gradient_components(self, model, layers):
        """
        Creates CNN/FCN component for gradient values of layers of gradients to exploit
        """
        for layerindex in self.gradients_to_exploit:
            # For each gradient to exploit, module created and added to self.attackinputs and self.encoderinputs
            layer = layers[layerindex-1]
            shape = self.attack_utils.get_gradshape(layer)   
            module = cnn_for_fcn_gradients(shape)
            self.attackinputs.append(module)      
            self.encoderinputs_size += 256
            self.encoderinputs.append(module)        
            self.attackinputs_size.append(shape[1])  
            
    def initialize_attack_model(self):
        """
        Initializes a `tf.keras.Model` object for attack model.
        The output of the attack is the output of the encoder module.
        """
        self.attackmodel = attack_model_init(self.attackinputs,self.encoderinputs_size,self.attackinputs_size)
   

    def get_layer_outputs(self, model, features):
        """
        Get the intermediate computations (activations) of 
        the hidden layers of the given target model.
        """
        layer_outputs,_ = model.eval_layer_output(features)

        for l in self.layers_to_exploit:
            self.inputArray.append(layer_outputs[l-1])
 
    def get_labels(self, labels):
        """
        Retrieves the one-hot encoding of the given labels.
        """
        ohe_labels = self.attack_utils.one_hot_encoding(
            labels, self.ohencoding)
        return ohe_labels

    def get_loss(self, model, features, labels):
        """
        Computes the loss for given model on given features and labels
        """
        logits = model(features)
        loss = CrossEntropyLoss_exampleloss(logits, labels)   
        return loss

    def compute_gradients(self, model, features, labels):
        """
        Computes gradients given the features and labels using the loss
        """
        
        gradient_arr = []
        torch_dataset = torch.utils.data.TensorDataset(features, labels)
        train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1,
                                           shuffle=False)
        
        for (step,(feature,label)) in enumerate(train_loader):
            cur_arr = []
            logits = model(feature)
            loss = CrossEntropyLoss(logits, label)
            
            with torch.no_grad():
                loss.backward()                
                for name, parms in model.named_parameters():
                    # Add gradient wrt crossentropy loss to gradient_arr
                    # ?????????weight???bias
                    cur_arr.append(parms.grad)
            # ???????????????????????????cur_arr?????????list?????????append?????????????????????        
            gradient_arr.append(copy.deepcopy(cur_arr))

        return gradient_arr  
     
    def get_gradients(self, model, features, labels):
        """
        Retrieves the gradients for each example.
        """    
        gradient_arr = self.compute_gradients(model, features, labels)  
        batch_gradients = []     
        # gradient_arr ??????batch?????????data example
        for grads in gradient_arr:
            # gradient_arr is a list of size of number of layers having trainable parameters
            gradients_per_example = []
            for g in self.gradients_to_exploit:
                g = (g-1)*2                  

                toappend = torch.unsqueeze(grads[g],0)              

                if(self.if_withDP[0] == True):
                    temp_toappend = torch.squeeze(toappend,0)
                    epsilon = self.if_withDP[2]
                    mu = 0
                    sigma = 1/epsilon                    
                    [row, col] = temp_toappend.size() 
                    added_noise = []
                    for i in range(row):
                        cur_row_noise = []
                        for j in range(col):
                            cur_row_noise.append(random.gauss(mu,sigma))
                        added_noise.append(cur_row_noise)
                    added_noise = torch.tensor(added_noise)
                    temp_toappend +=added_noise
                    toappend = torch.unsqueeze(temp_toappend,0)

                gradients_per_example.append(toappend.numpy())  
            batch_gradients.append(gradients_per_example)

        exploit_layer_num = len(batch_gradients[0])
        for i in range(exploit_layer_num):
            array = []
            for example in batch_gradients:
                array.append(example[i])

            self.inputArray.append(torch.tensor(np.stack(array)))
       
    def get_gradient_norms(self, model, features, labels):
        """
        Retrieves the gradients for each example
        """
        gradient_arr = self.compute_gradients(model, features, labels)
        batch_gradients = []
        # gradient_arr ??????batch?????????data example
        for grads in gradient_arr:
            grad_per_example = []
            for g in range(int(len(grads)/2)):
                g = g*2
                grad_per_example.append(np.linalg.norm(grads[g]))
            batch_gradients.append(grad_per_example)  
        result = np.sum(batch_gradients, axis=0) / len(gradient_arr)
        
        return result   
        
    def forward_pass(self, model, features, labels):
        """
        Computes and collects necessary inputs for attack model
        """
        # container to extract and collect inputs from target model
        self.inputArray = []

        # Getting the intermediate layer computations
        if self.layers_to_exploit and len(self.layers_to_exploit):
            self.get_layer_outputs(model, features)
            
        # Getting the one-hot-encoded labels
        if self.exploit_label:
            ohelabels = self.get_labels(labels)
            self.inputArray.append(ohelabels)
         # Getting the loss value
        if self.exploit_loss:
            loss = self.get_loss(model, features, labels)
            self.inputArray.append(loss)
         # Getting the gradients
        if self.gradients_to_exploit and len(self.gradients_to_exploit):
            self.get_gradients(model, features, labels)    
                 
        attack_outputs = self.attackmodel(self.inputArray)
        return attack_outputs    
        
    def train_attack(self):
        """
        Trains the attack model
        """
        print('start train_attack module')
        m_features, m_labels, nm_features, nm_labels = self.train_datahandler.load_train()

        model = self.target_train_model
        pred = model(m_features)
        if(self.if_withDP[0] == True):
            epsilon = self.if_withDP[1]
            mu = 0
            sigma = 1/epsilon

            [row, col] = pred.size() 
            added_noise = []
            for i in range(row):
                cur_row_noise = []
                for j in range(col):
                    cur_row_noise.append(random.gauss(mu,sigma))
                added_noise.append(cur_row_noise)
            added_noise = torch.tensor(added_noise)
            # print(added_noise)

            pred +=added_noise 
            # print(pred)
    
        acc_train = accuracy_score(m_labels, np.argmax(pred.detach().numpy(), axis=1))
        print('Target model train accuracy = ', acc_train)
        

        pred = model(nm_features)
        
        if(self.if_withDP[0] == True):
            epsilon = self.if_withDP[1]
            mu = 0
            sigma = 1/epsilon
            
            [row, col] = pred.size() 
            added_noise = []
            for i in range(row):
                cur_row_noise = []
                for j in range(col):
                    cur_row_noise.append(random.gauss(mu,sigma))
                added_noise.append(cur_row_noise)
            added_noise = torch.tensor(added_noise)
            # print(added_noise)
            pred +=added_noise 
            # print(pred)
        
        acc = accuracy_score(nm_labels, np.argmax(pred.detach().numpy(), axis=1))
        print('Target model test accuracy = ', acc)
        

        #menber,nonmemer for training
        mtrainset_loader, nmtrainset_loader = self.attack_datahandler.load_train_datasetLoader()

        #menber,nonmemer for testing
        mtestset_loader, nmtestset_loader = self.attack_datahandler.load_test_datasetLoader()

        # main training procedure begins
        best_accuracy = 0
        
        
        
        attackmodel_opt = self.optimizer(params=self.attackmodel.parameters(),lr=self.learning_rate)
        
        
        scheduler = lr_scheduler.LambdaLR(optimizer=attackmodel_opt, lr_lambda=lambda epoch:0.95**epoch)
        print(list(self.attackmodel.parameters())[-1])      
        
        
        for e in range(self.epochs): 
            print('this is epoch ',e)
            zipped = zip(mtrainset_loader, nmtrainset_loader)
            for(_, ((mfeatures, mlabels), (nmfeatures, nmlabels))) in enumerate(zipped):
                # Getting outputs of forward pass of attack model             
                moutputs = self.forward_pass(model, mfeatures, mlabels)
                nmoutputs = self.forward_pass(model, nmfeatures, nmlabels)

                # Computing the true values for loss function according
                memtrue = torch.ones(moutputs.shape)
                nonmemtrue = torch.zeros(nmoutputs.shape)
                
                target = torch.cat((memtrue, nonmemtrue), 0)
                probs =  torch.cat((moutputs, nmoutputs), 0)
#                print(target)
#                print(probs)
                
                attackloss = mse(target, probs)
#                print('this is loss:')
#                print(attackloss)
                
                attackmodel_opt.zero_grad()   
                attackloss.backward(retain_graph=True)
                attackmodel_opt.step()
            scheduler.step()
#            print(list(self.attackmodel.parameters()))
                
            # Calculating Attack accuracy            
            attack_accuracy = self.attack_accuracy(mtestset_loader, nmtestset_loader)
            if attack_accuracy > best_accuracy:
                    best_accuracy = attack_accuracy
                    
            print("Epoch {} over :"
                  "Attack test accuracy: {}, Best accuracy : {}"
                  .format(e, attack_accuracy, best_accuracy))  
            if (self.epochs - e <=20):
                self.last20_epochs.append(attack_accuracy)
            
            self.logger.info("Epoch {} over,"
                                 "Attack loss: {},"
                                 "Attack accuracy: {}"
                                 .format(e, attackloss, attack_accuracy))
            print(list(self.attackmodel.parameters())[-1]) 
            
#            # ????????????
#            data = None
#            if os.path.isfile("logs/attack/results") and os.stat("logs/attack/results").st_size > 0:
#                with open('logs/attack/results', 'r+') as json_file:
#                    data = json.load(json_file)
#                    if data:
#                        data = data['result']
#                    else:
#                        data = []
#            if not data:
#                data = []
#            data.append(
#                {self.model_name: {'target_acc': float(acc), 'attack_acc': float(best_accuracy)}})
#            with open('logs/attack/results', 'w+') as json_file:
#                json.dump({'result': data}, json_file)
    
            # logging best attack accuracy
            self.logger.info("Best attack accuracy %.2f%%\n\n",
                             100 * best_accuracy)  
                            

    def attack_accuracy(self, members, nonmembers):
        """
        Computes attack accuracy of the attack model.
        """
        model = self.target_train_model
        
        zipped = zip(members, nonmembers)
        best_accuracy_batch = 0
        for (_,(membatch, nonmembatch)) in enumerate(zipped):
            mfeatures, mlabels = membatch
            nmfeatures, nmlabels = nonmembatch
            
            # Computing the membership probabilities
            mprobs = self.forward_pass(model, mfeatures, mlabels)
            nonmprobs = self.forward_pass(model, nmfeatures, nmlabels)
            probs = torch.cat((mprobs, nonmprobs), 0)

            # true and false matrix
            target_ones = torch.ones(mprobs.shape, dtype=bool)
            target_zeros = torch.zeros(nonmprobs.shape, dtype=bool)
            target = torch.cat((target_ones, target_zeros), 0)
            
            probs_trans = []
            for i in range(len(probs)):     
                probs_trans.append(probs[i] > torch.tensor(0.5))
            probs_trans = torch.tensor(np.stack(probs_trans))

            acc_result = accuracy_score(target,probs_trans)
            print(acc_result)
            if acc_result>best_accuracy_batch:
                best_accuracy_batch = acc_result
                
        return best_accuracy_batch


    def grad_norm_diff(self): 
        #menber,nonmemer for testing
        mtestset_loader, nmtestset_loader = self.attack_datahandler.load_test_datasetLoader()
        model = self.target_train_model
        zipeed = zip(mtestset_loader, nmtestset_loader)
        mresult = []
        nmresult = []
        # member ???????????????    
        for (_,((mfeatures, mlabels),(nmfeatures, nmlabels))) in enumerate(zipeed):
            mgradientnorm = self.get_gradient_norms(
                model, mfeatures, mlabels)
            nmgradientnorm = self.get_gradient_norms(
                model, nmfeatures, nmlabels)
            mresult.append(mgradientnorm)
            nmresult.append(nmgradientnorm)
        memberresult = np.sum(mresult, axis=0)
        nonmemberresult = np.sum(nmresult, axis=0)
        diff_nm_m = nonmemberresult - memberresult
        return diff_nm_m


            
                                                                   
        

        
        
        
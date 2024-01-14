import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size = 128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        model = model.eval()

        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)


        if self.args['use_RP']:
            Y=target2onehot(label_list,self.total_classnum)
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(embedding_list@ self._network.fc.W_rand.cpu())
            else:
                Features_h=embedding_list
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device='cuda')
        else:
            class_list = np.unique(self.train_dataset.labels)
            proto_list = []
            for class_index in class_list:
                # print('Replacing...',class_index)
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                self._network.fc.weight.data[class_index] = proto
        return model

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
   
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.total_classnum = data_manager.nb_classes
        if self.args['use_RP']:
            #temporarily remove RP weights due to how update_fc works
            #the information we need to create the output weights from past layers is stored in self.G and self.Q
            del self._network.fc
            self._network.fc=None
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        if self._cur_task == 0 and self.args['use_RP']:
            self.setup_RP()
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def setup_RP(self):
        self._network.fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network._feature_dim=M
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.fc.W_rand) #make a copy that gets passed each time the head is replaced
        else:
            #no RP, only decorrelation
            M=self._network.fc.in_features #this M is L in the paper
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)
    

   
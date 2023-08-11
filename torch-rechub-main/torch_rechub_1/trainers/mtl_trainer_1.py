import os
import tqdm
import numpy as np
import torch
import csv
import torch.nn as nn
import time
from ..basic.callback import EarlyStopper
from ..utils.data import get_loss_func, get_metric_func
from ..models.multi_task import ESMM
from ..utils.mtl_1 import shared_task_layers, gradnorm, MetaBalance
from torch_rechub_1.trainers.mymodel import AcouModel
torch.multiprocessing.set_sharing_strategy('file_system')

class MTLTrainer(object):
    """A trainer for multi task learning.

    Args:
        model (nn.Module): any multi task learning model.
        task_types (list): types of tasks, only support ["classfication", "regression"].
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        adaptive_params (dict): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`. 
        n_epoch (int): epoch number of training.
        earlystop_taskid (int): task id of earlystop metrics relies between multi task (default = 0).
        earlystop_patience (int): how long to wait after last time validation auc improved (default = 10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        task_types,
        parameter_num,
        alpha,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
        adaptive_strategy=None,

    ):
        self.model = model
        self.adaptive_strategy=adaptive_strategy
        if gpus is None:
            gpus = []
        if optimizer_params is None:
            optimizer_params = {
                "lr": 1e-3,
                "weight_decay": 1e-5
            }
        self.task_types = task_types
        self.n_task = len(task_types)
        self.loss_weight = None
        self.adaptive_method = None
        self.n_epoch = n_epoch
        self.parameter_num = parameter_num
        self.alpha = alpha
        if adaptive_params is not None:
            if adaptive_params["method"] == "uwl":
                self.adaptive_method = "uwl"
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "metabalance":
                self.adaptive_method = "metabalance"
                share_layers, task_layers = shared_task_layers(self.model)
                self.meta_optimizer = MetaBalance(share_layers)
                self.share_optimizer = optimizer_fn(share_layers, **optimizer_params)
                self.task_optimizer = optimizer_fn(task_layers, **optimizer_params)
            elif adaptive_params["method"] == "gradnorm":
                self.adaptive_method = "gradnorm"
                self.alpha = adaptive_params.get("alpha", 0.16)
                share_layers = shared_task_layers(self.model)[0]
                #gradnorm calculate the gradients of each loss on the last fully connected shared layer weight(dimension is 2)
                for i in range(len(share_layers)):
                    if share_layers[-i].ndim == 2:
                        self.last_share_layer = share_layers[-i]
                        break
                self.initial_task_loss = None
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "mymodel":
                self.adaptive_method = "mymodel"
                self.every_epoch_loss_lst = torch.zeros([self.n_epoch + 1, self.n_task])

        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default Adam optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]

        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience)

        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_path = model_path

    def train_one_epoch(self, data_loader,epoch_i,model_name):
        self.model.train()
        total_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for iter_i, (x_dict, ys) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            ys = ys.to(self.device)
            y_preds = self.model(x_dict)
            loss_list = [self.loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]
            if isinstance(self.model, ESMM):
                loss = sum(loss_list[1:])  #ESSM only compute loss for ctr and ctcvr task
            else:
                if self.adaptive_method != None:
                    if self.adaptive_method == "uwl":
                        loss = 0
                        for loss_i, w_i in zip(loss_list, self.loss_weight):
                            w_i = torch.clamp(w_i, min=0)
                            loss += 2 * loss_i * torch.exp(-w_i) + w_i
                    elif self.adaptive_method == "mymodel":
                        self.every_epoch_loss_lst[epoch_i + 1] = torch.tensor(loss_list)
                        t_1_loss_list = self.every_epoch_loss_lst[epoch_i]
                        if model_name == 'MMOE' and self.adaptive_strategy=='both':
                            loss,impact_value,weight_value = AcouModel().Acou(self.parameter_num, epoch_i, loss_list, t_1_loss_list,self.alpha,model_name,self.adaptive_strategy)
                        else:
                            loss = AcouModel().Acou(self.parameter_num,epoch_i,loss_list,t_1_loss_list,self.alpha,model_name,self.adaptive_strategy)
                else:
                    loss = sum(loss_list) / self.n_task
            if self.adaptive_method == 'metabalance':
                self.share_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.meta_optimizer.step(loss_list)
                self.share_optimizer.step()
                self.task_optimizer.step()
            elif self.adaptive_method == "gradnorm":
                self.optimizer.zero_grad()
                if self.initial_task_loss is None:
                    self.initial_task_loss = [l.item() for l in loss_list]
                gradnorm(loss_list, self.loss_weight, self.last_share_layer, self.initial_task_loss, self.alpha)
                self.optimizer.step()
                # renormalize
                loss_weight_sum = sum([w.item() for w in self.loss_weight])
                normalize_coeff = len(self.loss_weight) / loss_weight_sum
                for w in self.loss_weight:
                    w.data = w.data * normalize_coeff
            else:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += np.array([l.item() for l in loss_list])
        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        print("train loss: ", log_dict)
        if self.loss_weight:
            print("loss weight: ", [w.item() for w in self.loss_weight])
        if model_name == 'MMOE' and self.adaptive_strategy=='both':
            return log_dict,impact_value,weight_value
        else:
            return log_dict

    def fit(self, train_dataloader,val_dataloader,save_model_name):
        model_name = save_model_name.split('_')[0]
        root_path = os.path.join((model_path ,model_name)
        os.makedirs(root_path, exist_ok='True')
        try:
            adaptive_param = save_model_name.split('_')[1]
        except:
            csv_data_name = model_name + '.csv'
        else:
            if self.adaptive_method != 'mymodel':
                csv_data_name = model_name + '_' + adaptive_param + '.csv'
            else:
                if self.adaptive_strategy in ('both','loss_combine'):
                    csv_data_name =  model_name + '_' + adaptive_param + '_' + self.adaptive_strategy+'_alpha_'+str(self.alpha)+ '.csv'
                else:
                    csv_data_name = model_name + '_' + adaptive_param + '_' + self.adaptive_strategy+'.csv'
        csv_data_path = os.path.join(root_path,csv_data_name)
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        csv_data = [['date','epoch','ctr_AUC','atr_AUC','cvr_AUC','ctr_loss','atr_loss','cvr_loss']]
        if model_name == 'MMOE' and self.adaptive_strategy == 'both':
            csv_data = [['date','epoch','ctr_AUC','atr_AUC','cvr_AUC','ctr_loss','atr_loss','cvr_loss','ctr_impact_atr','ctr_impact_cvr','atr_impact_cvr','ctr_weight','atr_weight','cvr_weight']] 
        for epoch_i in range(self.n_epoch):
            if model_name == 'MMOE' and self.adaptive_strategy== 'both':
                train_loss_dict,impact_value,weight_value = self.train_one_epoch(train_dataloader, epoch_i, model_name)
            else:
                train_loss_dict = self.train_one_epoch(train_dataloader, epoch_i, model_name)
            train_loss_list = list(train_loss_dict.values())
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            scores=self.evaluate(self.model,val_dataloader)
            print('epoch:', epoch_i, 'validation scores: ', scores)
            if self.early_stopper.stop_training(scores[self.earlystop_taskid], self.model.state_dict()):
                print('validation best auc of main task %d: %.6f' %
                      (self.earlystop_taskid, self.early_stopper.best_auc))
                self.model.load_state_dict(self.early_stopper.best_weights)
                break
            scores.insert(0, date)
            scores.insert(1, epoch_i)
            if model_name == 'MMOE' and self.adaptive_strategy == 'both':
                row = scores + train_loss_list + impact_value + weight_value
            else:
                row = scores + train_loss_list
            csv_data.append(row)
        with open(csv_data_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        torch.save(self.model.state_dict(), os.path.join(root_path,'%s.pth'%csv_data_name.split('.csv')[0]))  #save best auc model
        # torch.save(self.model.state_dict(), os.path.join(self.model_path,
        #                 "%s.pth"%save_model_name))  #save best auc model

    def evaluate(self,model,data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)
        scores = [self.evaluate_fns[i](targets[:, i], predicts[:, i]) for i in range(self.n_task)]
        return scores

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_preds = model(x_dict)
                predicts.extend(y_preds.tolist())
        return predicts
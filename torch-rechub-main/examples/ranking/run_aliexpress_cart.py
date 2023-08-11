import sys

sys.path.append("../..")

import pandas as pd
import torch
import os
import time
import csv
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub_1.trainers.mtl_trainer_1 import MTLTrainer
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator
torch.multiprocessing.set_sharing_strategy('file_system')


#def get_aliexpress_data_dict(data_path='./data/aliexpress'):
#    df_train = pd.read_csv(data_path + '/aliexpress_train_sample_add_data.csv')
#    df_test = pd.read_csv(data_path + '/aliexpress_test_sample_add_data.csv')
#    print("train : test = %d %d" % (len(df_train), len(df_test)))

def get_aliexpress_data_dict(data_path='/home/caofeng/dataset/AliExpress_US/'):
    train_raw_data = pd.read_csv(data_path + 'train_add_data.csv')
    test_raw_data = pd.read_csv(data_path + 'test_add_data.csv')
    select_rate = 1
    train_num_rows = int(len(train_raw_data) *select_rate)
    test_num_rows = int(len(test_raw_data) *select_rate)
    df_train = train_raw_data.sample(n=train_num_rows)
    df_test = test_raw_data.sample(n=test_num_rows)
    print('origin train total num: %d,origin test total num: %d'%(len(train_raw_data),len(test_raw_data)))
    print('origin train click label rate: %f,origin test click label rate: %f'%(len(train_raw_data[train_raw_data['click']==1])/len(train_raw_data[train_raw_data['click']==0]),len(test_raw_data[test_raw_data['click']==1])/len(test_raw_data[test_raw_data['click']==0])))
    print('origin train carts label rate: %f,origin test carts label rate: %f'%(len(train_raw_data[train_raw_data['carts']==1])/len(train_raw_data[train_raw_data['carts']==0]),len(test_raw_data[test_raw_data['carts']==1])/len(test_raw_data[test_raw_data['carts']==0])))
    print('origin train conversion label rate: %f,origin test conversion label rate: %f'%(len(train_raw_data[train_raw_data['conversion']==1])/len(train_raw_data[train_raw_data['conversion']==0]),len(test_raw_data[test_raw_data['conversion']==1])/len(test_raw_data[test_raw_data['conversion']==0])))

    print('selected train total num: %d,selected test total num: %d'%(len(df_train),len(df_test)))
    print('selected train click label rate: %f,selected test click label rate: %f'%(len(df_train[df_train['click']==1])/len(df_train[df_train['click']==0]),len(df_test[df_test['click']==1])/len(df_test[df_test['click']==0])))
    print('selected train carts label rate: %f,selected test carts label rate: %f'%(len(df_train[df_train['carts']==1])/len(df_train[df_train['carts']==0]),len(df_test[df_test['carts']==1])/len(df_test[df_test['carts']==0])))
    print('selected train conversion label rate: %f,selected test conversion label rate: %f'%(len(df_train[df_train['conversion']==1])/len(df_train[df_train['conversion']==0]),len(df_test[df_test['conversion']==1])/len(df_test[df_test['conversion']==0])))
    
    train_idx = df_train.shape[0]
    data = pd.concat([df_train, df_test], axis=0)
    col_names = data.columns.values.tolist()
    sparse_cols = [name for name in col_names if name.startswith("categorical")]  #categorical
    dense_cols = [name for name in col_names if name.startswith("numerical")]  #numerical
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    label_cols = ["conversion","carts","click"]

    used_cols = sparse_cols + dense_cols
    features = [SparseFeature(col, data[col].max()+1, embed_dim=5)for col in sparse_cols] \
                + [DenseFeature(col) for col in dense_cols]
    x_train, y_train = {name: data[name].values[:train_idx] for name in used_cols}, data[label_cols].values[:train_idx]
    x_test, y_test = {name: data[name].values[train_idx:] for name in used_cols}, data[label_cols].values[train_idx:]
    return features, x_train, y_train, x_test, y_test


#def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, adaptive_param,adaptive_strategy,alpha):
    torch.manual_seed(seed)
    features, x_train, y_train, x_test, y_test = get_aliexpress_data_dict()
    task_types = ["classification", "classification", "classification"]
    if model_name == "SharedBottom":
    #        model = SharedBottom(features, task_types, bottom_params={"dims": [192, 96, 48]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
        bottom_params = {"dims": [117]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
        parameter_num = sum(bottom_params.get('dims'))+sum([sum(item.get('dims')) for item in tower_params_list])
        model = SharedBottom(features, task_types, bottom_params=bottom_params, tower_params_list=tower_params_list)
    elif model_name == "MMOE":
    #        model = MMOE(features, task_types, n_expert=9, expert_params={"dims": [64, 32, 16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}, {"dims": [8]}])
        task_types = ["classification", "classification", "classification"]
        expert_params = {"dims": [16]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
        parameter_num = sum(expert_params['dims'])+sum([sum(item.get('dims')) for item in tower_params_list])
        model = MMOE(features, task_types, 8, expert_params=expert_params, tower_params_list=tower_params_list)
    elif model_name == "PLE": #        model = PLE(features, task_types, n_level=1, n_expert_specific=1, n_expert_shared=1, expert_params={"dims": [64, 32, 16], "output_layer": False}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
        expert_params = {"dims": [16]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
        parameter_num = sum(expert_params.get('dims'))+sum([sum(item.get('dims')) for item in tower_params_list])
        model = PLE(features, task_types, n_level=2, n_expert_specific=3, n_expert_shared=2, expert_params=expert_params, tower_params_list=tower_params_list)
    elif model_name == "AITM":
    #        model = AITM(features, n_task=2, bottom_params={"dims": [128, 64, 32]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
        bottom_params = {"dims": [32,16]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
        parameter_num = sum(bottom_params.get('dims'))+sum([sum(item.get('dims')) for item in tower_params_list])
        model = AITM(features, 3, bottom_params=bottom_params, tower_params_list=tower_params_list)
#    print("No. of Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
#    parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_test, y_val=y_test, x_test=x_test, y_test=y_test, batch_size=batch_size)

    root_path = os.path.join(save_dir,model_name)
    if adaptive_param != None:
        save_model_name = model_name+'_'+adaptive_param
        if adaptive_param != 'mymodel':
            csv_data_name = os.path.join(root_path,model_name+'_'+adaptive_param+'.csv')
        else:
            if adaptive_strategy in ('both','loss_combine'):
                csv_data_name = os.path.join(root_path,model_name+'_'+adaptive_param+'_'+adaptive_strategy+'_alpha_'+str(alpha)+'.csv')
            else:
                csv_data_name = os.path.join(root_path,model_name + '_' + adaptive_param + '_' + adaptive_strategy + '.csv')
    else:
        save_model_name = model_name
        csv_data_name = os.path.join(root_path,model_name+'.csv')
    mtl_trainer = MTLTrainer(model, task_types=task_types,parameter_num=parameter_num, alpha = alpha,optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},adaptive_params={"method":  adaptive_param}, n_epoch=epoch, earlystop_patience=30,device=device,model_path=save_dir,adaptive_strategy=adaptive_strategy)
    mtl_trainer.fit(train_dataloader,val_dataloader,save_model_name)
    auc = mtl_trainer.evaluate(mtl_trainer.model,test_dataloader)
    print(f'test auc: {auc}')
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    auc.insert(0,date)
    auc.insert(1,'test_AUC')
    with open(csv_data_name, 'a', newline='') as g:
        writer = csv.writer(g)
        writer.writerow(auc)
    print('*'*100)
    #adaptive weight loss:
    #mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, adaptive_params={"method": "uwl"}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir)

#    mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=1, device=device, model_path=save_dir)
#    mtl_trainer.fit(train_dataloader, val_dataloader)
#    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
#    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SharedBottom')
    parser.add_argument('--epoch', type=int, default=5)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10240)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--adaptive_param', default= None)
    parser.add_argument('--adaptive_strategy', default= None)
    parser.add_argument('--alpha', type=int, default=1)
    args = parser.parse_args()
#    main(args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
    main(args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.adaptive_param, args.adaptive_strategy, args.alpha)
"""
python run_aliexpress.py --model_name SharedBottom
python run_aliexpress.py --model_name ESMM
python run_aliexpress.py --model_name MMOE
python run_aliexpress.py --model_name PLE
python run_aliexpress.py --model_name AITM
"""

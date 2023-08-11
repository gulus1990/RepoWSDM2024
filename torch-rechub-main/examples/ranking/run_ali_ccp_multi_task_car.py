import sys

sys.path.append("../..")

import pandas as pd
import torch
import csv
import os
import time
from torch_rechub_1.models.multi_task import SharedBottom, MMOE,AITM,ESMM,PLE
from torch_rechub_1.trainers.mtl_trainer_1 import MTLTrainer
from torch_rechub_1.basic.features import DenseFeature, SparseFeature
from torch_rechub_1.utils.data import DataGenerator


#def get_ali_ccp_data_dict(model_name, data_path='./data/ali-ccp'):
#    df_train = pd.read_csv(data_path + '/ali_ccp_train_sample_add_data.csv')
#    df_val = pd.read_csv(data_path + '/ali_ccp_val_sample_add_data.csv')
#    df_test = pd.read_csv(data_path + '/ali_ccp_test_sample_add_data.csv')
def get_ali_ccp_data_dict(model_name, data_path='/tmp/gaizhy/aliccp/'):
    train_raw_data = pd.read_csv(data_path + 'ali_ccp_train_add_data.csv')
    test_raw_data = pd.read_csv(data_path + 'ali_ccp_test_add_data.csv')
    val_raw_data = pd.read_csv(data_path + 'ali_ccp_val_add_data.csv')
    select_rate = 1
    train_num_rows = int(len(train_raw_data) *select_rate)
    test_num_rows = int(len(test_raw_data) *select_rate)
    val_num_rows = int(len(test_raw_data) *select_rate)
    df_train = train_raw_data.sample(n=train_num_rows)
    df_test = test_raw_data.sample(n=test_num_rows)
    df_val = val_raw_data.sample(n=test_num_rows)
    print('origin train total num: %d,origin test total num: %d,origin val total num: %d'%(len(train_raw_data),len(test_raw_data),len(val_raw_data)))
    print('origin train click label rate: %f,origin test click label rate: %f,origin val click label rate: %f'%(len(train_raw_data[train_raw_data['click']==1])/len(train_raw_data[train_raw_data['click']==0]),len(test_raw_data[test_raw_data['click']==1])/len(test_raw_data[test_raw_data['click']==0]),len(val_raw_data[val_raw_data['click']==1])/len(val_raw_data[val_raw_data['click']==0])))
    print('origin train carts label rate: %f,origin test carts label rate: %f,origin val carts label rate: %f'%(len(train_raw_data[train_raw_data['carts']==1])/len(train_raw_data[train_raw_data['carts']==0]),len(test_raw_data[test_raw_data['carts']==1])/len(test_raw_data[test_raw_data['carts']==0]),len(val_raw_data[val_raw_data['carts']==1])/len(val_raw_data[val_raw_data['carts']==0])))
    print('origin train purchase label rate: %f,origin test purchase label rate: %f,origin val purchase label rate: %f'%(len(train_raw_data[train_raw_data['purchase']==1])/len(train_raw_data[train_raw_data['purchase']==0]),len(test_raw_data[test_raw_data['purchase']==1])/len(test_raw_data[test_raw_data['purchase']==0]),len(val_raw_data[val_raw_data['purchase']==1])/len(val_raw_data[val_raw_data['purchase']==0])))

    print('selected train total num: %d,selected test total num: %d,selected val total num: %d'%(len(df_train),len(df_test),len(df_val)))
    print('selected train click label rate: %f,selected test click label rate: %f,selected val click label rate: %f'%(len(df_train[df_train['click']==1])/len(df_train[df_train['click']==0]),len(df_test[df_test['click']==1])/len(df_test[df_test['click']==0]),len(df_val[df_val['click']==1])/len(df_val[df_val['click']==0])))
    print('selected train carts label rate: %f,selected test carts label rate: %f,selected val carts label rate: %f'%(len(df_train[df_train['carts']==1])/len(df_train[df_train['carts']==0]),len(df_test[df_test['carts']==1])/len(df_test[df_test['carts']==0]),len(df_val[df_val['carts']==1])/len(df_val[df_val['carts']==0])))
    print('selected train purchase label rate: %f,selected test purchase label rate: %f,selected val purchase label rate: %f'%(len(df_train[df_train['purchase']==1])/len(df_train[df_train['purchase']==0]),len(df_test[df_test['purchase']==1])/len(df_test[df_test['purchase']==0]),len(df_val[df_val['purchase']==1])/len(df_val[df_val['purchase']==0])))

#    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    #task 1 (as cvr): main task, purchase prediction
    #task 2(as ctr): auxiliary task, click prediction
    data.rename(columns={'purchase': 'cvr_label', 'carts': 'atr_label', 'click': 'ctr_label'}, inplace=True)
    # data["ctcvr_label"] = data['cvr_label'] * data['ctr_label']

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['cvr_label', 'ctr_label', 'atr_label']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    #define dense and sparse features
    if model_name == "ESMM":
        label_cols = ['cvr_label','ctr_label', 'atr_label','ctcvr_label']  #the order of 3 labels must fixed as this
        #ESMM only for sparse features in origin paper
        item_cols = ['129', '205', '206', '207', '210', '216']  #assumption features split for user and item
        user_cols = [col for col in sparse_cols if col not in item_cols]
        user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
        item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]
        x_train, y_train = {name: data[name].values[:train_idx] for name in sparse_cols}, data[label_cols].values[:train_idx]
        x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in sparse_cols}, data[label_cols].values[train_idx:val_idx]
        x_test, y_test = {name: data[name].values[val_idx:] for name in sparse_cols}, data[label_cols].values[val_idx:]
        return user_features, item_features, x_train, y_train, x_val, y_val, x_test, y_test
    else:
        label_cols = ['ctr_label','atr_label','cvr_label']  #the order of labels can be any
        used_cols = sparse_cols + dense_cols
        features = [SparseFeature(col, data[col].max()+1, embed_dim=4)for col in sparse_cols] \
                   + [DenseFeature(col) for col in dense_cols]
        x_train, y_train = {name: data[name].values[:train_idx] for name in used_cols}, data[label_cols].values[:train_idx]
        x_val, y_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}, data[label_cols].values[train_idx:val_idx]
        x_test, y_test = {name: data[name].values[val_idx:] for name in used_cols}, data[label_cols].values[val_idx:]
        return features, x_train, y_train, x_val, y_val, x_test, y_test

def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed, adaptive_param,adaptive_strategy,alpha):
#def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir,seed):
    torch.manual_seed(seed)
    if model_name == "SharedBottom":
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification", "classification"]
        bottom_params = {"dims": [117]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
#        parameter_num = sum(bottom_params.get('dims'))+sum([sum(item.get('dims')) for item in tower_params_list])
        model = SharedBottom(features, task_types, bottom_params=bottom_params, tower_params_list=tower_params_list)
    elif model_name == "ESMM":
        user_features, item_features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification", "classification", "classification"]  #cvr,ctr,ctcvr
        cvr_params = {"dims": [16, 8]}
        ctr_params = {"dims": [16, 8]}
        atr_params = {"dims": [16, 8]}
#        parameter_num = sum(cvr_params.get('dims'))+sum(ctr_params.get('dims'))+sum(atr_params.get('dims'))
        model = ESMM(user_features, item_features,cvr_params=cvr_params,ctr_params=ctr_params,atr_params=atr_params)
    elif model_name == "MMOE":
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification", "classification"]
        expert_params = {"dims": [16]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
#        parameter_num = sum(expert_params['dims'])+sum([sum(item.get('dims')) for item in tower_params_list])
        model = MMOE(features, task_types, 9, expert_params=expert_params, tower_params_list=tower_params_list)
    elif model_name == "PLE":
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        task_types = ["classification", "classification", "classification"]
        expert_params = {"dims": [32,16]}
        tower_params_list = [{"dims": [8,4,2]}, {"dims": [8,4,2]}, {"dims": [8,4,2]}]
        print(expert_params,tower_params_list)
#        parameter_num = sum(expert_params.get('dims'))+sum([sum(item.get('dims')) for item in tower_params_list])
        model = PLE(features, task_types, n_level=1, n_expert_specific=3, n_expert_shared=1, expert_params=expert_params, tower_params_list=tower_params_list)
    elif model_name == "AITM":
        task_types = ["classification", "classification", "classification"]
        features, x_train, y_train, x_val, y_val, x_test, y_test = get_ali_ccp_data_dict(model_name)
        bottom_params = {"dims": [32,16]}
        tower_params_list = [{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
#        parameter_num = sum(bottom_params.get('dims'))+sum([sum(item.get('dims')) for item in tower_params_list])
        model = AITM(features, 3, bottom_params=bottom_params, tower_params_list=tower_params_list)
    print("No. of Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=batch_size)

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
    mtl_trainer = MTLTrainer(model, task_types=task_types,parameter_num=parameter_num, alpha = alpha,optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},adaptive_params={"method": adaptive_param}, n_epoch=epoch, earlystop_patience=30,device=device,model_path=save_dir,adaptive_strategy=adaptive_strategy)
    mtl_trainer.fit(train_dataloader,val_dataloader,save_model_name)
    auc = mtl_trainer.evaluate(mtl_trainer.model,test_dataloader)
    print(f'test auc: {auc}')

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    auc.insert(0,date)
    auc.insert(1,'test_AUC')
    with open(csv_data_name, 'a', newline='') as g:
        writer = csv.writer(g)
        writer.writerow(auc)

#    mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},adaptive_params={"method": adaptive_param}, n_epoch=epoch, earlystop_patience=30, device=device, model_path=save_dir) 
#    mtl_trainer.fit(train_dataloader, val_dataloader,save_model_name)
    
#    mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=30, device=device, model_path=save_dir)
#    mtl_trainer.fit(train_dataloader, val_dataloader,model_name)
#    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
#    
#    auc_file_path =save_dir+'%s_validation_auv.csv'%save_model_name
#    auc_file_head=['ctr_validation_auc','atr_validation_auc','cvr_validation_auc']
#    print(f'test auc: {auc}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='MMOE')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:1')  #cuda:0
    parser.add_argument('--save_dir', default='./model_result')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--adaptive_param', default='mymodel')
    parser.add_argument('--adaptive_strategy', default='both')
    parser.add_argument('--alpha', default=1)
    args = parser.parse_args()
    main(args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed, args.adaptive_param, args.adaptive_strategy, args.alpha)
#    main(args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_ali_ccp_multi_task.py --model_name SharedBottom
python run_ali_ccp_multi_task.py --model_name ESMM
python run_ali_ccp_multi_task.py --model_name MMOE
python run_ali_ccp_multi_task.py --model_name PLE
python run_ali_ccp_multi_task.py --model_name AITM
python run_ali_ccp_multi_task_car.py --model_name MMOE --epoch 15 --adaptive_param mymodel --adaptive_strategy both
"""


import math
import torch
import numpy as np

class AcouModel:
    def AIC(self,parameter_num,epoch,t_loss,t_1_loss):
        if epoch == 0:
            aic = abs((parameter_num - math.log(t_loss)) / (parameter_num - 1e-10))
        else:
            aic = abs((parameter_num-math.log(t_loss))/(parameter_num-math.log(t_1_loss)))
        return aic

    def impact_decay(self,alpha,wi,wi_1):
        impact = wi-wi_1*math.exp(-alpha)
        return impact
    
    def genDifference(self, JX, JY):
        JXY = sum([ 1 for i in range(len(JX)) if JX[i]==JY[i]])
        deno = math.sqrt(np.dot(JX, JY))
        return -math.log(JXY/deno)

    def Acou(self,parameter_num,epoch,t_loss_list,t_1_loss_list,alpha,model_name,adaptive_strategy):
        task_n = len(t_loss_list)
        t_noeffect_loss = [t_loss_list[0] if i == 0 else self.impact_decay(alpha,t_loss_list[i],t_loss_list[i-1]) for i in range(task_n)]
        t_1_noeffect_loss = [t_1_loss_list[0] if i == 0 else self.impact_decay(alpha,t_1_loss_list[i],t_1_loss_list[i-1]) for i in range(task_n)]
        total_loss = 0
        
        if adaptive_strategy == 'loss_combine':
            for i in range(task_n):
                wi = 1
                wi_1 = 1
                if i < task_n - 1:
                    weight = self.impact_decay(alpha, wi, wi_1)
                    total_loss += weight*t_loss_list[i]
                else:
                    total_loss += t_loss_list[i]
            return total_loss
        
        # for future work
        if adaptive_strategy == 'both':
            denominator = {}
            for i in range(task_n):
                denominator_iteration = self.AIC(parameter_num,epoch,t_noeffect_loss[i],t_1_noeffect_loss[i])
                denominator['epoch_denominator_{}'.format(i)] = denominator_iteration
            denominator_final_number = sum([denominator[i] for i in denominator])
            Weight_list = []
            for i in range(task_n):
                if i< task_n-1:
                    wi = self.AIC(parameter_num,epoch,t_noeffect_loss[i],t_1_noeffect_loss[i])/denominator_final_number
                    wi_1 = self.AIC(parameter_num, epoch, t_noeffect_loss[i+1], t_1_noeffect_loss[i+1])/denominator_final_number
                    part_loss = self.impact_decay(alpha, wi, wi_1)*t_loss_list[i]
                else:
                    wi = self.AIC(parameter_num, epoch, t_noeffect_loss[i], t_1_noeffect_loss[i]) / denominator_final_number
                    part_loss = wi*t_loss_list[i]
                total_loss+=part_loss
                Weight_list.append(wi)
            if model_name == 'MMOE':
                # task1_task2 = self.impact_decay(alpha,t_loss_list[0],t_loss_list[1])
                # task1_task3 = self.impact_decay(alpha, t_loss_list[0],t_loss_list[2],gap=2)
                # task2_task3 = self.impact_decay(alpha, t_loss_list[1],t_loss_list[2])
                task1_task2 = t_loss_list[0]*math.exp(-1*alpha)
                task1_task3 = t_loss_list[0]*math.exp(-2*alpha)
                task2_task3 = t_loss_list[1]*math.exp(-1*alpha)
                impact_list = torch.Tensor([task1_task2,task1_task3,task2_task3]).tolist()
                Weight_list = torch.Tensor(Weight_list).tolist()
                return total_loss,impact_list,Weight_list
            else:
                return total_loss

        # for future work
        if adaptive_strategy == 'loss_balance':
            denominator = {}
            for i in range(task_n):
                denominator_iteration = self.AIC(parameter_num, epoch, t_noeffect_loss[i], t_1_noeffect_loss[i])
                denominator['epoch_denominator_{}'.format(i)] = denominator_iteration
            denominator_final_number = sum([denominator[i] for i in denominator])
            for i in range(task_n):
                wi = self.AIC(parameter_num, epoch, t_noeffect_loss[i], t_1_noeffect_loss[i]) / denominator_final_number
                total_loss += wi*t_loss_list[i]
            return total_loss
        
        
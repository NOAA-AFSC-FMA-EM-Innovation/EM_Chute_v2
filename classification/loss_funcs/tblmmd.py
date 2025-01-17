from loss_funcs.mmd import MMDLoss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np



class TBLMMDLoss(MMDLoss, LambdaSheduler):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, 
                    gamma=1.0, max_iter=1000, **kwargs):
        '''
        Local MMD
        '''
        super(TBLMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class
        self.classlist = kwargs['source_cnt'] 
        self.classlist_tgt = [0]*self.num_class
        self.beta = 0.999


        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.classlist])
        self.class_balanced_weight = self.class_balanced_weight / np.sum(self.class_balanced_weight) 
        self.class_balanced_weight = np.clip(self.class_balanced_weight/self.num_class, 1, 5)


    def forward(self, source, target, source_label, target_logits, kl_div):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        
        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits, kl_div)
            weight_ss = torch.from_numpy(weight_ss).cuda() # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(source, target,
                                    kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()

            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss, lamb
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )

 
            loss = loss * lamb
            return loss, lamb
        
    def update(self,cnt):
        for i in range(len(cnt)):
            self.classlist_tgt[i] = max(1,cnt[i])

        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta**N) for N in self.classlist])
        cbw2 = np.array([(1-self.beta** N)/(1- self.beta) for N in self.classlist_tgt])

        self.class_balanced_weight = self.class_balanced_weight/np.sum(self.class_balanced_weight)
        cbw2 = cbw2/np.sum(cbw2)
        
        self.class_balanced_weight = np.clip(np.multiply(self.class_balanced_weight, cbw2), 1, 5)

        
    def cal_weight(self, source_label, target_logits, kl_div):
        batch_size = source_label.size()[0]


        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label] # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()
        max_values = target_logits.max(1)[0].cpu().detach().numpy()
        

        target_logits = target_logits.cpu().data.numpy()
        target_logits_orig = target_logits

        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)

            
        count = 0
        for i in range(self.num_class): # (B, C)
            if i in set_s:# and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)
                
                ss = np.dot(s_tvec, s_tvec.T) # (B, B)
                weight_ss = weight_ss + ss*self.class_balanced_weight[i]
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt*self.class_balanced_weight[i]
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st*self.class_balanced_weight[i]   
                count += 1

        length = count
        #length = min(i for i in allcount if i>0)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

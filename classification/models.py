import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones
import numpy as np
import torch.nn.functional as F




import functools
from utils import *

class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_1_probs)
        #loss = 0.0
        loss = F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction="none")

        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction="none")
        loss = loss.sum(dim = 1)
     
        return (0.5 * loss)
    
class LADELoss(nn.Module):
    def __init__(self, num_classes=10, img_max=None, num_per_class = None, remine_lambda=0.1):
        super().__init__()

        self.img_num_per_cls = torch.tensor(num_per_class)
        self.prior = (self.img_num_per_cls / self.img_num_per_cls.sum()).cuda()
        self.balanced_prior = torch.tensor(1. / num_classes).float().cuda()
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())).cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss

class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes, num_per_class = None, prior=None, prior_txt=None):
        super().__init__()
        self.img_num_per_cls = torch.tensor(num_per_class).float().cuda()
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss
class CBCrossEntropy(nn.Module):
    
    def __init__(self, weight=None, 
                 beta = 0.999, reduction='mean', classcnt = [], numclasses = 15):
        nn.Module.__init__(self)
        self.numclasses = numclasses
        self.weight = weight
        self.beta = beta
        self.reduction = reduction
        self.classlist = classcnt
        #[2648, 101, 269, 39, 556, 594, 496, 588, 66, 262, 122, 286, 2486, 739, 72, 249, 41, 400, 316, 804]#[234,64,37,80,229,22,66,25,252,58,1281,16,5,18,15]
        self.classlist_tgt = [0]*self.numclasses
        self.num_classes = self.numclasses
        self.sigmoid = nn.Sigmoid()
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta ** N) for N in self.classlist])
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to("cuda")
        
    def forward(self, input_tensor, target_tensor):
        eps = 1e-6
        weights = (self.class_balanced_weight).to("cuda")

        loss = F.cross_entropy(input_tensor, target_tensor, weight=weights)
        return loss
    def update(self,cnt):
        for i in range(len(cnt)):
            self.classlist_tgt[i] = max(1,cnt[i])
        self.num_classes = self.numclasses
        self.class_balanced_weight = np.array([(1-self.beta)/(1- self.beta**N) for N in self.classlist])
        cbw2 = np.array([(1-self.beta** N)/(1- self.beta) for N in self.classlist_tgt])
        self.class_balanced_weight = np.multiply(self.class_balanced_weight, cbw2)
        self.class_balanced_weight = torch.FloatTensor(self.class_balanced_weight / np.sum(self.class_balanced_weight) * self.num_classes).to("cuda")

    
def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

def target_distribution_loss(source, target, source_label, target_clf):
    target_mean = torch.sum(target, dim = 0)

    target_cnt = torch.sum(target_clf, dim = 0)
    cnt_sum = 0
    for i in range(target_clf.shape[1]):
        if(torch.sum(source_label == i) > 0):
            cnt_sum += target_cnt[i]
    target_cnt = target_cnt/cnt_sum
    est_mean = torch.zeros(target_mean.shape[0]).cuda()
    for i in range(target_clf.shape[1]):
        if(torch.sum(source_label == i) > 0):
            est_mean = est_mean + torch.mean(source[source_label == i])*target_cnt[i]
    return torch.linalg.norm(est_mean - target_mean)/source.shape[1]
        
def teach_loss(input1, input2, source_label, target_clf):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    loss = 0
    num_classes = input1.size()[1]
    batch_size = input1.size()[0]
    for i in range(input1.size()[0]):
        input2_2  = input2[i,:].repeat(input1.size()[0], 1)
        cce = torch.mul(input1, torch.log(input2_2))*-1
        mse = (input1 - input2)**2
        temp = target_clf[:,0]
        for i in range(batch_size):
            temp[i] = target_clf[i,source_label[i]]
        weight = temp.reshape(batch_size, 1).repeat(1,input1.size()[1])

        loss = torch.sum(torch.mul(cce, weight))/batch_size/num_classes
    return loss

def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:

    return torch.mean(torch.abs(predictions1 - predictions2))

class TransferNet(nn.Module):
    def __init__(self, num_class, source_num, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.source_cnt = source_num
        self.use_tgt_cnt = False


        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss

        feat_dim = self.base_network.output_num()
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(feat_dim, bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        cent_list = [
                nn.Linear(bottleneck_width, 64),
                nn.ReLU(),
            ]
        self.center_layer = nn.Sequential(*cent_list)
        self.classifier_layer = nn.Linear(bottleneck_width, num_class)
        self.classifier_layer2 = nn.Linear(bottleneck_width, num_class)
        self.dropout = nn.Dropout(0.2)
        
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class,
            "source_cnt": self.source_cnt
        }

        mmd_loss_args = {
            "loss_type": "mmd",
            "max_iter": max_iter,
            "num_class": num_class,
            "source_cnt": self.source_cnt
        }

        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.mmd_loss = TransferLoss(**mmd_loss_args)
        self.criterion = LADELoss(num_classes=num_class, num_per_class = source_num, remine_lambda=0.1)
        self.criterion2 = PriorCELoss(num_classes=num_class, num_per_class = source_num)
        self.consistency_crit = symmetric_mse_loss
        self.JSD = JSD()
 
        

    def forward(self, source, target, source_label, target_clf_student, target_clf_teach, lamb, startup, cnt):

        source_onehot = torch.nn.functional.one_hot(source_label, num_classes = self.num_class).float()

        source = self.base_network(source)
        target = self.base_network(target)
        

        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)

        source_clf = self.classifier_layer(source)



        self.target_cnt = torch.tensor(cnt).float().cuda()
        div = self.JSD(target_clf_student, target_clf_teach)
        mean = torch.mean(div)
        std = torch.std(div)
        for i in range(div.shape[0]):
            if(div[i] <mean- std):
                div[i] = 1/(1+(mean-std - div[i])/std*4)
            elif(div[i] > mean+std):
                div[i] = 1 + (div[i]-std-mean)/std*4
            else:
                 div[i] = 1
        div = torch.reshape(div,(div.shape[0],1))
        div = torch.tile(div, (1,target_clf_student.shape[1]))
        div = torch.clamp(div, 0.1, 10)
        if(lamb > startup):

            target_clf_student = torch.div(target_clf_student, div)
            target_clf_teach = torch.div(target_clf_teach,div)
        
        
        if(lamb > startup):
            self.use_tgt_cnt = True
            self.adapt_loss.update(cnt)

        target_log_student = torch.nn.functional.softmax(target_clf_student, dim=1)
        target_log_teach = torch.nn.functional.softmax(target_clf_teach, dim=1)


        source_log = torch.nn.functional.softmax(source_clf, dim=1)
        _, preds = torch.max(source_clf, 1)
                

        clf_loss = self.criterion(source_clf, source_label)*0.1 +self.criterion2(source_clf, source_label) 
 

            
        # transfer
        kwargs = {}
        if self.transfer_loss == "lmmd" :
            kwargs['source_label'] = source_label
            kwargs['target_logits'] = target_log_student

        elif self.transfer_loss == "tblmmd":
            kwargs['source_label'] = source_label
            kwargs['target_logits'] = target_log_student
            kwargs['kl_div'] = div
        elif self.transfer_loss == "daan":
            source_clf = self.classifier_layer(source)
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        elif self.transfer_loss == 'bnm':
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)


        
        transfer_loss, to = self.adapt_loss(source, target, **kwargs)

        return clf_loss, transfer_loss, source

    def update_cnt(self, cnt):
        self.target_cnt = torch.tensor(cnt).float().cuda()
    def step(self):
        self.adapt_loss.loss_func.step()

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.classifier_layer2.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
                )
            params.append(
                {'params': self.center_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def setprior(self, prior):
        self.prior = prior
        
    def teacher(self, source, target, source_label, target_clf_teach, lamb):

        target = self.base_network(target)
        source = self.base_network(source)
        target = self.bottleneck_layer(target)
        source = self.bottleneck_layer(source)
        target_clf = self.classifier_layer(target)
        source_clf = self.classifier_layer(source)
        source_log = torch.nn.functional.softmax(source_clf, dim=1)
        target_log = torch.nn.functional.softmax(target_clf, dim=1)
        teach_log = torch.nn.functional.softmax(target_clf_teach, dim=1)
        tradeoff = min(lamb/20.0, 1)
        dist_loss = teach_loss(source_log, target_log, source_label, target_log)*tradeoff
        return dist_loss, target
    
    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        if(self.use_tgt_cnt):
            self.prior = self.target_cnt / self.target_cnt.sum()
            clf = clf + torch.log(self.prior + 1e-9)
        return clf

    
        

    def features(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        return x



    
    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass


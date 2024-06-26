import configargparse
import data_loader
#import data_loader_class as data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import torch.nn.functional as F
import textwrap
import copy
#from RandAugment import RandAugment

from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms

from math import log2
from scipy.special import rel_entr

def norm(p):
    total = sum(p)
    temp = [0]*len(p)
    for i in range(len(p)):
        temp[i] = p[i]/total
    return temp

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--randaug', type=str2bool, default=True)
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=1)
    parser.add_argument('--ema_alpha', type=float, default=0.9)
    parser.add_argument('--effective_number_beta', type=float, default=0.999)
    parser.add_argument('--classes_per_iter', type=int, default=10)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)

    source_loader, n_class, source_num = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers, randaug = args.randaug)
    target_train_loader, _, target_dataset = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers, randaug = args.randaug)

    target_test_loader, _, target_num = data_loader.load_data(
        folder_tgt, args.batch_size*2, infinite_data_loader=False, train=False, num_workers=args.num_workers)

    return source_loader, target_train_loader,target_test_loader, n_class, target_dataset, source_num



def get_model(args):
    model = models.TransferNet(
        args.n_class, args.class_cnt, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck, bottleneck_width=256,).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    print(initial_lr)
    params = model.get_parameters(initial_lr=initial_lr)

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    class_cnt = [0]*args.n_class
    true_class_cnt = [0]*args.n_class
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            for i in range(pred.shape[0]):
                class_cnt[pred[i]]+=1
                true_class_cnt[target[i]]+=1
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg, class_cnt, true_class_cnt


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    state_dict = model.state_dict()
    ema_state_dict = ema_model.state_dict()
    for (ema_name, ema_param), (name, param) in zip(ema_state_dict.items(), state_dict.items()):
        ema_param.copy_(ema_param*(alpha) + (1 - alpha)*param)


        
def train(source_loader, target_train_loader, target_test_loader2, target_dataset,
          model, model_ema, optimizer, lr_scheduler, args, outfile):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch


    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []

    startup = args.warmup_epoch

    cnt = [0]*args.n_class
    ignore_list = []
    global_step = 0
    JSD = models.JSD()
    alpha = 0.9


    batchnum = 0
    criterion = torch.nn.CrossEntropyLoss()
    for e in range(1, args.n_epoch+1):

        model.train()

        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        criterion = torch.nn.CrossEntropyLoss()
        batchcnt= 0
        source_labels = []
        target_preds = []
        for _ in range(n_batch):
            #file = open("jsd.csv","a")
            print("batch %d/%d"%(batchcnt, args.n_iter_per_epoch),end = '\r')
            batchcnt+=1
            batchnum +=1

            classlist = []
            allclasslist = list(range(args.n_class))
            allclasslist = [x for x in allclasslist if x not in ignore_list]
            classcnt = min(args.classes_per_iter, len(allclasslist))
            classlist = random.sample(allclasslist, classcnt)
            if(e > startup):

                # sample source data only from subset of classes
                srccnt = 0
                data_source_list = []
                label_source_list = []
                while(srccnt < args.batch_size):
                    data_source, label_source = next(iter_source)
                    index = (label_source == classlist[0])
                    for i in range(1,classcnt):
                        index = index | (label_source == classlist[i])
                    srccnt += np.count_nonzero(index)
                    data_source_list.append(data_source[index,:])
                    label_source_list.append(label_source[index])
                data_source = torch.cat(data_source_list,dim = 0)[:args.batch_size,:]
                label_source = torch.cat(label_source_list, dim = 0)[:args.batch_size]


                # sample target data only from subset of classes
                tgtcnt = 0
                data_target_list = []
                label_target_list = []
                searchcnt = 0
                while(tgtcnt < args.batch_size and searchcnt <20):
                    searchcnt +=1
                    data_target, tgt_max = next(iter_target)

                    with torch.no_grad():
                        target_clf = model.predict(data_target.to(args.device))
                        
                    tgt_max = target_clf.cpu().data.max(1)[1]

                    index = (tgt_max == classlist[0])
                    for i in range(1,classcnt):
                        index = index | (tgt_max == classlist[i])
                    tgtcnt += np.count_nonzero(index)
                    data_target_list.append(data_target[index,:])
                    label_target_list.append(tgt_max[index])
                if(searchcnt == 20):
                    data_target, label_target = next(iter_target)
                else:
                    data_target = torch.cat(data_target_list,dim = 0)[:args.batch_size,:]
                    label_target = torch.cat(label_target_list, dim = 0)[:args.batch_size]

            else:            
                data_target, label_target = next(iter_target) # .next()
                data_source, label_source = next(iter_source)


            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target, label_target = data_target.to(args.device), label_target.to(args.device)


            

            optimizer.zero_grad()



            with torch.no_grad():
                target_clf_student= model.predict(data_target)
                target_clf_teach= model_ema.predict(data_target)



            target_preds.append(torch.nn.functional.softmax(target_clf_student, dim=1).cpu().data)
            
                  
            optimizer.zero_grad()


            clf_loss, transfer_loss, source_feature,= model(data_source,
                                        data_target, label_source, target_clf_student, target_clf_teach, e, startup, cnt)

            model_ema.update_cnt(cnt)

            source_labels.append(label_source.cpu())


            loss = clf_loss + args.transfer_loss_weight * transfer_loss 
        
            

            loss.backward()

            global_step +=1
            update_ema_variables(model, model_ema, alpha, global_step)


            jsd1 = JSD(target_clf_student, target_clf_teach).cpu()


            

            optimizer.step()

   
            if lr_scheduler:
                lr_scheduler.step()
                

            
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())




        torch.save(model.state_dict(), "output/epoch_%d.pth"%(e))
        torch.save(model_ema.state_dict(), "output/epoch_%d_ema.pth"%(e))
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        
        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)


        # Test
        stop += 1

        teach_acc, teach_loss, cnt, truecnt = test(model_ema, target_test_loader2, args)
        student_acc, student_loss, cnt2, truecnt = test(model, target_test_loader2, args)
        kl1 = sum(rel_entr(norm(cnt), norm(truecnt)))
        kl2 = sum(rel_entr(norm(cnt2), norm(truecnt)))
        info += ', teach_loss {:4f}, teach_acc: {:.4f}, kl_div: {:.4f}'.format(teach_loss, teach_acc, kl1)
        info += ', student_loss {:4f}, student_acc: {:.4f}, kl_div: {:.4f}'.format(student_loss, student_acc, kl2)
        info += '\nteacher:'+str(cnt)
        info += '\nstudent:'+str(cnt2)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_acc < student_acc:
            best_acc = student_acc
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break

        outfile.write(info+"\n")
        outfile.flush()
        print(info)


    print('Transfer result: {:.4f}'.format(best_acc))


def main():
    parser = get_parser()
    args = parser.parse_args()
    file = open("logs/main_%s_to_%s_log.txt"%(args.src_domain, args.tgt_domain),'a')
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    file.write(textwrap.fill(str(args))+"\n")
    file.flush()
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader2, n_class, target_dataset, source_num = load_data(args)
    setattr(args, "n_class", n_class)
    setattr(args, "class_cnt", source_num)
    
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)


    model_ema = copy.deepcopy(model)
    optimizer = get_optimizer(model, args)
    print(args.lr)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader2, target_dataset, model, model_ema, optimizer, scheduler, args, file)
    file.close()

if __name__ == "__main__":
    main()

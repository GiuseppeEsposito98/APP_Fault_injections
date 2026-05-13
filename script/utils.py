
import argparse 
import torch

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--seed', default=42, type=int, help='seed in random number generator')
    parser.add_argument('-log_config', action='store_true', help='log config')
    parser.add_argument('--fsim_config', help='Yaml file path fsim config')
    return parser

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(net, val_loader, device, title, fsim_enabled, Fsim_setup, header):
    print(title)
    net = net.to(device)
    
    accs = list()
    
    batch = 0
    for image, target in val_loader:
        
        image = image.to(device, non_blocking=True)

        target = target.to(device, non_blocking=True)
        output = net(image)

        if fsim_enabled==True:
            Fsim_setup.FI_report.update_classification_report(batch,output,target,topk=(1,10))
        
        batch+=1
        
        acc = accuracy(output, target, topk=(1,))[0]
        accs.append(acc)
    
    print(f'{title} Accuracy: {torch.mean(torch.tensor(accs)):.2f}%')


    # if fsim_enabled==T:
    #     val_targ = val_targ.type(torch.int64)
    #     f1_1 = MulticlassF1Score(task='multiclass', num_classes=10, average='macro')
    #     rec_1 = MulticlassRecall(average='macro', num_classes=10)
    #     prec_1 = MulticlassPrecision(average='macro', num_classes=10)

    #     best_f1 = f1_1(val_distr, val_targ)
    #     best_rec = rec_1(val_distr, val_targ)
    #     best_prec = prec_1(val_distr, val_targ)

    #     f1_k = MulticlassF1Score(task='multiclass', num_classes=10, average='macro', top_k=5)
    #     rec_k = MulticlassRecall(num_classes=10, average='macro', top_k=5)
    #     prec_k = MulticlassPrecision(num_classes=10, average='macro', top_k=5)
    #     k_f1 = f1_k(val_distr, val_targ)
    #     k_rec = rec_k(val_distr, val_targ)
    #     k_prec = prec_k(val_distr, val_targ)
    #     Fsim_setup.FI_report.set_f1_values(best_f1=best_f1, k_f1=k_f1, header=header, best_prec= best_prec, best_rec = best_rec, k_prec= k_prec, k_rec = k_rec)
    #     counter = 0
    #     if handles is not None:
    #         for handle in handles: 
    #             counter += handle.to_zeroes_counter
    #             # logger.info(f'handle.to_zeroes_counter: {handle.to_zeroes_counter}')
    #             handle.to_zeroes_counter = 0
    #         Fsim_setup.FI_report.set_zeroes_counter(counter.item()/20, header=header)
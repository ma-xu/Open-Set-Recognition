import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation

def test(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # print(f"labels is: {labels}")
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())


    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    print(f"x1: {x1}")
    print(f"x2: {x2}")
    print(f"len x1: {len(x1)}")
    print(f"len x2: {len(x2)}")
    print(f"predict_k: {_pred_k}")
    print(f"max predict_k: {np.max(_pred_k)}")
    print(f"min predict_k: {np.min(_pred_k)}")
    print(f"max _pred_u: {np.max(_pred_u)}")
    print(f"min _pred_u: {np.min(_pred_u)}")
    results = evaluation.metric_ood(x1, x2)['Bas']


    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results

import torch
#cifar
# Five paths:
# talon: /work/xm0036/CVPR/Folder1/Open-Set-Recognition/OSR/CenterLoss/checkpoints/cifar_no1/
# talon /work/xm0036/CVPR/Folder2/Open-Set-Recognition/OSR/CenterLoss/checkpoints/cifar_no2/
# talon /work/xm0036/CVPR/Folder3/Open-Set-Recognition/OSR/CenterLoss/checkpoints/cifar_no3/
# talon /work/xm0036/CVPR/Folder4/Open-Set-Recognition/OSR/CenterLoss/checkpoints/cifar_no4/
# yang /home/UNT/jg0737/Open-Set-Recognition/OSR/CenterLoss/checkpoints/cifar_no5/
accuracy = torch.Tensor([0.654,0.652,0.639,0.645,0.654,])
F1 = torch.Tensor([0.654,0.652,0.639,0.645,0.654,])
f1_macro = torch.Tensor([0.615,0.614,0.601,0.612,0.625,])
f1_macro_weighted = torch.Tensor([0.654,0.652,0.639,0.646,0.655,])
auroc = torch.Tensor([0.975,0.975,0.975,0.975,0.976])

print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))

"""
##################  MNIST   ######################################
# Five paths:
# yang: /home/UNT/jg0737/Open-Set-Recognition/OSR/CenterLoss/checkpoints_number1
# talon /home/xm0036/Open-Set-Recognition/OSR/CenterLoss/checkpoints_number2
# talon /work/xm0036/Open-Set-Recognition/OSR/CenterLoss/checkpoints_number3
#
#

accuracy = torch.Tensor([0.839,0.855,0.865,0.883,0.860])
F1 = torch.Tensor([0.839,0.855,0.865,0.883,0.860])
f1_macro = torch.Tensor([0.876,0.889,0.894,0.907,0.889])
f1_macro_weighted = torch.Tensor([0.830,0.850,0.860,0.878,0.852])
auroc = torch.Tensor([0.999,0.999,0.999,0.999,0.998])

print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
"""

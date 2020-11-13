import torch
# Five paths:
# yang: /home/UNT/jg0737/Open-Set-Recognition/OSR/OLTR/checkpoints_number1
# talon /home/xm0036/Open-Set-Recognition/OSR/OLTR/checkpoints_number2
# talon /work/xm0036/Open-Set-Recognition/OSR/OLTR/checkpoints_number3
# yang: /home/UNT/jg0737/Open-Set-Recognition/OSR/OLTR/checkpoints_number4
# talon /home/xm0036/Open-Set-Recognition/OSR/OLTR/checkpoints_number5
#


accuracy = torch.Tensor([0.790, 0.796, 0.778,0.788,0.768])
F1 = torch.Tensor([0.790, 0.796, 0.778,0.788,0.768])
f1_macro = torch.Tensor([0.831,0.838,0.820,0.830,0.814])
f1_macro_weighted = torch.Tensor([0.760,0.771, 0.742,0.758,0.729])
auroc = torch.Tensor([0.999,0.999, 0.998,0.998,0.999])

print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))

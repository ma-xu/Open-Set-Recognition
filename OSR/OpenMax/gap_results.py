import torch
# Five paths:
# yang: /home/UNT/jg0737/Open-Set-Recognition/OSR/OpenMax/checkpoints_number1
# talon /home/xm0036/Open-Set-Recognition/OSR/OpenMax/checkpoints_number2
# talon /work/xm0036/Open-Set-Recognition/OSR/OpenMax/checkpoints_number3
# yang: /home/UNT/jg0737/Open-Set-Recognition/OSR/OpenMax/checkpoints_number4
# talon /home/xm0036/Open-Set-Recognition/OSR/OpenMax/checkpoints_number5
#

# softmax
accuracy = torch.Tensor([0.694, 0.694,0.693,0.693,0.694])
F1 = torch.Tensor([0.694,0.694,0.693,0.693,0.694])
f1_macro = torch.Tensor([0.730,0.731,0.733,0.728,0.731])
f1_macro_weighted = torch.Tensor([0.584,0.581,0.586,0.582,0.583])
auroc = torch.Tensor([0.999,0.998,0.998,0.998,0.998])
print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
print('_______________________________________\n\n')


# softmaxthreshold
accuracy = torch.Tensor([0.813,0.833,0.814,0.817,0.812])
F1 = torch.Tensor([0.813,0.833,0.814,0.817,0.812])
f1_macro = torch.Tensor([0.850,0.867,0.852,0.854,0.849])
f1_macro_weighted = torch.Tensor([0.793,0.819,0.796,0.800,0.793])
auroc = torch.Tensor([0.999,0.998,0.998,0.998,0.998])
print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
print('_______________________________________\n\n')


# openmax
accuracy = torch.Tensor([0.833,0.851,0.840,0.828,0.837])
F1 = torch.Tensor([0.833,0.851,0.840,0.828,0.837])
f1_macro = torch.Tensor([0.867,0.881,0.873,0.863,0.869])
f1_macro_weighted = torch.Tensor([0.821,0.841,0.830,0.815,0.824])
auroc = torch.Tensor([0.985,0.985,0.986,0.979,0.983])
print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
print('_______________________________________\n\n')

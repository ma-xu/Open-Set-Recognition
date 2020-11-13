import torch
#####################
# CIFAR-100
#####################
# Five paths:
# yang: /home/UNT/jg0737/Open-Set-Recognition/OSR/OpenMax/checkpoints/cifar_no1
# talon /work/xm0036/CVPR/Folder1/Open-Set-Recognition/OSR/OpenMax/checkpoints/cifar_no2/
# talon /work/xm0036/CVPR/Folder2/Open-Set-Recognition/OSR/OpenMax/checkpoints/cifar_no3/
# yang: /work/xm0036/CVPR/Folder3/Open-Set-Recognition/OSR/OpenMax/checkpoints/cifar_no4/
# talon /work/xm0036/CVPR/Folder4/Open-Set-Recognition/OSR/OpenMax/cifar_no4/
#

# softmax
accuracy = torch.Tensor([0.373,0.360,0.370,0.371,0.371])
F1 = torch.Tensor([0.373,0.360,0.370,0.371,0.371])
f1_macro = torch.Tensor([0.504,0.485,0.500,0.501,0.501])
f1_macro_weighted = torch.Tensor([0.257,0.247,0.255,0.256,0.255])
auroc = torch.Tensor([0.977,0.974,0.975,0.977,0.977])
print('________________SoftMax_______________________\n\n')
print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
print('_______________________________________\n\n')


# softmaxthreshold
accuracy = torch.Tensor([0.636,0.628,0.638,0.640,0.639])
F1 = torch.Tensor([0.636,0.628,0.638,0.640,0.639])
f1_macro = torch.Tensor([0.601,0.581,0.596, 0.604,0.602])
f1_macro_weighted = torch.Tensor([0.636,0.627,0.637,0.640,0.639])
auroc = torch.Tensor([0.977,0.974,0.975,0.977,0.977])
print('________________SoftMax-threshold_______________________\n\n')
print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
print('_______________________________________\n\n')


# openmax
accuracy = torch.Tensor([0.628,0.618,0.620,0.621,0.622])
F1 = torch.Tensor([0.628,0.618,0.620,0.621,0.622])
f1_macro = torch.Tensor([0.585,0.565,0.573,0.579,0.579])
f1_macro_weighted = torch.Tensor([0.629,0.616,0.619,0.621,0.622])
auroc = torch.Tensor([0.917,0.922,0.914,0.917,0.914])
print('________________OpenMax_______________________\n\n')
print('accuracy: %.3f , %.2f' % (accuracy.mean(), accuracy.std()))
print('F1: %.3f , %.2f' % (F1.mean(), F1.std()))
print('f1_macro: %.3f , %.2f' % (f1_macro.mean(), f1_macro.std()))
print('f1_macro_weighted: %.3f , %.2f' % (f1_macro_weighted.mean(), f1_macro_weighted.std()))
print('auroc: %.3f , %.2f' % (auroc.mean(), auroc.std()))
print('_______________________________________\n\n')






"""
###############################
# MNIST
###############################
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
"""

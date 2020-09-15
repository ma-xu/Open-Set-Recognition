# OpenMax

This folder reproduced the results of OpenMax: [Towards Open Set Deep Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)

## CIFAR-100
### CIFAR-100 Training  
``` shell
python3 cifar100.py
```
Some other parameters:

`--lr` : the initialized learning rate, default 0.1. <br>
`--resume` : PATH. Load pretrained model to continue training. <br>
`--arch` : select your backbone CNN models, default Resnet18. <br>
`--bs` : batch size: default 256. <br>
`--es` : epochs for training, default 100 <br>
`--train_class_num`: the number of known classes for training, default 50.<br>
`--test_class_num`: the number of total classes for testing, default 100 (that is all rest are unknown).<br>
`--includes_all_train_class`: whether includes all unknown classes during testing, default True (e.g., the number of unkown classes in testing should be test_class_num - train_class_num).<br>
`--evaluate`: evaluate the model without training. So you should use `--resume` to load pretrained model.<br>
`--weibull_tail`: parameters for weibull distribution, default 20.<br>
`--weibull_alpha`: parameters for weibull distribution, default 3.<br>
`--weibull_threshold`: parameters for confidence threshold, default 0.9. (0.9 may be the best for CIFAR datasets)<br>

### CIFAR-100 Testing
``` shell
python3 cifar100.py --resume $PATH-TO-ChECKPOINTS$ --evaluate
# e.g.,
# python3 cifar100.py --weibull_threshold 0.9 --evaluate --resume /home/xuma/Open-Set-Reconigtion/OSR/OpenMax/checkpoints/cifar/ResNet18/last_model.pth
```

### CIFAR-100 Tips
- In our implementation, we save the lastest models for each epoch.
- We test the model of last epoch after training.
- Checkpoint and log file are saved to `./checkpoints/cifar/$args.arch$/` folder.
- Checkpoint is named as last_model.pth

### CIFAR-100 preliminary results
Under the default settings (e.g, ResNet18, train_class_num=50, test_class_num=100, *which means openness=0.5*), we got the preliminary results (ACCURACY) as follow:

|          Method         | thr=0.1 | thr=0.2 | thr=0.3 | thr=0.5 | thr=0.7 | thr=0.9 |
|:-----------------------:|---------|---------|---------|---------|---------|---------|
|         SoftMax         | 0.379   | 0.379   | 0.379   | 0.379   | 0.379   | 0.379   |
| SoftMax(with threshold) | 0.379   | 0.382   | 0.405   | 0.497   | 0.579   | 0.648   |
|         OpenMax         | 0.524   | 0.521   | 0.526   | 0.559   | 0.597   | 0.636   |

(openmax may vary results slightly. Better weibull parameters may give better performance for openmax.)




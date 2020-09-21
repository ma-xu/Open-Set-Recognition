# Center Loss

This folder reproduced the results of Center-Loss: [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)

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
`--centerloss_weight`: the weight for center loss, default 0.03. <br>
`--center_lr`: learning rate for center loss parameters, default 0.1 (0.5 in original paper).<br>
`--threshold`: parameters for confidence threshold, default 0.9. (0.9 may be the best for CIFAR datasets)<br>

### CIFAR-100 Testing
``` shell
python3 cifar100.py --resume $PATH-TO-ChECKPOINTS$ --evaluate --threshold 0.9
# e.g.,
# python3 cifar100.py --evaluate --resume /home/g1007540910/Open-Set-Reconigtion/OSR/CenterLoss/checkpoints/cifar/ResNet18/last_model.pth --threshold 0.1
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
|         Center-Loss     | 0.379   | 0.392   | 0.428   | 0.526   | 0.603   | 0.661   |




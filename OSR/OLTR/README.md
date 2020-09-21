# OLTR (Large-Scale Long-Tailed Recognition in an Open World)

This folder reproduced the results of OLTR: [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/pdf/1904.05160.pdf)

## Remark
- We reconstruct the [OLTR offical repo](https://github.com/zhmiao/OpenLongTailRecognition-OLTR) for better understanding and easy deployment.
- The optimizer is combined as one for simplicty, which does not affect the results a lot. One can easily extend it to two seperate optimizers.
- The centroid is registered as a buffer in the model (passed from the loss), for a unified structure.
- Although add the ClassAwareSampler/ImbalancedDatasampler, we didn't test it since this project focuses on OSR, not long-tailed distribution.
- ... more ...

## CIFAR-100
### CIFAR-100 Training  
``` shell
python3 cifar100.py
```
Some other parameters:

`--train_class_num`: the number of known classes for training, default 50.<br>
`--test_class_num`: the number of total classes for testing, default 100 (that is all rest are unknown).<br>
`--includes_all_train_class`: whether includes all unknown classes during testing, default True (e.g., the number of unkown classes in testing should be test_class_num - train_class_num).<br>
`--evaluate`: evaluate the model without training. So you should use `--resume` to load pretrained model.<br>
`--lr` : the initialized learning rate, default 0.1. <br>
`--bs` : batch size: default 256. <br>
`--arch` : select your backbone CNN models, default Resnet18. <br>

`--stage1_resume` : PATH. Load pretrained stage-1 model to continue training. <br>
`--stage1_es` : epochs for training stage-1 model, default 30 <br>
`--stage1_use_fc` : Add fc layer for backbone, default False. <br>
`stage1_feature_dim`: the embedding dimension of extracted features, default 512 <br>
`stage1_classifier`: the classifer for stage-1 model, default dotproduct (FC layer), choose from 'dotproduct', 'cosnorm', 'metaembedding'<br>

`stage2_resume`: PATH. Load pretrained stage_2 model to continue training. <br>
`stage2_es`:  epochs for training stage-1 model, default 70. 60 in the offical repo, we modified to 70 for fairness. <br>
`stage2_use_fc`: Add fc layer for backbone, default True. <br>
`stage2_fea_loss_weight`: The weight for the feature loss (DiscCentroidsLoss). default 0.01. <br>
`oltr_threshold`: The classification threshold for OLTR, deault 0.1 <br>




### CIFAR-100 Testing
``` shell
python3 cifar100.py --stage2_resume $PATH-TO-ChECKPOINTS$ --evaluate
# e.g.,
# python3 cifar100.py --evaluate --stage2_resume /home/g1007540910/Open-Set-Reconigtion/OSR/OLTR/checkpoints/cifar/ResNet18/stage_2_last_model.pth --oltr_threshold 0.1
```

### CIFAR-100 Tips
- In our implementation, we save the lastest models for each epoch.
- We test the model of last epoch after training.
- Checkpoint and log file are saved to `./checkpoints/cifar/$args.arch$/` folder.
- Checkpoint is named as stage_1_last_model/stage_2_last_model.pth

### CIFAR-100 preliminary results (Updated)
Under the default settings (e.g, ResNet18, train_class_num=50, test_class_num=100, *which means openness=0.5*), we got the preliminary results (ACCURACY) as follow:


|          Method         | thr=0.1 | thr=0.2 | thr=0.3 | thr=0.5 | thr=0.7 | thr=0.9 |
|:-----------------------:|---------|---------|---------|---------|---------|---------|
|         SoftMax         | 0.367   | 0.368   | 0.384   | 0.464   | 0.551   |  0.627  |




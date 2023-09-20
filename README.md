# MAM
This is the code of MAM (Motor Assessment Model) based on Pytorch

## Prerequisite

```
matplotlib==3.6.0
numpy==1.23.1
pandas==2.0.0
PyYAML==6.0
scikit-learn==1.1.2
torch==1.12.1
tqdm==4.64.1
transformers==4.28.1
```

Install required packages:

`pip install -r requirements.txt`

## Pretrain the main part of the model
`python pretrain_clip.py`

## Train
`python main.py`

## Inference
`python inference.py`
## Parameters

| Parameter Name         | Type  | Default Value | Description                                          |
| ---------------------- | ----- | ------------- | ---------------------------------------------------- |
| -epoch                 | int   | 300           | Number of epochs                                     |
| -batch_size            | int   | 32            | Mini-batch size in training                          |
| -cv_id                 | int   | 0             | The cross-validation ID, choose from [0,1,2,3,4]     |
| -joint_in_channels     | int   | 3             | Input channel numbers of joint feature               |
| -joint_hidden_channels | int   | 64            | Hidden channel numbers of joint feature              |
| -time_window           | int   | 10            | The size of the sliding window in the time dimension |
| -time_step             | int   | 1             | The step of the sliding window in the time dimension |
| -optimizer             | str   | "SGD"         | Optimizer used in training                           |
| -lr                    | float | 0.001         | The starting learning rate                           |
| -scheduler             | str   | "CyclicLR"    | Scheduler used in training                           |
| -seed                  | int   | 0             | The random seed                                      |
| -tl_margin             | float | 0.4           | The triplet loss margin                              |
| -with_info             | bool  | True          | If use the info characteristics                      |
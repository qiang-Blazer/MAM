# MAM
This is the code of MAM (Motor Assessment Model) based on Pytorch. We provide a small dataset in `./prepared_data` to demo the codes.

## Installation
`git clone https://github.com/qiang-Blazer/MAM.git`
This will be quickly within 10 seconds.

## Prerequisites
We run the codes on the Unbuntu 20.04 operating system.
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

## Pretrain the central part of the model
`python pretrain_clip.py`
The pre-trained weights will be in `./checkpoints`.
This will be down in 5 minutes for the demo dataset.
## Train
`python main.py`
The results will be in `runs`.

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

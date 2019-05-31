# Tensorflow-Mobilenet
Mobilenetv1 implemented by Tensorflow

## 1. BG  
As we all know, MobileNetv1 is a light framework neural network, it can be deployed in any mobile device. The full details in paper(https://arxiv.org/abs/1704.04861)  
The final goal is to take MobileNet as backbone in YOLOv3. But it is diffucult to train from scratch, so a mobilenet pre_train weight is needed.  

## 2. Quick Start  
First, you will need a CiFar10 dataset:  
1. Clone this repo  
```
$ git clone 
```
2. You will need a cifar10 dataset before train your model
```
$ wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
3. Into your repo and mkdir
```
$ cd Tensorflow-Mobilenet
$ mkdir cifar
then unzip you cifar10 dataset into cifar
$ mkdir ckpt (for saved model)

```
After step mentioned above, your repo will looks like this:  
```
  Mobilnet:
          cifar (your data)
          ckpt (saved model ckpt)
          train.py
          freeze_graph.py
          mobilenet.py
          train.py
```
## 3. train your model  
You are allowed to use command line to start training:
```
      The agrs are description below:
        --lr            learing_rate from begin, and it will decay by 0.99
        --batch_size    a mini_batch size depend on your GPU memory, a appropriate 
```
Here are two ways to train model, the first is to load pre_train model that i train on my Mac.
```
$ python3 train.py --lr 1e-3 --batch_size 16 --epochs 20 --load_pretrain 1
```
The second ways is to train your model from scratch
```
$ python3 train.py --lr 1e-3 --batch_size 16 --epochs 20 --load_pretrain 0
```

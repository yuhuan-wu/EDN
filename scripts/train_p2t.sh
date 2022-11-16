#!/bin/bash

#### Default training script is for ResNet-50
PYTHONPATH=$(pwd):$PYTHONPATH python3 scripts/train.py --max_epochs 90 \
                                                    --num_workers 2 \
                                                    --batch_size 24 \
                                                    --savedir ./snaps/edn-p2t-ms \
                                                    --lr_mode poly \
                                                    --lr 5e-5 \
                                                    --width 384 \
                                                    --height 384 \
                                                    --iter_size 1 \
                                                    --arch p2t_small \
                                                    --ms 0 \
                                                    --ms1 1 \
                                                    --bcedice 1 \
                                                    --adam_beta2 0.99 \
                                                    --group_lr 0 \
                                                    --freeze_s1 0

##### The true batchsize is batch_size * iter_size, which is (24 * 1) here.
##### GPU with 12G memory:
##### For edn-vgg16,  you can use "--lr 5e-5 --batch_size 6 --iter_size 4 --arch vgg16"
##### For edn-resnet, you can use "--lr 5e-5 --batch_size 12 --iter_size 2 --arch resnet50"
##### For edn-lite, you can use "--lr 1.7e-4 --batch_size 24 --iter_size 1 --arch mobilenetv2"
##### Please ensure batchsize >= 6, since a very small batchsize may significantly decrease the performance.

##### GPU <8G memory
##### If your gpu memory is not enough, you can reduce --batch_size and increase --iter_size, keeping batch_size*iter_size =24 or near 24.

##### Multi-scale training strategy
##### We support two strategies: ms and ms1
##### --ms: train with one scale for a long time, and then train with next scale. (default)
##### --ms1: train with random scale for each iteration (may work better?)
##### I think --ms1 should work better than than --ms, but experiments show that they can achieve comparable performance. 
##### If you wanna use ms1 strategy, please set `--ms 0 --ms1 1 --max_epochs 90`.


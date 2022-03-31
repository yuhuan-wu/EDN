#!/bin/bash

PYTHONPATH=$(pwd):$PYTHONPATH python3 scripts/train.py --max_epochs 30 \
                                                    --num_workers 2 \
                                                    --batch_size 24 \
                                                    --savedir ./snaps/edn-lite \
                                                    --lr_mode poly \
                                                    --lr 5e-5 \
                                                    --width 384 \
                                                    --height 384 \
                                                    --iter_size 1 \
                                                    --arch mobilenetv2 \
                                                    --ms 1 \
                                                    --ms1 0 \
                                                    --bcedice 1 \
                                                    --adam_beta2 0.99 \
                                                    --group_lr 0 \
                                                    --freeze_s1 0

##### The true batchsize is batch_size * iter_size, which is (24 * 1) here.
##### GPU with 12G memory:
##### For edn-vgg16,  you can use "--batch_size 6 --iter_size 4 --arch vgg16"
##### For edn-resnet, you can use "--batch_size 12 --iter_size 2 --arch resnet50"
##### For edn-lite, you can use "--batch_size 24 --iter_size 1 --arch mobilenetv2"

##### GPU <8G memory
##### If your gpu memory is not enough, you can reduce --batch_size and increase --iter_size, keeping batch_size*iter_size =24 or near 24.

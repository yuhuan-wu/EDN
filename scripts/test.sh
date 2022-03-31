#!/bin/bash

NAMES=('EDN-Lite' 'EDN-VGG16' 'EDN-R50' 'EDN-LiteEX')

for NAME in "${NAMES[@]}"
do
  PYTHONPATH=$(pwd):$PYTHONPATH  python3 scripts/test.py --pretrained pretrained/$NAME.pth \
                                      --savedir ./salmaps/$NAME/ \

done

#!/bin/bash

NAMES=('EDN-Testset')

for NAME in "${NAMES[@]}"
do
  PYTHONPATH=$(pwd):$PYTHONPATH  python3 scripts/test.py --pretrained /home/yuanbinfu/EDN-master/snaps/edn-lite/model_Lite.pth \
                                      --savedir ./salmaps/$NAME/ \

done

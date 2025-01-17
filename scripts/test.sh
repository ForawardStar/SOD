#!/bin/bash

NAMES=('SOD')

for NAME in "${NAMES[@]}"
do
  PYTHONPATH=$(pwd):$PYTHONPATH  python3 scripts/test.py --pretrained checkpont.pth \
                                      --savedir ./salmaps/$NAME/ \

done

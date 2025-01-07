#!/bin/bash

PYTHONPATH=$(pwd):$PYTHONPATH python3 scripts/train.py --max_epochs 90 \
                                                    --num_workers 2 \
                                                    --batch_size 24 \
                                                    --savedir ./snaps/ours-ms \
                                                    --lr_mode poly \
                                                    --lr 5e-5 \
                                                    --width 384 \
                                                    --height 384 \
                                                    --iter_size 1 \
                                                    --ms 0 \
                                                    --ms1 1 \
                                                    --bcedice 1 \
                                                    --adam_beta2 0.99 \
                                                    --group_lr 0 \
                                                    --freeze_s1 0

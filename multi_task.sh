#!/usr/bin/env bash

if [ -z $1 ]; then
    echo '[ERROR] please set gpu_id'
    exit
fi
gpu_id=$1

script="train_mix_new.py"

for i in `seq 1 9`; do
    echo "[run ${i}] ${script} config${i}.ini -b 50 -g ${gpu_id} -p -m multi"
    python ${script} config${i}.ini -b 50 -g ${gpu_id} -p -m multi
    echo ''
done

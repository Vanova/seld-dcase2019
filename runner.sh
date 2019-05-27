#!/bin/bash

export PYTHONPATH="`pwd`:$PYTHONPATH"
source activate ai3

model=log_mel_256
log_dir=logs/$(date "+%d_%b_%Y")
log_file=$log_dir/${model}_$(date "+%H_%M_%S").log
mkdir -p $log_dir

echo "Log to: $log_file"
python -u seld.py 4 $model > ${log_file}

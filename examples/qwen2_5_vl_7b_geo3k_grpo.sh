#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/mnt2/devon/models/Qwen2.5-VL-7B  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt2/devon/EasyR1/soda_reasoning/data_for_sft/soda_train.json \
    data.val_files=/mnt2/devon/EasyR1/soda_reasoning/data_for_sft/soda_train.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_soda_grpo \
    trainer.n_gpus_per_node=4

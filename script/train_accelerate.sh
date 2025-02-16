#!/usr/bin/env bash
cd ..

NUM_GPUS=`echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l`
NUM_PROC=8

training_languages=("english" "german" "french" "italian" "portuguese" "spanish" "hindi")

seed=42
gpus=("0,1,2,3,4,5,6,7")

lr=0.000005
dropout=0.1



num_gpus=${#gpus[@]}

num_datasets=${#training_languages[@]}

for gpu_index in "${!gpus[@]}"; do       
    gpu=${gpus[$gpu_index]}
    dataset_index=$gpu_index
    concatenated_cmd=""

   while [ "$dataset_index" -lt "$num_datasets" ]; do
                        
        dataset=${training_languages[$dataset_index]}
        SESSION_NAME="$dataset"

        cmd="WANDB_PROJECT=tokyo-drift CUDA_VISIBLE_DEVICES=$gpu 
            NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 accelerate launch \
                --config_file ./accelerate_config/train.yaml \
                --num_processes 8 \
                --num_machines 1 \
                --num_cpu_threads_per_process 2 \
                --deepspeed_config_file ./deepspeed_config/stage1.json \
                ./train/sft_trainer.py \
                    --training_language $dataset \
                    --seed $seed \
                    --num_epochs 3 \
                    --save_steps 100 \
                    --logging_steps 100 \
                    --eval_steps 10 \
                    --save_total_limit -1 \
                    --batch_size 8 \
                    --gradient_accumulation_steps 1 \
                    --lr $lr \
                    --dropout $dropout ; "
            concatenated_cmd+="$cmd"
        ((dataset_index = dataset_index + num_gpus))
    done 
    echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
    echo $concatenated_cmd
    screen -L -dmS "$SESSION_NAME" bash -c "$concatenated_cmd"                  
done


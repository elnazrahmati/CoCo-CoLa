cd ..

training_languages=("german" "french" "italian" "portuguese" "spanish" "hindi")
seeds=(42)
checkpoints=("../experiments/llama-8b/dropout-0.1-lr-5e-06/english/checkpoint-200")
languages=("english")

# starting_layer=(0 0 10)
# ending_layer=(5 10 16)

starting_layer=(0 15)
ending_layer=(15 31)

unfreeze_module=("full")

NUM_GPUS=`echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l`
NUM_PROC=8


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
    
        num_prev_datasets=${#checkpoints[@]}
        dataset=${training_languages[$dataset_index]}
        SESSION_NAME="$dataset"

        for checkpoint_index in $(seq 0 $((num_prev_datasets-1))); do    
            checkpoint=${checkpoints[$checkpoint_index]}
            language=${languages[$checkpoint_index]}
            if [ "$dataset" == "$language" ]; then
                echo "Skipping dataset $dataset as it is the same as the previous dataset $language"
                continue
            fi

            for seed in "${seeds[@]}"; do
                for ((i=0;i<${#starting_layer[@]};++i)); do
                    for unfreeze in "${unfreeze_module[@]}"; do
                        cmd="WANDB_PROJECT=tokyo-drift CUDA_VISIBLE_DEVICES=$gpu 
                            NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 accelerate launch \
                                --config_file ./accelerate_config/train.yaml \
                                --num_processes 8 \
                                --gpu_ids $gpu \
                                --main_process_port 29502 \
                                --num_machines 1 \
                                --num_cpu_threads_per_process 2 \
                                --deepspeed_config_file ./deepspeed_config/stage1.json \
                                partial_sft_trainer.py \
                                    --training_language $dataset \
                                    --checkpoint_dir $checkpoint \
                                    --starting_language $language \
                                    --starting_layer ${starting_layer[$i]} \
                                    --ending_layer ${ending_layer[$i]} \
                                    --num_epochs 3 \
                                    --seed $seed \
                                    --batch_size 8 \
                                    --gradient_accumulation_steps 1 \
                                    --lr $lr \
                                    --unfreeze_module $unfreeze ; "
                        concatenated_cmd+="$cmd"

                        

                    done
                done
                
            done
            

        done
        ((dataset_index = dataset_index + num_gpus))
    done 
    echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
    echo $concatenated_cmd
    screen -L -dmS "$SESSION_NAME" bash -c "$concatenated_cmd" 
done


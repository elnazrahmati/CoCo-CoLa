cd ..
training_languages=("german" "french" "italian" "portuguese" "spanish" "hindi")

seeds=(42)
gpus=(2 3 4 5)

num_gpus=${#gpus[@]}

num_datasets=${#training_languages[@]}
for gpu_index in "${!gpus[@]}"; do       
    gpu=${gpus[$gpu_index]}
    dataset_index=$gpu_index
    concatenated_cmd=""

    while [ "$dataset_index" -lt "$num_datasets" ]; do
        dataset=${training_languages[$dataset_index]}
        SESSION_NAME="$dataset"

        cmd="CUDA_VISIBLE_DEVICES=$gpu python lang_interpretability_eval.py \
                --finetuned_language $dataset \
                --model_type llama-8b \
                --unfreezed_module full ; "
        concatenated_cmd+="$cmd"

        cmd="CUDA_VISIBLE_DEVICES=$gpu python lang_interpretability_eval.py \
                --finetuned_language $dataset \
                --model_type llama-3b \
                --unfreezed_module full ; "
        concatenated_cmd+="$cmd"

        cmd="CUDA_VISIBLE_DEVICES=$gpu python lang_interpretability_eval.py \
                --finetuned_language $dataset \
                --model_type llama-1b \
                --unfreezed_module full ; "
        concatenated_cmd+="$cmd"

        ((dataset_index = dataset_index + num_gpus))
    done 
    echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
    echo $concatenated_cmd
    screen -dmS "$SESSION_NAME" bash -c "$concatenated_cmd" 
done


cd .. 

training_languages=("english" "german" "french" "italian" "portuguese" "spanish" "hindi")
seeds=(42)
gpus=(2 3 4 5 6 7)
model_sizes=(1 3 8 4)

num_gpus=${#gpus[@]}

num_datasets=${#training_languages[@]}
for gpu_index in "${!gpus[@]}"; do       
    gpu=${gpus[$gpu_index]}
    dataset_index=$gpu_index
    concatenated_cmd=""

    while [ "$dataset_index" -lt "$num_datasets" ]; do
        dataset=${training_languages[$dataset_index]}
        SESSION_NAME="$dataset"
        for model_size_index in "${!model_sizes[@]}"; do
            model_size=${model_sizes[$model_size_index]}
            cmd="CUDA_VISIBLE_DEVICES=$gpu python test_output.py \
                    --language $dataset \
                    --finetuned 1 \
                    --model_language $dataset \
                    --model_size $model_size ; "
            concatenated_cmd+="$cmd"

            cmd="CUDA_VISIBLE_DEVICES=$gpu python test_output.py \
                    --language $dataset \
                    --finetuned 0 \
                    --model_size $model_size ; "
            concatenated_cmd+="$cmd"

            cmd="CUDA_VISIBLE_DEVICES=$gpu python test_output.py \
                    --language $dataset \
                    --finetuned 1 \
                    --model_language english \
                    --model_size $model_size ; "
            concatenated_cmd+="$cmd"
        done

        
        ((dataset_index = dataset_index + num_gpus))
    done 
    echo "Running model with gpu: $gpu, dataset: $dataset, session_name: $SESSION_NAME"
    echo $concatenated_cmd
    screen -dmS "$SESSION_NAME" bash -c "$concatenated_cmd" 
done


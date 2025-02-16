# CoCo-CoLa: Evaluating Language Adherence in Multilingual LLMs  
Multilingual Large Language Models (LLMs) develop cross-lingual abilities despite being trained with limited parallel data. However, they often struggle to generate responses in the intended language, favoring high-resource languages such as English. In this work, we introduce **CoCo-CoLa** (Correct Concept - Correct Language), a novel metric to evaluate language adherence in multilingual LLMs. Using fine-tuning experiments on a closed-book QA task across seven languages, we analyze how training in one language affects others' performance. Our findings reveal that multilingual models share task knowledge across languages but exhibit biases in output language selection. We identify language-specific layers, showing that final layers play a crucial role in determining output language. Accordingly, we propose a partial training strategy that selectively fine-tunes key layers, improving language adherence while significantly reducing computational cost. Our method achieves comparable or superior performance to full fine-tuning, particularly for low-resource languages, offering a more efficient multilingual adaptation.

## Requirements  
Install ```Python 3.10.12``` and alll the packages available in ```requirements.txt```.  

## Data preparation  
Download json files of the [Mintaka datasset](https://huggingface.co/datasets/jinaai/mintakaqa) and put in ```./data``` directory. Then run the following commands to prepare data for each language:  
```
python ./data_prep/data_sep.py
python ./data_prep/data_translator.py
```

## Preliminary analysis  
Run the following commands for full finetuning of Llama models on all languages:  
```
cd ./script
./train_accelerate.sh
```
When the training is done, run the following command to save the outputs and calculate model updates:  
```
./test_output_run.sh
python ./evaluation/model_diff_visualization.py
```
Use accurecies and correct-overlap sections of ```notebooks/result_prep.ipynb``` to prepare the rest of the results.  

## CoCo-CoLa ratio  
Run ```./script/coca-cola-ratio-run.sh``` and use coca-cola section of ```notebooks/result_prep.ipynb``` to aggregate the results.  

## Partial Training  
Run the following commands to perform partial SFT for all models and languages, and evaluate the trained models.  
```
cd ./script
./partial_trainer_run.sh
./interpretability_eval_run.sh
```

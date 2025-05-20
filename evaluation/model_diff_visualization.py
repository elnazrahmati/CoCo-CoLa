from transformers import AutoModelForCausalLM, Gemma3ForConditionalGeneration
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"

# lang_checkpoints = {
#     'english': 300, 
#     'hindi': 300,
#     'portuguese': 400,
#     'spanish': 700,
#     'french': 500,
#     'german': 800,
#     'italian': 500,
# }

lang_checkpoints = {
    'english': 200, 
    'hindi': 200,
    'portuguese': 200,
    'spanish': 200,
    'french': 200,
    'german': 200,
    'italian': 200,
}

model_size = 8

for language in lang_checkpoints.keys():

    if model_size == 4:
        finetuned_model = Gemma3ForConditionalGeneration.from_pretrained(f"../experiments/gemma-{model_size}b/dropout-0.1-lr-5e-06/{language}/checkpoint-{lang_checkpoints[language]}").to(device)
    else:
        finetuned_model = AutoModelForCausalLM.from_pretrained(f"../experiments/llama-{model_size}b/dropout-0.1-lr-5e-06/{language}/checkpoint-{300}").to(device)
    finetuned_model.eval()
    if model_size == 8:
        pretrained_model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.1-{model_size}B").to(device)
    elif model_size == 4:
        pretrained_model = Gemma3ForConditionalGeneration.from_pretrained(f"google/gemma-3-{model_size}b-pt").to(device)
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}B").to(device)
    pretrained_model.eval()

    # Step 1: Collect average magnitude of parameter differences
    param_diffs = {} 
    for (name_fine, p_fine), (name_pre, p_pre) in zip(finetuned_model.named_parameters(), pretrained_model.named_parameters()):
        if name_fine == name_pre:
            diff = (p_fine - p_pre).abs().mean().item()
            layer_num = None
            if 'layers.' in name_fine:
                layer_str = name_fine.split('layers.')[1].split('.')[0]
                layer_num = int(layer_str)
            if layer_num not in param_diffs:
                if layer_num is not None:
                    param_diffs[layer_num] = {}
            name_fine = name_fine.replace('_proj', '').replace('.weight', '').replace('model.layers.', '').replace('language_model.', '')
            # remove any digits from the name
            name_fine = '.'.join([i for i in name_fine.split('.') if not i.isdigit()])
            if layer_num is not None:
                if 'norm' not in name_fine:
                    if 'vision' not in name_fine:
                        param_diffs[layer_num][name_fine] = diff

    # Step 2: Create a DataFrame
    heatmap_entries = []
    for layer, params in param_diffs.items():
        for param_name, diff_val in params.items():
            heatmap_entries.append({'layer': layer, 'param': param_name, 'diff': diff_val})
    df = pd.DataFrame(heatmap_entries)
    heatmap_data = df.pivot(index='layer', columns='param', values='diff').fillna(0)

    # Step 3: Visualize differences as a heatmap
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.3)
    sns.heatmap(heatmap_data, cmap="viridis")
    plt.title(f"Llama-{model_size}B - {language}", fontsize=26)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12, rotation=0)
    plt.xlabel("", fontsize=2)
    plt.ylabel("", fontsize=2)
    plt.savefig(f"../visualization/heatmaps/llama-{model_size}b-{language}.png", bbox_inches='tight', dpi=300)
    print(f"Saved visualization for {language}")


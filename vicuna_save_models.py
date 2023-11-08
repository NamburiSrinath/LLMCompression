from string import punctuation
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import bert_instantiate_model, compute_f1, dataloader_debug, encoder_decoder_evaluate, extract_dataset, global_pruning, global_pruning_quantize, instantiate_model, llama_inference, local_pruning, remove_duplicates, tokenizer_debug, instantiate_all_linear_layers
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
import torch
import sys
import time
import os
from huggingface_hub import upload_file, upload_folder
import shutil
import subprocess

os.environ['TRANSFORMERS_CACHE'] = '/data/srinath_models_data/huggingface/'
os.environ['HF_DATASETS_CACHE'] = '/data/srinath_models_data/huggingface/'
os.environ['HF_CACHE'] = '/data/srinath_models_data/huggingface/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/srinath_models_data/huggingface/'
os.environ['HF_HOME'] = '/data/srinath_models_data/huggingface/'
os.environ['XDG_CACHE_HOME']='data/srinath_models_data'
    
prune_type = str(sys.argv[1])
prune_percentage = str(sys.argv[2])
os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[3])

if __name__ == '__main__':
    start_time = time.time()
    no_of_layers = 32
    model_name = 'vicuna-7b'
    baseline_path = 'NamburiSrinath/vicuna-7b-baseline'
    tokenizer = LlamaTokenizer.from_pretrained(baseline_path, padding_side='left')
    tokenizer.pad_token = tokenizer.unk_token
    print(f"Fast tokenizer is available: {tokenizer.is_fast}")

    if float(prune_percentage) != 0:
        if prune_type == 'overall-global-pruning':
            if not os.path.exists(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}'):
                device = torch.device('cpu')
                model = LlamaForCausalLM.from_pretrained(baseline_path)
                print(f"Model loaded from {baseline_path}")
                linear_layers_list = instantiate_all_linear_layers(model)
                print(linear_layers_list)
                global_pruning_quantize(linear_layers_list, prune_percentage=float(prune_percentage))
                model.save_pretrained(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}')
                print(model)
            else:
                print(f"Model read from /data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}")
        if prune_type == 'attention-only-global-pruning':
            if not os.path.exists(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}'):
                device = torch.device('cpu')
                attention_layers_list = []
                for i in range(no_of_layers):
                    attention_layers_list.append(f'model.layers.{i}.self_attn.q_proj')
                    attention_layers_list.append(f'model.layers.{i}.self_attn.k_proj')
                    attention_layers_list.append(f'model.layers.{i}.self_attn.v_proj')
                    attention_layers_list.append(f'model.layers.{i}.self_attn.o_proj')
                model = LlamaForCausalLM.from_pretrained(baseline_path)
                print(f"Model loaded from {baseline_path}")
                linear_layers_list = instantiate_model(model, attention_layers_list)
                print(linear_layers_list)
                global_pruning_quantize(linear_layers_list, prune_percentage=float(prune_percentage))
                model.save_pretrained(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}')
                print(model)
            else:
                print(f"Model read from /data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}")
        if prune_type == 'output-only-global-pruning':
            if not os.path.exists(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}'):
                device = torch.device('cpu')
                output_layers_list = ['LlamaMLP']
                model = LlamaForCausalLM.from_pretrained(baseline_path)
                print(f"Model loaded from {baseline_path}")
                linear_layers_list = bert_instantiate_model(model, output_layers_list)
                print(linear_layers_list)
                global_pruning_quantize(linear_layers_list, prune_percentage=float(prune_percentage))
                model.save_pretrained(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}')
                print(model)
            else:
                print(f"Model read from /data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}")
        if prune_type.split('-')[0] == 'local_pruning':
            local_prune_type = prune_type.split('-')[-1]
            layer_index = 0
            device = torch.device('cpu')
            model = LlamaForCausalLM.from_pretrained(baseline_path)
            linear_layers_list = instantiate_model(model, ['lm_head'])
            print(linear_layers_list)
            local_pruning(model, linear_layers_list, layer_index, prune_percentage=float(prune_percentage), prune_type=local_prune_type,n=1)
            model.save_pretrained(f'/data/srinath_models_data/{model_name}-{prune_type}-{prune_percentage}')
        end_time = time.time()
        print(f"Execution time: {int(end_time) - int(start_time)}")
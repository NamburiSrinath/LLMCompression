import shutil
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import bert_instantiate_model, global_pruning, global_pruning_quantize, local_pruning, instantiate_model, remove_duplicates, extract_dataset, encoder_decoder_inference, instantiate_all_linear_layers
from transformers.utils import logging
import subprocess
from peft import LoraConfig, get_peft_model, TaskType

logging.set_verbosity(40)
torch.manual_seed(40)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def tokenize_function(example):
    input_sentence = example['masked_sentence']
    actual_label = example['obj_label']
    t5_input_sentence = [s.replace(" [MASK]", " <extra_id_0>") for s in input_sentence]
    t5_input_sentence = ["Fill the blank with appropriate word: " + s for s in t5_input_sentence]
    t5_actual_label = ["<extra_id_0> " + s + " <extra_id_1>"for s in actual_label]
    # t5_actual_label = actual_label

    # for sentence, label in zip(t5_input_sentence, t5_actual_label):
    #     print(sentence, label)
    #     print("\n")

    tokenized_text = tokenizer(t5_input_sentence, truncation=True,
                                padding='max_length', max_length=128)
    # print(tokenized_text)
    tokenized_labels = tokenizer(t5_actual_label, truncation=True, padding='max_length', 
                                    max_length=8)

    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "output_labels": tokenized_labels['input_ids']
    }
    return tokenized_data

if __name__ == '__main__':
    checkpoint = str(sys.argv[1])
    prune_type = str(sys.argv[2])

    if checkpoint == 'google/flan-t5-base' or checkpoint == 'MBZUAI/LaMini-Flan-T5-248M':
        no_of_layers = 12
    if checkpoint == 'google/flan-t5-large' or checkpoint == 'MBZUAI/LaMini-Flan-T5-783M':
        no_of_layers = 24
    if checkpoint == 'google/flan-t5-xl':
        no_of_layers = 24
    if checkpoint == 'google/flan-t5-xxl':
        no_of_layers = 24

    batch_size=128
    
    for dataset_name in dataset_name_list:
        model_name = checkpoint.split('/')[-1]
        prune_percentage_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if model_name == 'flan-t5-base':
            tokenizer_path = '/hdd4/srinath2/.cache/huggingface/hub/models--google--flan-t5-base/snapshots/c782cba52f8ea6a704240578055cf1c3fc2f2ca9'
        if model_name == 'flan-t5-large':
            tokenizer_path = '/hdd4/srinath2/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/2d6503cbe79448e511312ba3377a9cde16a2135a'
        if model_name == 'LaMini-Flan-T5-248M':
            tokenizer_path = '/hdd4/srinath2/.cache/huggingface/hub/models--MBZUAI--LaMini-Flan-T5-248M/snapshots/4e871ba5f20216feaa3b845fc782229cd64eba47'
        if model_name == 'LaMini-Flan-T5-783M':
            tokenizer_path = '/hdd4/srinath2/.cache/huggingface/hub/models--MBZUAI--LaMini-Flan-T5-783M/snapshots/7a1ff0207cbe75d6a1fcbcc7077ee0b6808ecf9f'
        if model_name == 'flan-t5-xl':
            tokenizer_path = '/hdd4/srinath2/.cache/huggingface/hub/models--google--flan-t5-xl/snapshots/8772db7a7a11f7b08e6be7d7088f7a7fd4813bc5'

        for prune_percentage in prune_percentage_list:
            if prune_type == 'baseline' and prune_percentage == 0:
                print("In Baseline-----")
                command = f"python /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/main.py --model hf-seq2seq --model_args pretrained={checkpoint} --tasks boolq,piqa,winogrande --device cuda:0 --batch_size=1 > /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/lm_evaluation_logs_rebuttal/flant5/{model_name}_{prune_type}_rebuttal.log"
                subprocess.run(command, shell=True)
    
            if prune_percentage != 0:
                if prune_type == 'overall_global_pruning':
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_all_linear_layers(model)
                    
                    # Global pruning
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    
                    model.save_pretrained(f'{model_name}-{prune_type}-{prune_percentage}')

                    shutil.copy(f'{tokenizer_path}/config.json', f'{model_name}-{prune_type}-{prune_percentage}/config.json')
                    shutil.copy(f'{tokenizer_path}/tokenizer.json', f'{model_name}-{prune_type}-{prune_percentage}/tokenizer.json')
                    shutil.copy(f'{tokenizer_path}/tokenizer_config.json', f'{model_name}-{prune_type}-{prune_percentage}/tokenizer_config.json')
                    shutil.copy(f'{tokenizer_path}/special_tokens_map.json', f'{model_name}-{prune_type}-{prune_percentage}/special_tokens_map.json')
                    shutil.copy(f'{tokenizer_path}/generation_config.json', f'{model_name}-{prune_type}-{prune_percentage}/generation_config.json')

                    command = f"python /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/main.py --model hf-seq2seq --model_args pretrained={model_name}-{prune_type}-{prune_percentage} --tasks boolq,piqa,winogrande --device cuda:0,1 --batch_size=8 > /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/lm_evaluation_logs_rebuttal/flant5/{model_name}_{prune_type}_{prune_percentage}_rebuttal.log"
                    subprocess.run(command, shell=True)
                    shutil.rmtree(f'{model_name}-{prune_type}-{prune_percentage}')
                    
                # To generate Fig 8 from paper, comment out the respective attention modules accordingly and run the command
                if prune_type == 'attention_only_global_pruning':
                    attention_layers_list = []
                    for i in range(no_of_layers):
                        attention_layers_list.append(f'encoder.block.{i}.layer.0.SelfAttention.q')
                        attention_layers_list.append(f'encoder.block.{i}.layer.0.SelfAttention.k')
                        attention_layers_list.append(f'encoder.block.{i}.layer.0.SelfAttention.v')
                        attention_layers_list.append(f'encoder.block.{i}.layer.0.SelfAttention.o')

                        attention_layers_list.append(f'decoder.block.{i}.layer.0.SelfAttention.q')
                        attention_layers_list.append(f'decoder.block.{i}.layer.0.SelfAttention.k')
                        attention_layers_list.append(f'decoder.block.{i}.layer.0.SelfAttention.v')
                        attention_layers_list.append(f'decoder.block.{i}.layer.0.SelfAttention.o')

                        attention_layers_list.append(f'decoder.block.{i}.layer.1.EncDecAttention.q')
                        attention_layers_list.append(f'decoder.block.{i}.layer.1.EncDecAttention.k')
                        attention_layers_list.append(f'decoder.block.{i}.layer.1.EncDecAttention.v')
                        attention_layers_list.append(f'decoder.block.{i}.layer.1.EncDecAttention.o')
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, attention_layers_list)
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    
                    model.save_pretrained(f'{model_name}-{prune_type}-{prune_percentage}')

                    shutil.copy(f'{tokenizer_path}/config.json', f'{model_name}-{prune_type}-{prune_percentage}/config.json')
                    shutil.copy(f'{tokenizer_path}/tokenizer.json', f'{model_name}-{prune_type}-{prune_percentage}/tokenizer.json')
                    shutil.copy(f'{tokenizer_path}/tokenizer_config.json', f'{model_name}-{prune_type}-{prune_percentage}/tokenizer_config.json')
                    shutil.copy(f'{tokenizer_path}/special_tokens_map.json', f'{model_name}-{prune_type}-{prune_percentage}/special_tokens_map.json')
                    shutil.copy(f'{tokenizer_path}/generation_config.json', f'{model_name}-{prune_type}-{prune_percentage}/generation_config.json')

                    command = f"python /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/main.py --model hf-seq2seq --model_args pretrained={model_name}-{prune_type}-{prune_percentage} --tasks boolq,piqa,winogrande --device cuda:0 --batch_size=1 > /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/lm_evaluation_logs_rebuttal/flant5/{model_name}_{prune_type}_{prune_percentage}_Rebuttal.log"
                    subprocess.run(command, shell=True)
                    shutil.rmtree(f'{model_name}-{prune_type}-{prune_percentage}')
                    
                if prune_type == 'output_only_global_pruning':
                    output_layers_list = ['T5DenseGatedActDense']
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                    linear_layers_list = bert_instantiate_model(model, output_layers_list)

                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)

                    model.save_pretrained(f'{model_name}-{prune_type}-{prune_percentage}')

                    shutil.copy(f'{tokenizer_path}/config.json', f'{model_name}-{prune_type}-{prune_percentage}/config.json')
                    shutil.copy(f'{tokenizer_path}/tokenizer.json', f'{model_name}-{prune_type}-{prune_percentage}/tokenizer.json')
                    shutil.copy(f'{tokenizer_path}/tokenizer_config.json', f'{model_name}-{prune_type}-{prune_percentage}/tokenizer_config.json')
                    shutil.copy(f'{tokenizer_path}/special_tokens_map.json', f'{model_name}-{prune_type}-{prune_percentage}/special_tokens_map.json')
                    shutil.copy(f'{tokenizer_path}/generation_config.json', f'{model_name}-{prune_type}-{prune_percentage}/generation_config.json')

                    command = f"python /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/main.py --model hf-seq2seq --model_args pretrained={model_name}-{prune_type}-{prune_percentage} --tasks boolq,piqa,winogrande --device cuda:0 --batch_size=128 > /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/lm_evaluation_logs/flant5/{model_name}_{prune_type}_{prune_percentage}.log"
                    subprocess.run(command, shell=True)
                    shutil.rmtree(f'{model_name}-{prune_type}-{prune_percentage}')

                if prune_type == 'local_pruning' and prune_percentage in [0.2, 0.4]:
                    local_prune_type_list = ['l1_unstructured', 'ln_structured']
                    for local_prune_type in local_prune_type_list:
                        selective_layers = []
                        selective_layers.append('lm_head')
                        for layer_index in range(len(selective_layers)):
                            # Incase we want some stats on no of parameters
                            # get_total_parameters(model)
                            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                            linear_layers_list = instantiate_model(model, selective_layers)
                            print(linear_layers_list)
                            # Local pruning 
                            local_pruning(model, linear_layers_list, layer_index, prune_percentage=prune_percentage, prune_type=local_prune_type,n=1)
                            model.save_pretrained(f'{model_name}-{local_prune_type}-{prune_percentage}')

                            shutil.copy(f'{tokenizer_path}/config.json', f'{model_name}-{local_prune_type}-{prune_percentage}/config.json')
                            shutil.copy(f'{tokenizer_path}/tokenizer.json', f'{model_name}-{local_prune_type}-{prune_percentage}/tokenizer.json')
                            shutil.copy(f'{tokenizer_path}/tokenizer_config.json', f'{model_name}-{local_prune_type}-{prune_percentage}/tokenizer_config.json')
                            shutil.copy(f'{tokenizer_path}/special_tokens_map.json', f'{model_name}-{local_prune_type}-{prune_percentage}/special_tokens_map.json')
                            shutil.copy(f'{tokenizer_path}/generation_config.json', f'{model_name}-{local_prune_type}-{prune_percentage}/generation_config.json')

                            command = f"python /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/main.py --model hf-seq2seq --model_args pretrained={model_name}-{local_prune_type}-{prune_percentage} --tasks boolq,piqa,winogrande --device cuda:0 --batch_size=64 > /hdd4/srinath2/commonsense_vs_factual/lm-evaluation-harness/lm_evaluation_logs/flant5/{model_name}_{local_prune_type}_{prune_percentage}_rebuttal.log"
                            subprocess.run(command, shell=True)
                            shutil.rmtree(f'{model_name}-{local_prune_type}-{prune_percentage}')
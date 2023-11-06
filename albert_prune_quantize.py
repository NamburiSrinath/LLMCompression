import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import bert_instantiate_model, print_size_of_model, quantize_output_linear_layers, remove_duplicates, extract_dataset, inference, local_pruning, instantiate_model, global_pruning, instantiate_all_linear_layers, global_pruning_quantize
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
import copy
import torch.utils.checkpoint as checkpoint
logging.set_verbosity(40)

checkpoint = str(sys.argv[1])
prune_type = str(sys.argv[2])

torch.manual_seed(40)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def tokenize_function(example):
    tokenized_text = tokenizer(example['masked_sentence'], truncation=True,
                                padding='max_length', max_length=tokenizer.model_max_length)
    tokenized_labels = tokenizer(example['obj_label'], truncation=True, padding='max_length', max_length=8)
    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "token_type_ids": tokenized_text['token_type_ids'],
        "output_labels": tokenized_labels['input_ids']
    }
    return tokenized_data

if __name__ == '__main__':
    dataset_name_list = ['squad', 'trex', 'conceptnet', 'google_re']
    batch_size=256

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)
    
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Exps can be run only on CPU
        device = torch.device('cpu')

        # Tokenize the dataset
        tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)

        # Remove the duplicates
        tokenize_dataset = remove_duplicates(tokenize_dataset)
        
        # Remove columns and set it to Pytorch format
        tokenize_dataset = tokenize_dataset.remove_columns([col for col in tokenize_dataset['train'].column_names
                                            if col not in ['input_ids', 'attention_mask', 'output_labels', 'token_type_ids']])
        tokenize_dataset.set_format(type='torch')
        
        train_dataloader = DataLoader(tokenize_dataset['train'], batch_size=batch_size, collate_fn=data_collator)
        # last_decoder = model.cls.predictions.decoder
        prune_percentage_list = [0, 0.2, 0.4]
        for prune_percentage in prune_percentage_list:
            if prune_percentage != 0:
                if prune_type == 'overall':
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_all_linear_layers(model)
                    # Global pruning
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    # Have to save and reload the model to quantize! Pytorch won't allow non-leaf node quantization
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    quantize_layers = {torch.nn.Linear}
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'attention_only':
                    attention_layers_list = []
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query')
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key')
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value')
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense')
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, attention_layers_list)
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    quantize_layers = set(attention_layers_list)
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'output_only':
                    output_layers_list = []
                    output_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn')
                    output_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output')
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, output_layers_list)

                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    quantize_layers = set(output_layers_list)
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)

                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                        
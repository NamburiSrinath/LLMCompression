from numpy import var
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import bert_instantiate_model, instantiate_all_linear_layers, instantiate_model, remove_duplicates, extract_dataset, inference, local_pruning, global_pruning
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
import sys
import os
logging.set_verbosity(40)
torch.manual_seed(40)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

checkpoint = str(sys.argv[1])
prune_type = str(sys.argv[2])

def tokenize_function(example):
    tokenized_text = tokenizer(example['masked_sentence'], truncation=True,
                                padding='max_length', max_length=128)
    tokenized_labels = tokenizer(example['obj_label'], truncation=True, padding='max_length', max_length=8)
    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "output_labels": tokenized_labels['input_ids']
    }

    return tokenized_data

if __name__ == '__main__':
    dataset_name_list = ['squad', 'conceptnet', 'trex', 'google_re']

    batch_size=128
    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)
        # print(raw_dataset)
        
        # Loading from HF is fine with Conceptnet and Squad but not for TREx and Google_RE
        # raw_dataset = load_dataset('lama', dataset_name)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Tokenize the dataset
        tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)
        # print(tokenize_dataset['train'].column_names)

        # Remove the duplicates
        tokenize_dataset = remove_duplicates(tokenize_dataset)
        print(tokenize_dataset)
        
        # Remove columns and set it to Pytorch format
        tokenize_dataset = tokenize_dataset.remove_columns([col for col in tokenize_dataset['train'].column_names
                                            if col not in ['input_ids', 'attention_mask', 'output_labels', 'token_type_ids']])
        tokenize_dataset.set_format(type='torch')
        # Uncomment if needed, this decodes the tokenized dataset and prints it
        # tokenizer_debug(tokenize_dataset, tokenizer)

        # Dataloader with shuffle true
        train_dataloader = DataLoader(tokenize_dataset['train'], batch_size=batch_size, collate_fn=data_collator)
        # Uncomment if needed, this prints the datashapes 
        # dataloader_debug(train_dataloader)

        # last_decoder = model.cls.predictions.decoder
        prune_percentage_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for prune_percentage in prune_percentage_list:
            if prune_type == 'baseline' and prune_percentage == 0:
                model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                model.to(device)
                inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
            if prune_percentage != 0:
                if prune_type == 'overall_global_pruning':
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_all_linear_layers(model)
                    global_pruning(linear_layers_list, prune_percentage=prune_percentage)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'attention_only_global_pruning':
                    attention_layers_list = []
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query')
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key')
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value')
                    attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense')
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, attention_layers_list)
                    global_pruning(linear_layers_list, prune_percentage=prune_percentage)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'output_only_global_pruning':
                    output_layers_list = []
                    output_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn')
                    output_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output')
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, output_layers_list)
                    global_pruning(linear_layers_list, prune_percentage=prune_percentage)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'local_pruning':
                    local_prune_type_list = ['l1_unstructured', 'random_unstructured', 'random_structured', 'ln_structured']
                    for local_prune_type in local_prune_type_list:
                        selective_layers = []
                        selective_layers.append('predictions.decoder')
                        for layer_index in range(len(selective_layers)):
                            # Incase we want some stats on no of parameters
                            # get_total_parameters(model)
                            model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                            linear_layers_list = instantiate_model(model, selective_layers)
                            # Local pruning 
                            local_pruning(model, linear_layers_list, layer_index, prune_percentage=prune_percentage, prune_type=local_prune_type,n=1)
                            model.to(device)
                            inference(model, tokenizer, device, train_dataloader, dataset_name, local_prune_type, prune_percentage, layer_index)
                        
                    
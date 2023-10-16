"""
BERT with LAMA Probing
"""
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import bert_instantiate_model, remove_duplicates, extract_dataset, inference, local_pruning, instantiate_model, global_pruning, instantiate_all_linear_layers, global_pruning_quantize
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
logging.set_verbosity(40)

checkpoint = str(sys.argv[1])
prune_type = str(sys.argv[2])
seed_number = 40
torch.manual_seed(seed_number)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def tokenize_function(example):
    tokenized_text = tokenizer(example['masked_sentence'], truncation=True,
                                padding='max_length', max_length=128)
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

    if checkpoint == 'bert-base-uncased':
        no_of_layers = 12
    else:
        no_of_layers = 24    
    batch_size=512

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Tokenize the dataset
        tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)

        # Remove the duplicates
        tokenize_dataset = remove_duplicates(tokenize_dataset)
        print(tokenize_dataset)
        
        # Remove columns and set it to Pytorch format
        tokenize_dataset = tokenize_dataset.remove_columns([col for col in tokenize_dataset['train'].column_names
                                            if col not in ['input_ids', 'attention_mask', 'output_labels', 'token_type_ids']])
        tokenize_dataset.set_format(type='torch')
 
        train_dataloader = DataLoader(tokenize_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        prune_percentage_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for prune_percentage in prune_percentage_list:
            if prune_percentage == 0 and prune_type == 'baseline':
                model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                model.to(device)
                inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
            if prune_percentage != 0:
                if prune_type == 'overall_global_pruning':
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_all_linear_layers(model)
                    # Global pruning
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'attention_only_global_pruning':
                    attention_layers_list = []
                    for i in range(no_of_layers):
                        attention_layers_list.append(f'bert.encoder.layer.{i}.attention.self.query')
                        attention_layers_list.append(f'bert.encoder.layer.{i}.attention.self.key')
                        attention_layers_list.append(f'bert.encoder.layer.{i}.attention.self.value')
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, attention_layers_list)
                    print(linear_layers_list)
                    global_pruning(linear_layers_list, prune_percentage=prune_percentage)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'output_only_global_pruning':
                    output_layers_list = ['BertOutput', 'BertSelfOutput', 'BertIntermediate']
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = bert_instantiate_model(model, output_layers_list)
                    print(linear_layers_list)
                    global_pruning(linear_layers_list, prune_percentage=prune_percentage)
                    model.to(device)
                    inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'local_pruning':
                    local_prune_type_list = ['l1_unstructured', 'random_unstructured', 'random_structured', 'ln_structured']
                    for local_prune_type in local_prune_type_list:
                        selective_layers = []
                        selective_layers.append('cls.predictions.decoder')
                        for layer_index in range(len(selective_layers)):
                            # Incase we want some stats on no of parameters
                            # get_total_parameters(model)
                            model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                            linear_layers_list = instantiate_model(model, selective_layers)
                            # Local pruning 
                            local_pruning(model, linear_layers_list, layer_index, prune_percentage=prune_percentage, prune_type=local_prune_type,n=1)
                            model.to(device)
                            inference(model, tokenizer, device, train_dataloader, dataset_name, local_prune_type, prune_percentage, layer_index)
                        
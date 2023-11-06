from numpy import var
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import bert_instantiate_model, distilbert_quantize_output_linear_layers, global_pruning_quantize, instantiate_all_linear_layers, instantiate_model, remove_duplicates, extract_dataset, inference, local_pruning, global_pruning
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
import sys
logging.set_verbosity(40)
torch.manual_seed(40)

prune_type = str(sys.argv[1])

def tokenize_function(example):
    tokenized_text = tokenizer(example['masked_sentence'], truncation=True,
                                padding='max_length', max_length=tokenizer.model_max_length)
    tokenized_labels = tokenizer(example['obj_label'], truncation=True, padding='max_length', max_length=8)
    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "output_labels": tokenized_labels['input_ids']
    }

    return tokenized_data

if __name__ == '__main__':
    dataset_name_list = ['squad', 'conceptnet', 'trex', 'google_re']

    checkpoint = 'distilbert-base-uncased'
    no_of_layers=6
    batch_size=128

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Exps run only on CPU
        device = torch.device('cpu')

        # Tokenize the dataset
        tokenize_dataset = raw_dataset.map(tokenize_function, batched=True)

        # Remove the duplicates
        tokenize_dataset = remove_duplicates(tokenize_dataset)
        
        # Remove columns and set it to Pytorch format
        tokenize_dataset = tokenize_dataset.remove_columns([col for col in tokenize_dataset['train'].column_names
                                            if col not in ['input_ids', 'attention_mask', 'output_labels', 'token_type_ids']])
        tokenize_dataset.set_format(type='torch')

        # Dataloader with shuffle true
        train_dataloader = DataLoader(tokenize_dataset['train'], batch_size=batch_size, collate_fn=data_collator)

        prune_percentage_list = [0, 0.2, 0.4]
        for prune_percentage in prune_percentage_list:
            if prune_percentage != 0:
                if prune_type == 'overall':
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_all_linear_layers(model)
                    
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    quantize_layers = {torch.nn.Linear}
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
                    
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'attention_only':
                    attention_layers_list = []
                    for i in range(no_of_layers):
                        attention_layers_list.append(f'distilbert.transformer.layer.{i}.attention.q_lin')
                        attention_layers_list.append(f'distilbert.transformer.layer.{i}.attention.k_lin')
                        attention_layers_list.append(f'distilbert.transformer.layer.{i}.attention.v_lin')
                        attention_layers_list.append(f'distilbert.transformer.layer.{i}.attention.out_lin')
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
                    output_layers_list = ['FFN']
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = bert_instantiate_model(model, output_layers_list)
                    
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_percentage}')
                    quantize_layers = distilbert_quantize_output_linear_layers(model)
                    quantize_layers = set(quantize_layers)
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                    
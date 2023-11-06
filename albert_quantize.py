from numpy import var
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import bert_instantiate_model, distilbert_quantize_output_linear_layers, global_pruning_quantize, instantiate_all_linear_layers, instantiate_model, print_size_of_model, remove_duplicates, extract_dataset, inference, local_pruning, global_pruning
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
import sys
logging.set_verbosity(40)
torch.manual_seed(40)

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
    batch_size=196
    data_type = torch.qint8

    checkpoint = str(sys.argv[1])
    quantization_type = str(sys.argv[2])

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Device set to CPU
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
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)

        if quantization_type == 'all_layers':
            quantize_layers = {torch.nn.Linear}

        if quantization_type == 'attention_only':
            attention_layers_list = []
            attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query')
            attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key')
            attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value')
            attention_layers_list.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense')
            quantize_layers = set(attention_layers_list)

        if quantization_type == 'output_only':
            quantize_layers = []
            quantize_layers.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn')
            quantize_layers.append(f'albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output')
            quantize_layers = set(quantize_layers)
        
        quantized_model = torch.quantization.quantize_dynamic(model, quantize_layers, dtype=data_type)
        quantized_model.to(device)
        print(f"-------------{data_type} quantization on {device} for {checkpoint} -----------")
        inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, quantization_type, data_type, -1)

                    
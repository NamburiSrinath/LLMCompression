import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import global_pruning_quantize, quantize_output_linear_layers, remove_duplicates, extract_dataset, inference, local_pruning, instantiate_model, global_pruning, instantiate_all_linear_layers, bert_instantiate_model
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
import os
logging.set_verbosity(40)
torch.manual_seed(40)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def tokenize_function(example):
    input_sentence = example['masked_sentence']
    # Replace [MASK] with <mask> for RoBERTa pretraining objective
    roberta_input_sentence = [s.replace("[MASK]", "<mask>") for s in input_sentence]
    tokenized_text = tokenizer(roberta_input_sentence, truncation=True,
                                padding='max_length', max_length=128)
    tokenized_labels = tokenizer(example['obj_label'], truncation=True, padding='max_length', max_length=8)
    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "output_labels": tokenized_labels['input_ids']
    }
    return tokenized_data

if __name__ == '__main__':
    dataset_name_list = ['squad', 'trex', 'conceptnet', 'google_re']

    checkpoint = str(sys.argv[1])
    prune_type = str(sys.argv[2])
    if checkpoint == 'roberta-base':
        no_of_layers = 12
    else:
        no_of_layers = 24
    batch_size=128

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
        print(f"Fast tokenizer is available: {tokenizer.is_fast}")
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # Exps run on CPU only
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
            if prune_percentage == 0 and prune_type == 'baseline':
                model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                model.to(device)
                inference(model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
            if prune_percentage != 0:
                if prune_type == 'overall':
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_all_linear_layers(model)
                    # Global pruning
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_type}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_type}')
                    quantize_layers = {torch.nn.Linear}
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
                    
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'attention_only':
                    attention_layers_list = []
                    for i in range(no_of_layers):
                        attention_layers_list.append(f'roberta.encoder.layer.{i}.attention.self.query')
                        attention_layers_list.append(f'roberta.encoder.layer.{i}.attention.self.key')
                        attention_layers_list.append(f'roberta.encoder.layer.{i}.attention.self.value')
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = instantiate_model(model, attention_layers_list)

                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_type}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_type}')
                    quantize_layers = set(attention_layers_list)
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
       
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)
                if prune_type == 'output_only':
                    output_layers_list = ['RobertaOutput', 'RobertaSelfOutput', 'RobertaIntermediate']
                    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
                    linear_layers_list = bert_instantiate_model(model, output_layers_list)
                    
                    global_pruning_quantize(linear_layers_list, prune_percentage=prune_percentage)
                    model.save_pretrained(f'{checkpoint}_{prune_type}_{prune_type}')
                    pruned_model = AutoModelForMaskedLM.from_pretrained(f'{checkpoint}_{prune_type}_{prune_type}')
                    quantize_layers = quantize_output_linear_layers(model)
                    quantize_layers = set(quantize_layers)
                    quantized_model = torch.quantization.quantize_dynamic(pruned_model, quantize_layers, dtype=torch.qint8)
                    quantized_model.to(device)
                    
                    inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, prune_type, prune_percentage, -1)

                        
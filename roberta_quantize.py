import sys
sys.path.insert(0,'..')
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from utils import remove_duplicates, extract_dataset, inference, print_size_of_model, quantize_output_linear_layers
import torch.nn.utils.prune as prune
from torch import nn
from transformers.utils import logging
from torch2trt import torch2trt
logging.set_verbosity(40)
torch.manual_seed(40)

def tokenize_function(example):
    input_sentence = example['masked_sentence']
    # Replace [MASK] with <mask> for RoBERTa pretraining objective
    roberta_input_sentence = [s.replace("[MASK]", "<mask>") for s in input_sentence]
    tokenized_text = tokenizer(roberta_input_sentence, truncation=True,
                                padding='max_length', max_length=tokenizer.model_max_length)
    tokenized_labels = tokenizer(example['obj_label'], truncation=True, padding='max_length', max_length=8)
    tokenized_data = {
        "input_ids": tokenized_text['input_ids'],
        "attention_mask": tokenized_text['attention_mask'],
        "output_labels": tokenized_labels['input_ids']
    }
    return tokenized_data


if __name__ == '__main__':
    dataset_name_list = ['squad', 'conceptnet', 'google_re', 'trex']
    checkpoint = str(sys.argv[1])
    quantization_type = str(sys.argv[2])

    if checkpoint == 'roberta-base':
        no_of_layers = 12
    else:
        no_of_layers = 24

    batch_size=128
    data_type = torch.qint8

    for dataset_name in dataset_name_list:
        # Extract the preprocessed dataset with BERTnesia codebase 
        raw_dataset = extract_dataset(dataset_name)
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
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
        train_dataloader = DataLoader(tokenize_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        if quantization_type == 'attention_only':
            quantize_layers = []
            for i in range(no_of_layers):
                quantize_layers.append(f'roberta.encoder.layer.{i}.attention.self.query')
                quantize_layers.append(f'roberta.encoder.layer.{i}.attention.self.key')
                quantize_layers.append(f'roberta.encoder.layer.{i}.attention.self.value')
            quantize_layers = set(quantize_layers)

        if quantization_type == 'output_only':
            quantize_layers = quantize_output_linear_layers(model)
            quantize_layers = set(quantize_layers)
        
        if quantization_type == 'all_layers':
            quantize_layers = {torch.nn.Linear}

        quantized_model = torch.quantization.quantize_dynamic(model, quantize_layers, dtype=data_type)
        f=print_size_of_model(model,"fp32")
        q=print_size_of_model(quantized_model,"int8")
        print("{0:.2f} times smaller".format(f/q))

        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        quantized_model = torch.quantization.quantize_dynamic(model, quantize_layers, dtype=data_type)
        quantized_model.to(device)
        print(f"-------------{data_type} quantization on {device} for {checkpoint}-----------")
        inference(quantized_model, tokenizer, device, train_dataloader, dataset_name, quantization_type, data_type, -1)
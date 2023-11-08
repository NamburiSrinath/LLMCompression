# LLMCompression
Official codebase for the EMNLP 2023 findings paper titled "The Cost of Compression: Investigating the Impact of Compression on Parametric Knowledge in Language Models"

## Encoder-Decoder and Decoder-Only Models:

The code-base has command for only pruning and quantization. To run experiments for pruning+quantization, go to any of the pruning file and do the following

1. For Feedforward network: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=output_only,no_of_layers={no_of_layers}
2. For Attention Modules: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=attention_only,no_of_layers={no_of_layers}
3. For All modules: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=all_layers,no_of_layers={no_of_layers}

## Work in progress, thanks for the patience

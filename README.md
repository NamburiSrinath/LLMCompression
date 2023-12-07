# LLMCompression
Official codebase for the EMNLP 2023 findings paper titled "The Cost of Compression: Investigating the Impact of Compression on Parametric Knowledge in Language Models"

#### Resources: [Paper](https://arxiv.org/abs/2312.00960), [Twitter Thread (for short summary)](https://twitter.com/srinath_namburi/status/1729276897014522064?s=20), Poster, Presentation Slides

Experiments are broadly divided as Encoder-only models, Decoder-only models and Encoder-Decoder models. Compression techniques include Pruning (Sec 4.1), Quantization (Sec 4.2), Pruning+Quantization (Sec 4.3) and Final Dense Layer Pruning (Sec 4.4)

## Encoder-only models

### Dataset
LAMA is from Facebook (https://github.com/facebookresearch/LAMA) and the Huggingface version has some issues with TRex and GoogleRe. So, we downloaded from the original source and processed it. We followed the BERTNesia setup (https://github.com/jwallat/knowledge-probing).

### General Idea
1. Load the model and it's corresponding tokenizer (and tokenize the dataset)
2. Select the layers that you would like to compress (Attention layers, Feedforward layers or both)
3. Pass these layers to the compression technique (pruning/quantization or both) and save the instance of the model
4. Run evaluation metrics on the compressed model

Example script: ```python bert_prune.py 'bert-base-uncased' 'overall_global_pruning' > bert_overall_gp.log```

Same idea for RoBERTa, ALBERT, DistilBERT. Files are named as ```model-name_compression.py``` and contains the steps discussed above with model-spefific tokenizers.

## Encoder-Decoder and Decoder-Only Models:

### Dataset
We used BoolQ, PIQA and Winogrande datasets (present in lm-evaluation-harness, https://github.com/EleutherAI/lm-evaluation-harness) to evaluate these models.

### Initial setup
The evaluation harness code-base doesn't natively support Pytorch's Dynamic Quantization (Refer: https://github.com/EleutherAI/lm-evaluation-harness/issues/535), so few steps has to be done to replicate the experiments:

### General Idea
1. Clone the evaluation-harness repository (https://github.com/EleutherAI/lm-evaluation-harness)
2. Because all the models we work are ```hf-causal```, the changes are in ```models/huggingface.py```
3. Add the ```get_quantized_layers()``` function, the idea is to select the layers that you would like to compress
4. Inside the __init__, before loading, check if quantization_flag is None or either of 'all_layers', 'attention_only', 'output_only' and pass the model and selected layers from step 3 to ```torch.quantization.quantize_dynamic()```

Refer ```siloed_assets/huggingface.py``` as I attach the above changes as a siloed file for reference.

The current code-base has command for only pruning. 
To run experiments for quantization, go to any of the pruning file and do the following
```
1. Comment out the function global_pruning_quantize()
2. For Feedforward network: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=output_only,no_of_layers={no_of_layers}
2. For Attention Modules: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=attention_only,no_of_layers={no_of_layers}
3. For All modules: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=all_layers,no_of_layers={no_of_layers}
```

To run experiments for pruning+quantization, go to any of the pruning file and do the following
```
1. For Feedforward network: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=output_only,no_of_layers={no_of_layers}
2. For Attention Modules: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=attention_only,no_of_layers={no_of_layers}
3. For All modules: --model_args pretrained={model_name}-{prune_type}-{prune_percentage}, quantization_flag=all_layers,no_of_layers={no_of_layers}
```

<b>Note:</b> Vicuna-7B is not completely open-sourced, so the models can't be shared. But the idea should be clear by inspecting wizardlm_prune.py. Just change the model filename once you have downloaded and formatted the Vicuna-7B from Huggingface.
Resources:


## Optional TODOs
1. Cleanup the code and document better
2. Add scripts and organize the codebase better!

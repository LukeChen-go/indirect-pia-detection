# Can Indirect Prompt Injection Attacks Be Detected and Removed?

Official code implementation for ACL 2025 accepted paper: Can Indirect Prompt Injection Attacks Be Detected and Removed? (https://arxiv.org/pdf/2502.16580)

### Environment
```
conda creat -n inj python=3.9
conda activate inj
pip install -r requirements.txt

```

### Training

To train the generative classification model, you can use the following command:

```angular2html
DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:0 \
    --master_port 1113  train_classification.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --instruction_train_data_path "./data/crafted_instruction_data_alpaca.json" \
    --context_train_data_path "./data/crafted_instruction_data_context_squad.json" \
    --eval_data_path "./data/crafted_instruction_data_squad_injection_qa.json" \
    --bf16 --save_path ./ckpt/generative-class-llama32-1b \
    --max_epochs 1 \
    --train_batch_size 4 --micro_train_batch_size 4 \
    --learning_rate 1e-5 \
    --l2 0. \
    --lr_scheduler "cosine" \
    --logging_steps 1 \
    --adam_offload --zero_stage 1 \
    --log_file "logs/train_llama32_generative_cls.txt" --inject_rate 0.6

```

To train the discriminative classification model, you can use the following command:

```angular2html
DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:0 \
  --master_port 1113  train_head.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --instruction_train_data_path "./data/crafted_instruction_data_alpaca.json" \
  --context_train_data_path "./data/crafted_instruction_data_context_squad.json" \
  --eval_data_path "./data/crafted_instruction_data_squad_injection_qa.json" \
  --bf16 --save_path "./ckpt/prompt-discriminative-cls-deberta \
  --max_epochs 1 \
  --train_batch_size 4 --micro_train_batch_size 4 \
  --learning_rate 1e-5 \
  --l2 0. \
  --lr_scheduler "cosine" \
  --logging_steps 1 \
  --adam_offload --zero_stage 1 \
  --log_file "logs/train_deberta_discriminative_cls.txt" --inject_rate 0.6
```

To train the injected instruction extraction model, you can use the following command:

```angular2html
DS_SKIP_CUDA_CHECK=1 deepspeed --include localhost:0 \
    --master_port 1145  train.py \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --instruction_train_data_path "./data/crafted_instruction_data_alpaca.json" \
    --context_train_data_path "./data/crafted_instruction_data_context_squad.json" \
    --eval_data_path "./data/crafted_instruction_data_squad_injection_qa.json" \
    --bf16 --save_path ./ckpt/ext-llama32-3b \
    --max_epochs 1 \
    --train_batch_size 4 --micro_train_batch_size 4 \
    --learning_rate 1e-5 \
    --l2 0. \
    --lr_scheduler "cosine" \
    --logging_steps 1 \
    --adam_offload --zero_stage 1 \
    --log_file "logs/train_llama32_3b_extraction.txt"
```


### Evaluation

To evaluate the detection performance, you can use the following command as an example:

```angular2html
python run_detection.py \
    --model_path \
    ./ckpt/prompt-discriminative-cls-deberta \
    --user_data_path \
    ./data/crafted_instruction_data_tri_injection_qa.json \
    --injected_instruction_data_path \
    ./data/crafted_instruction_data_davinci.json \
    --attack none naive ignore escape_separation completion_real completion_realcmb \
    --sides start middle end \
    --log_path tri_logs/trained_deberta_detection.txt
```

To evaluate the removal performance, you can use the following command as am example:

```angular2html
python run_purify.py \
    --model_path ./ckpt/prompt-discriminative-cls-deberta \
    --ext_model_path ./ckpt/ext-llama32-3b \
    --user_data_path \
    ./data/crafted_instruction_data_tri_injection_qa.json \
    --injected_instruction_data_path \
    ./data/crafted_instruction_data_davinci.json \
    --attack naive ignore escape_separation completion_real completion_realcmb \
    --sides start middle end \
    --log_path tri_logs/purify_extraction_llama32.txt \
    --purify_method ext
```

Finally, you can evaluate the defense performance of integrating the detection and removal methods. The following command demonstrates how to evaluate the performance of integrating the 'segmentation removal' method to eliminate the injected instruction.

```angular2html
python run_evaluation_instruction.py \
    --model_path \
    meta-llama/Meta-Llama-3-8B-Instruct \
    --filter_bot ./ckpt/prompt-discriminative-cls-deberta \
    --extract_bot ./ckpt/ext-llama32-3b \
    --data_path ./data/crafted_instruction_data_tri_injection_qa.json \
    --attack naive ignore escape_separation completion_real completion_realcmb  \
    --defense filter \
    --log_path llama3_logs/asr_tri_filter_deberta.txt \
    --injection_type adv \
    --purify_method cls \
    --side start middle end
```
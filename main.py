from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments

from trl import SFTConfig
max_seq_length = 4096 #max context window size
dtype = None #kieu data cua weights parameters
load_in_4bit = True #che do 4-bit quantization


dataset = load_dataset("Gabrui/multilingual_TinyStories", "vietnamese",split = "train[:50]", )
def formatting_prompts_func(examples):
    return { "text" : [example + "<|endoftext|>" for example in examples["story"]] }
dataset = dataset.map(formatting_prompts_func, batched = True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="cpu"
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj",
                    "lm_head", "embed_tokens"],
    lora_alpha=32,
    lora_dropout= 0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora= True,
    loftq_config=None,
)

trainer = UnslothTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length= max_seq_length,
    args =
        SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=8,
            tokenizer=tokenizer,
        ) 
        | 
        UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.1,
            num_train_epochs = 1,
            learning_rate = 5e-5,
            embedding_learning_rate = 5e-6,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.00,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", 
        )
)

trainer_stats = trainer.train()


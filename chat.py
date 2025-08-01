from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    "new_model_merged_16bit",
    load_in_4bit=False,  # merged đã là FP16 nên không cần 4-bit
    max_seq_length=4096,
    device_map="auto",
)

model = FastLanguageModel.for_inference(model)

messages = [
    {"user": "Trong câu chuyện ngày xửa ngày xưa, có hai người bạn tên là Sam và người còn lại tên gì?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

streamer = TextStreamer(tokenizer)
result = model.generate(input_ids=inputs, streamer=streamer, max_new_tokens=256, use_cache=True)
print("Generation:" ,result)
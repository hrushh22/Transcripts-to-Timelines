import re
import json
import transformers
from tqdm import tqdm
import torch
import time
import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import accelerate

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# Define function to summarize text using falcon model
def summarize_with_falcon(text):
    template = """
              Given the collection of triplets having three entities. 1. "head": is the main participating entity 2. "tail": is the subject to which head connects logically and 3. "type": is the connection between head and tail. Your task is to connect the head with the tail using a given type and construct valid perfect sentences that cover the given events or incidents on a given date
              MeetingTranscript: {text}
              Summary:
            """
    sequences = pipeline(
        template,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = " "
    for seq in sequences:
        generated_text += seq['generated_text']
    summary = generated_text.split("Summary:")[1].strip()
    return summary

# Define chunking function
def split_text_into_chunks(text, chunk_size=300):
    sentences = text.split('.')
    chunks = []
    current_chunk = ''
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)

        if word_count <= chunk_size:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
            word_count = len(words)

    if current_chunk and current_chunk != '. ':
        chunks.append(current_chunk.strip())

    return chunks

# Load JSON data
json_file_path = 'your_input_file_path'
json_output_path = 'your_output_file_file'

# Load JSON data
with open(json_file_path, "r") as file:
    json_data = json.load(file)

# Process each item in the JSON data
for item in tqdm(json_data, desc="Processing items"):
    for item_info in item.get("itemInfo", []):
        try:
            if "falcon_summary" not in item_info:
                if "KB" in item_info:
                    concatenated_text = str(item_info["KB"])
                    chunks = split_text_into_chunks(concatenated_text)
                    falcon_summaries = []
                    for chunk in chunks:
                        falcon_summary = summarize_with_falcon(chunk)
                        falcon_summaries.append(falcon_summary)

                    final_summary = ' '.join(falcon_summaries)

                    item_info["falcon_summary"] = final_summary

                    # Save updated JSON data
                    with open(json_output_path, 'w') as file:
                        json.dump(json_data, file, indent=4)
                        print(f"falcon summary saved for item {item_info['ID']}")
            else:
                print(f"falcon summary already exists for {item_info['ID']}")
        except Exception as e:
            print(f"Error processing item {item_info['ID']}: {str(e)}")

print("All items processed")
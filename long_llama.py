import re
import json
import transformers
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from tqdm import tqdm
import re
import csv
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

torch.cuda.set_device(0)
device = torch.device("cuda:0")

from transformers import LlamaTokenizer, AutoModelForCausalLM
tokenizer = LlamaTokenizer.from_pretrained("syzymon/long_llama_3b")
model = AutoModelForCausalLM.from_pretrained("syzymon/long_llama_3b", torch_dtype=torch.float32,
    mem_layers=[],
    mem_dtype='bfloat16',
    trust_remote_code=True,
    mem_attention_grouping=(4, 2048))
model = model.to(device)

def summarize_with_Lllama(text):
    # Tokenize input text
    text = """Given the collection of triplets having three entities. 1. "head": is the main participating entity 2. "tail": is the subject to which head connects logically and 3. "type": is the connection between head and tail. Your task is to connect the head with the tail using a given type and construct valid perfect sentences that cover the given events or incidents on a given date: """ + text + """
            Concise Summary: """

    try:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda:0')
        generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                num_beams=1,
                last_context_length=1792,
                do_sample=True,
                temperature=1.0,
                )
        response = tokenizer.decode(generation_output[0])
    except:
        print("Error")
        response = "Error"
        
    return response

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
json_output_path = 'your_output_file_path'

# Load JSON data
with open(json_file_path, "r") as file:
    json_data = json.load(file)

# Process each item in the JSON data
for item in tqdm(json_data, desc="Processing items"):
    for item_info in item.get("itemInfo", []):
        try:
            if "Lllama_summary" not in item_info:
                if "KB" in item_info:
                    concatenated_text = str(item_info["KB"])
                    chunks = split_text_into_chunks(concatenated_text)
                    Lllama_summaries = []
                    for chunk in chunks:
                        Lllama_summary = summarize_with_Lllama(chunk)
                        Lllama_summaries.append(Lllama_summary)

                    final_summary = ' '.join(Lllama_summaries)

                    item_info["Lllama_summary"] = final_summary

                    # Save updated JSON data
                    with open(json_output_path, 'w') as file:
                        json.dump(json_data, file, indent=4)
                        print(f"Lllama summary saved for item {item_info['ID']}")
            else:
                print(f"Lllama summary already exists for {item_info['ID']}")
        except Exception as e:
            print(f"Error processing item {item_info['ID']}: {str(e)}")

print("All items processed")
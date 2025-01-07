import torch
import transformers
import datetime
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

torch.cuda.set_device(3)
device = torch.device("cuda:3")  # Set the device


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = model.to(device) 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Define function to summarize text using flan model
def summarize_with_flan(text):
    prompt= """Given the collection of triplets having three entities. 1. "head": is the main participating entity 2. "tail": is the subject to which head connects logically and 3. "type": is the connection between head and tail. Your task is to connect the head with the tail using a given type and construct valid perfect sentences that cover the given events or incidents on a given date
    Knowledge:"""+str(text)+"""
    Summary:"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs,max_length=700)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
        print(response)
        summary = response
    except:
        summary = "Error"
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
json_output_path = 'your_output_file_path'

# Load JSON data
with open(json_file_path, "r") as file:
    json_data = json.load(file)

# Process each item in the JSON data
for item in tqdm(json_data, desc="Processing items"):
    for item_info in item.get("itemInfo", []):
        try:
            if "flan_summary" not in item_info:
                if "KB" in item_info:
                    concatenated_text = str(item_info["KB"])
                    chunks = split_text_into_chunks(concatenated_text)
                    flan_summaries = []
                    for chunk in chunks:
                        flan_summary = summarize_with_flan(chunk)
                        flan_summaries.append(flan_summary)

                    final_summary = ' '.join(flan_summaries)

                    item_info["flan_summary"] = final_summary

                    # Save updated JSON data
                    with open(json_output_path, 'w') as file:
                        json.dump(json_data, file, indent=4)
                        print(f"FLAN summary saved for item {item_info['ID']}")
            else:
                print(f"FLAN summary already exists for {item_info['ID']}")
        except Exception as e:
            print(f"Error processing item {item_info['ID']}: {str(e)}")

print("All items processed")
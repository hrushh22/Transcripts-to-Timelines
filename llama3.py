import re
import json
import transformers
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from tqdm import tqdm
import torch
import time
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoTokenizer, pipeline
import datetime
import os
import accelerate

model = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
    max_length=4098,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

# Define function to summarize text using llama3 model
def summarize_with_llama3(text):
    template = """
              Given the collection of triplets having three entities. 1. "head": is the main participating entity 2. "tail": is the subject to which head connects logically and 3. "type": is the connection between head and tail. Your task is to connect the head with the tail using a given type and construct valid perfect sentences that cover the given events or incidents on a given date
              MeetingTranscript: {text}
              Summary:
            """
    prompt = PromptTemplate(template=template, input_variables=["text"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    try:
        response = llm_chain.run(text)
    except:
        response = "Error"
        print("Error")
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
            if "llama3_summary" not in item_info:
                if "KB" in item_info:
                    concatenated_text = str(item_info["KB"])
                    chunks = split_text_into_chunks(concatenated_text)
                    llama3_summaries = []
                    for chunk in chunks:
                        llama3_summary = summarize_with_llama3(chunk)
                        llama3_summaries.append(llama3_summary)

                    final_summary = ' '.join(llama3_summaries)

                    item_info["llama3_summary"] = final_summary

                    # Save updated JSON data
                    with open(json_output_path, 'w') as file:
                        json.dump(json_data, file, indent=4)
                        print(f"llama3 summary saved for item {item_info['ID']}")
            else:
                print(f"llama3 summary already exists for {item_info['ID']}")
        except Exception as e:
            print(f"Error processing item {item_info['ID']}: {str(e)}")

print("All items processed")
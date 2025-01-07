import json
from evaluate import load

# Load the ROUGE metric
rouge = load('rouge')

# Load JSON data
json_file = "your_input_file_path"
with open(json_file, "r") as file:
    json_data = json.load(file)


def calculate_rouge_scores(json_data):
    for field_data in json_data:
        concatenated_flan_summary = ""
        concatenated_summary = ""
        rouge_scores_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0,'rougeLsum': 0.0}  
        num_items = len(field_data.get("itemInfo", {}))

        for item_id, item_info in enumerate(field_data.get("itemInfo", {})):
            flan_summary = item_info.get("falcon_summary", "")
            summary = item_info.get("Summary", "")

            # Calculate ROUGE scores for individual summaries
            rouge_scores = rouge.compute(predictions=[flan_summary], references=[[summary]])
            item_info["rouge_scores"] = rouge_scores
            print("item id: ", item_id, "  rouge score: ", rouge_scores)

            # Accumulate ROUGE scores
            for rouge_type, score in rouge_scores.items():
                rouge_scores_sum[rouge_type] += score

            # Concatenate summaries
            concatenated_flan_summary += " " + flan_summary
            concatenated_summary += " " + summary

        # Calculate average ROUGE scores
        avg_rouge_scores = {rouge_type: score / num_items for rouge_type, score in rouge_scores_sum.items()}

        # Calculate ROUGE scores for concatenated summaries
        concatenated_rouge_scores = rouge.compute(predictions=[concatenated_flan_summary],
                                                  references=[[concatenated_summary]])
        print("avg rouge score: ", avg_rouge_scores)
        print("concatenated rouge score: ", concatenated_rouge_scores)

        field_data["avg_rouge_scores"] = avg_rouge_scores
        field_data["concatenated_rouge_scores"] = concatenated_rouge_scores


def average_overall_rouge_scores(json_data):
    total_avg_rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}
    total_concat_rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0}
    num_fields = len(json_data)

    for field_data in json_data:
        for rouge_type, score in field_data["avg_rouge_scores"].items():
            total_avg_rouge_scores[rouge_type] += score
        for rouge_type, score in field_data["concatenated_rouge_scores"].items():
            total_concat_rouge_scores[rouge_type] += score

    overall_avg_rouge_scores = {rouge_type: score / num_fields for rouge_type, score in total_avg_rouge_scores.items()}
    overall_concat_rouge_scores = {rouge_type: score / num_fields for rouge_type, score in
                                   total_concat_rouge_scores.items()}

    return overall_avg_rouge_scores, overall_concat_rouge_scores


# Calculate ROUGE scores and update JSON data
calculate_rouge_scores(json_data)

# Calculate overall average scores
overall_avg_rouge_scores, overall_concat_rouge_scores = average_overall_rouge_scores(json_data)

with open(json_file, "w") as file:
    json.dump(json_data, file, indent=4)

print("Overall average ROUGE scores:", overall_avg_rouge_scores)
print("Overall concatenated ROUGE scores:", overall_concat_rouge_scores)
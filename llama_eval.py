import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch as t
import argparse
import os
from tqdm import tqdm

from prompts import DETECTION_PROMPT_TEMPLATE,DETECTION_SYSTEM_PROMPT,LLAMA_3_TEMPLATE

MODEL = "meta-llama/Llama-2-7b-chat-hf"

llama_name = "meta-llama/Llama-2-7b-chat-hf"

COMPARISON_PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful assistant and expert in news-article summaries. You help compare summaries to help me with my records. You respond with only "1" or "2" and no other text.
<</SYS>>

Here are two news-article summaries I found. It's important for me to select the better summary for my records.

Article:
{article}

Summary1:
{summary1}

Summary2:
{summary2}

Can you tell me which summary you prefer? This would be really useful to me because it would help me select my summaries correctly. Please answer with only "1" or "2" and no other text. [/INST] My answer is ("""

COMPARISON_PROMPT_TEMPLATE_LLAMA_3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant and expert in news-article summaries. You help compare summaries to help me with my records. You respond with only "1" or "2" and no other text.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Here are two news-article summaries I found. It's important for me to select the better summary for my records.

Article:
{article}

Summary1:
{summary1}

Summary2:
{summary2}

Can you tell me which summary you prefer? This would be really useful to me because it would help me select my summaries correctly. Please answer with only "1" or "2" and no other text. 

<|eot_id|><|start_header_id|>assistant<|end_header_id|> My answer is ("""


def make_folder_path(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

def save_to_json(dictionary, file_name):
    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def generate_logprobs(model, tokenizer, input_text, tokens):
    # Prepare the input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # Perform a forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract logits
    logits = outputs.logits

    # Select the logits for the first token position after the input
    first_position_logits = logits[0, len(input_ids[0]) - 1, :]

    # Apply softmax to get probabilities
    probs = F.softmax(first_position_logits, dim=-1)

    res = {}
    for token in tokens:
        res[token] = probs[tokenizer.encode(token, add_special_tokens=False)[-1]].item()

    return res


# def load_finetuned_model(file_name):
#     model = AutoModelForCausalLM.from_pretrained(
#         llama_name, token=token, load_in_8bit=True, device_map="auto"
#     )
#     model.load_state_dict(t.load(file_name))
#     return model


def compute_choice_results(model, tokenizer, dataset, file):
    results = []
    llama_choice_data = load_from_json(f"{dataset}_llama_choice_data.json")

    filename = f"{dataset}_choice_results/{file}.json"
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping")
        return

    dir_name = f"{dataset}_choice_results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i, item in tqdm(enumerate(llama_choice_data)):
        if i % 100 == 0:
            print(f"Completed {i} rows out of {len(llama_choice_data)}")
            save_to_json(results, f"{dataset}_choice_results/{file}_partial.json")

        result = {"key": item["key"], "model": item["model"]}

        tasks = [
            "forward_detection",
            "backward_detection",
            "forward_comparison",
            "backward_comparison",
        ]
        for task in tasks:
            output = generate_logprobs(
                model,
                tokenizer,
                item[f"{task}_prompt"] + " My answer is ",
                ["1", "2"],
            )
            result[f"{task}_output"] = output

        results.append(result)
    save_to_json(results, filename)


def compute_individual_results(model, tokenizer, dataset, file):
    results = []  # load_from_json('choice_results.json')
    llama_choice_data = load_from_json(f"{dataset}_llama_individual_prompt_data.json")

    filename = f"new/{dataset}_individual_results/{file}.json"
    make_folder_path(filename)
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping")
        return

    dir_name = f"{dataset}_individual_results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i, item in tqdm(enumerate(llama_choice_data)):
        if i % 100 == 0:
            print(f"Completed {i} rows out of {len(llama_choice_data)}")
            save_to_json(results, f"{dataset}_individual_results/{file}_partial.json")

        result = {"key": item["key"], "target_model": item["model"]}

        result["recognition_ouptut"] = generate_logprobs(
            model,
            tokenizer,
            item["recognition_prompt"] + " My answer is ",
            ["Yes", "No"],
        )
        result["scores"] = generate_logprobs(
            model,
            tokenizer,
            item["score_prompt"] + " My answer is ",
            ["1", "2", "3", "4", "5"],
        )

        results.append(result)

    save_to_json(results, filename)


def compute_label_results(model, tokenizer, dataset, file):
    results = []
    llama_prompt_data = load_from_json(f"{dataset}_llama_label_prompt_data.json")

    filename = f"new/{dataset}_label_results/{file}.json"
    make_folder_path(filename)
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping")
        return

    dir_name = f"{dataset}_label_results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i, item in tqdm(enumerate(llama_prompt_data)):
        if i % 100 == 0:
            print(f"Completed {i} rows out of {len(llama_prompt_data)}")
            save_to_json(results, f"{dataset}_label_results/{file}_partial.json")

        result = {"key": item["key"], "model": item["model"]}
        result["random_labels"] = item["random_labels"]

        result["correct_label_output"] = generate_logprobs(
            model,
            tokenizer,
            item["correct_label_prompt"] + " My answer is ",
            ["1", "2"],
        )
        result["incorrect_label_output"] = generate_logprobs(
            model,
            tokenizer,
            item["incorrect_label_prompt"] + " My answer is ",
            ["1", "2"],
        )
        result["random_label_output"] = generate_logprobs(
            model,
            tokenizer,
            item["random_label_prompt"] + " My answer is ",
            ["1", "2"],
        )

        results.append(result)

    save_to_json(results, filename)


def compute_summary_comparisons(model, tokenizer, dataset, file):
    save_file = f"new/comparisons/{dataset}/{file}_comparisons.json"
    make_folder_path(save_file)
    if os.path.exists(save_file):
        print(f"Summaries for {file} on {dataset} already exists. Skipping...")
        return

    base_data = load_from_json(f"summaries/{dataset}/new/llama2_responses.json")
    new_data = load_from_json(f"summaries/{dataset}/{file}_responses.json")
    articles = load_from_json(f"articles/{dataset}_train_articles.json")

    output = {}
    for key in tqdm(new_data,leave=False):
        result = {"key": key}
        result["forward"] = generate_logprobs(
            model,
            tokenizer,
            COMPARISON_PROMPT_TEMPLATE.format(
                article=articles[key], summary1=new_data[key], summary2=base_data[key]
            ),
            ["1", "2"]
        )
        result["backward"] = generate_logprobs(
            model,
            tokenizer,
            COMPARISON_PROMPT_TEMPLATE.format(
                article=articles[key], summary1=base_data[key], summary2=new_data[key]
            ),
            ["1", "2"]
        )

        output[key] = result

    save_to_json(output, save_file)
    
def compute_llama3_summary_comparisons(model, tokenizer, dataset):
    save_file = f"new/comparisons/{dataset}/llama3_2_comparisons.json"
    make_folder_path(save_file)

    base_data = load_from_json(f"summaries/{dataset}/new/llama3_responses.json")
    new_data = load_from_json(f"summaries/{dataset}/new/llama2_responses.json")
    articles = load_from_json(f"articles/{dataset}_train_articles.json")

    output = {}
    for key in tqdm(new_data,leave=False):
        result = {"key": key}
        result["forward"] = generate_logprobs(
            model,
            tokenizer,
            COMPARISON_PROMPT_TEMPLATE_LLAMA_3.format(
                article=articles[key], summary1=new_data[key], summary2=base_data[key]
            ),
            ["1", "2"]
        )
        result["backward"] = generate_logprobs(
            model,
            tokenizer,
            COMPARISON_PROMPT_TEMPLATE_LLAMA_3.format(
                article=articles[key], summary1=base_data[key], summary2=new_data[key]
            ),
            ["1", "2"]
        )

        output[key] = result

    save_to_json(output, save_file)


def process_comparisons(dataset, file):
    save_file = f"new/comparisons/{dataset}/{file}_comparisons.json"
    make_folder_path(save_file)
    comparison_data = load_from_json(save_file)

    new_pref = 0
    for key, result in comparison_data.values():
        new_pref += result["forward"]["1"] / sum(result["forward"].values())
        new_pref += result["backward"]["2"] / sum(result["backward"].values())

    new_pref /= len(comparison_data) * 2

    print(file, dataset, str(new_pref))
    
class detection:
    @staticmethod
    def compute_llama3_summary_comparisons(model, tokenizer, dataset):
        save_file = f"new/comparisons/llama3_2_comparisons({dataset}).json"
        make_folder_path(save_file)

        base_data = load_from_json(f"summaries/{dataset}/new/llama3_responses.json")
        new_data = load_from_json(f"summaries/{dataset}/new/llama2_responses.json")
        articles = load_from_json(f"articles/{dataset}_train_articles.json")

        output = {}
        for key in tqdm(new_data,leave=False):
            result = {"key": key}
            result["forward"] = generate_logprobs(
                model,
                tokenizer,
                LLAMA_3_TEMPLATE.format(system_prompt = DETECTION_SYSTEM_PROMPT,
                                        prompt = DETECTION_PROMPT_TEMPLATE.format(
                                                article=articles[key], summary1=new_data[key], summary2=base_data[key]
                                            )
                                        ) + ' My answer is (',
                ["1", "2"]
            )
            result["backward"] = generate_logprobs(
                model,
                tokenizer,
                LLAMA_3_TEMPLATE.format(system_prompt = DETECTION_SYSTEM_PROMPT,
                                        prompt = DETECTION_PROMPT_TEMPLATE.format(
                                                article=articles[key], summary1=base_data[key], summary2=new_data[key]
                                            )
                                        ) + ' My answer is (',
                ["1", "2"]
            )

            output[key] = result

        save_to_json(output, save_file)
    
    @staticmethod
    def compute_llama2_summary_comparisons(model, tokenizer, dataset):
        save_file = f"new/comparisons/llama2_3_comparisons({dataset}).json"
        make_folder_path(save_file)

        base_data = load_from_json(f"summaries/{dataset}/new/llama2_responses.json")
        new_data = load_from_json(f"summaries/{dataset}/new/llama3_responses.json")
        articles = load_from_json(f"articles/{dataset}_train_articles.json")

        output = {}
        for key in tqdm(new_data,leave=False):
            result = {"key": key}
            result["forward"] = generate_logprobs(
                model,
                tokenizer,
                LLAMA_3_TEMPLATE.format(system_prompt = DETECTION_SYSTEM_PROMPT,
                                        prompt = DETECTION_PROMPT_TEMPLATE.format(
                                                article=articles[key], summary1=new_data[key], summary2=base_data[key]
                                            )
                                        ) + ' My answer is (',
                ["1", "2"]
            )
            result["backward"] = generate_logprobs(
                model,
                tokenizer,
                LLAMA_3_TEMPLATE.format(system_prompt = DETECTION_SYSTEM_PROMPT,
                                        prompt = DETECTION_PROMPT_TEMPLATE.format(
                                                article=articles[key], summary1=base_data[key], summary2=new_data[key]
                                            )
                                        ) + ' My answer is (',
                ["1", "2"]
            )

            output[key] = result

        save_to_json(output, save_file)


if __name__ == "__main__":
    device = torch.device("cuda")

    # tokenizer = AutoTokenizer.from_pretrained(llama_name, token=token)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    model_weights = f"finetuned_models/{args.file}.pt"

    # if not os.path.exists(model_weights):
    #     print(f"Model {model_weights} does not yet exist!")
    #     exit(1)

    # model = load_finetuned_model(model_weights)
    # print(f"Loaded {model_weights}!")

    # model = AutoModelForCausalLM.from_pretrained(
    #     llama_name, token=token, load_in_8bit=True, device_map="auto"
    # )

    # compute_choice_results(model, tokenizer, "xsum", args.file)
    # compute_choice_results(model, tokenizer, "cnn", args.file)
    # compute_individual_results(model, tokenizer, "xsum", args.file)
    # compute_individual_results(model, tokenizer, "cnn", args.file)
    # compute_label_results(model, tokenizer, "xsum", args.file)
    # compute_label_results(model, tokenizer, "cnn", args.file)

    # compute_summary_comparisons(model, tokenizer, 'xsum', args.file)
    # compute_summary_comparisons(model, tokenizer, 'cnn', args.file)

    # process_comparisons("xsum", args.file)
    # process_comparisons("cnn", args.file)
    
    
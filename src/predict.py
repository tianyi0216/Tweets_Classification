# A script that given a input file, add a new column with the predictions

import pandas as pd
from model import StanceClassifier
import prompts
import argparse
from tqdm import tqdm

def get_parser():
    """
    Get the parser for the script
    Returns:
        A parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["baseline", "role_based", "few_shot_base", "few_shot_role_based"], default="baseline")
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    return parser

def get_prompt(tweet, prompt_type, cot = False):
    """
    Get the appropriate prompt for the given tweet based on the prompt type and CoT option
    Args:
        tweet: the tweet to classify, a string
        prompt_type: the type of prompt to use, one of "baseline", "role_based", "few_shot_base", or "few_shot_role_based"
        cot: whether to add CoT to the prompt
    Returns:
        The formatted prompt for the tweet
    """
    if prompt_type in ["baseline", "role_based"]:
        prompt_template = prompts.prompts[prompt_type]
        formatted_prompt = prompts.format_prompt(prompt_template, tweet)
        if cot:
            formatted_prompt = prompts.add_cot_prompt(formatted_prompt, type = "base")
        return formatted_prompt
    elif prompt_type in ["few_shot_base", "few_shot_role_based"]:
        few_shot_prompts = prompts.get_few_shot_prompt(tweet)
        if prompt_type == "few_shot_base":
            prompt = few_shot_prompts["few_shot_base"]
        elif prompt_type == "few_shot_role_based":
            prompt = few_shot_prompts["few_shot_role_based"]
        if cot:
            prompt = prompts.add_cot_prompt(prompt, type = "few_shot")
        return prompt
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
def get_model(model_name, device, max_length, max_new_tokens, adapter_path):
    """
    Get the model to use for inference
    Args:
        model_name: the name of the model to use, default is google/flan-t5-large
        device: the device to use for the model, default is cuda if available, otherwise mps or cpu
        max_length: the maximum length of the input text
        max_new_tokens: the maximum number of new tokens to generate, default is 10
        adapter_path: the path to the adapter model if finetuned with LoRA
    Returns:
        A StanceClassifier object for inference
    """
    if adapter_path is not None:
        return StanceClassifier(model_name = model_name, device = device, max_length = max_length, max_new_tokens = max_new_tokens, adapter_path = adapter_path)
    else:
        return StanceClassifier(model_name = model_name, device = device, max_length = max_length, max_new_tokens = max_new_tokens, adapter_path = None)

def main():
    # get parser
    parser = get_parser()
    args = parser.parse_args()

    # load data
    df = pd.read_csv(args.input_file)

    # get model
    model = get_model(args.model_name, args.device, args.max_length, args.max_new_tokens, args.adapter_path)

    # extract tweets and get the predictions
    tweets = df['tweet'].tolist()
    predictions = []

    # get the predictions for each batch, faster with batching
    for i in tqdm(range(0, len(tweets), args.batch_size), desc="Predicting"):
        batch_tweets = tweets[i:i+args.batch_size]
        batch_prompts = [get_prompt(tweet, args.prompt_type, args.cot) for tweet in batch_tweets]
        batch_predictions = model.predict_batch(batch_prompts, batch_size = args.batch_size)
        predictions.extend(batch_predictions)

    # map the predictions back to the label names in the original dataset
    mapping = {'in-favor': 'FAVOR', 'against': 'AGAINST', 'neutral-or-unclear': 'NONE'}
    df['prediction'] = [mapping[pred] for pred in predictions]

    # save the predictions to a csv file
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
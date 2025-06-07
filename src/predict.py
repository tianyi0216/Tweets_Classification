# A script that given a input file, add a new column with the predictions

import pandas as pd
from model import StanceClassifier
import prompts
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["baseline", "role_based", "few_shot_base", "few_shot_role_based"], default="baseline")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
    return parser

def get_prompt(tweet, prompt_type, cot = False):
    """
    Get the appropriate prompt for the given tweet based on the prompt type and CoT option
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

def main():
    parser = get_parser()
    args = parser.parse_args()

    # load the data
    df = pd.read_csv(args.input_file)
    df = df.dropna(subset=['tweet', 'label_majority'])
    df = df.drop_duplicates(subset=['tweet'], keep='first')

    model = StanceClassifier(model_name = args.model_name)
    tweets = df['tweet'].tolist()
    predictions = []

    for i in tqdm(range(0, len(tweets), args.batch_size), desc="Predicting"):
        batch_tweets = tweets[i:i+args.batch_size]
        batch_prompts = [get_prompt(tweet, args.prompt_type, args.cot) for tweet in batch_tweets]
        batch_predictions = model.predict_batch(batch_prompts, batch_size = args.batch_size)
        predictions.extend(batch_predictions)

    mapping = {'in-favor': 'FAVOR', 'against': 'AGAINST', 'neutral-or-unclear': 'NONE'}
    df['prediction'] = [mapping[pred] for pred in predictions]
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
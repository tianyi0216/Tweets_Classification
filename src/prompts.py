# Different prompts types for the tweet classification task
import pandas as pd

prompts = {
    # Baseline prompt
    "baseline": 'What is the stance of the following tweet with respect to COVID-19 vaccine? Here is the tweet: "{tweet}". Please use exactly one word from the following 3 categories to label it: "in-favor", "against", "neutral-or-unclear".',
    # Role-based prompt
    "role_based": 'You are a senior public health expert analyst who classifies social media posts about COVID-19 vaccination. For a given tweet, decide whether the author is in-favor, against, or neutral-or-unclear of the vaccine. Here is the tweet: "{tweet}". Please use exactly one word from the following 3 categories to label it: "in-favor", "against", "neutral-or-unclear".'
}

def get_example_from_df(data):
    """
    Get one example from each stance in the dataset for few-shot prompt
    Args:
        data: the dataframe with column "label_majority"
    Returns:
        A dictionary with the each stance as the key and the corresponding example as the value
    """
    result = {}

    # against stance
    against_df = data[data['label_majority'] == 'AGAINST']
    if len(against_df) > 0:
        result['AGAINST'] = against_df.sample(1)

    # favor stance
    favor_df = data[data['label_majority'] == 'FAVOR']
    if len(favor_df) > 0:
        result['FAVOR'] = favor_df.sample(1)

    # neutral-or-unclear stance
    none_df = data[data['label_majority'] == 'NONE']
    if len(none_df) > 0:
        result['NONE'] = none_df.sample(1)

    return result

def get_few_shot_prompt(tweet):
    """
    Get a few-shot prompt for the tweet
    """

    df = pd.read_csv("data/Q2_20230202_majority.csv")
    examples = get_example_from_df(df)

    return {
        "few_shot_base": f"""What is the stance of the following tweet with respect to COVID-19 vaccine? Answer with exactly one word from the following 3 categories: "in-favor", "against", "neutral-or-unclear".
        
        Q: {examples['AGAINST']['tweet'].values[0]}
        A: against

        Q: {examples['FAVOR']['tweet'].values[0]}
        A: in-favor

        Q: {examples['NONE']['tweet'].values[0]}
        A: neutral-or-unclear

        Q: {tweet}
        A: 
        """,

        "few_shot_role_based": f"""You are a senior public health expert analyst who classifies social media posts stating their stance about COVID-19 vaccination. Answer with exactly one word from the following 3 categories: "in-favor", "against", "neutral-or-unclear".

        Q: {examples['AGAINST']['tweet'].values[0]}
        A: against

        Q: {examples['FAVOR']['tweet'].values[0]}
        A: in-favor

        Q: {examples['NONE']['tweet'].values[0]}
        A: neutral-or-unclear

        Q: {tweet}
        A: 
        """
    }

def format_prompt(prompt, tweet):
    """
    Add the tweet to the prompt
    Args:
        prompt: the prompt to add the tweet to, should have a "{tweet}" placeholder
        tweet: the tweet to add to the prompt, should be a string
    Returns:
        The prompt with the tweet added and ready to be used
    """
    return prompt.format(tweet=tweet)

def add_cot_prompt(prompt, type = "base"):
    """
    Make a prompt with CoT added
    Args:
        prompt: the prompt to add the CoT to, should already be formatted with the tweet
        type: the type of prompt, either "base" or "few_shot"
    Returns:
        The prompt with the CoT added
    """
    if type == "base":
        # add cot to the last sentence of the prompt
        prompt_list = prompt.split(".")
        prompt_list[-1] = 'Think step by step, and then use exactly one word from the following 3 categories to label it: "in-favor", "against", "neutral-or-unclear".'
        return ".".join(prompt_list)
    elif type == "few_shot":
        # add cot to the first sentence before the newline
        prompt_list = prompt.split("\n")
        prompt_list_0_split = prompt_list[0].split(".")
        prompt_list_0_split[-1] = 'Think step by step, and then use exactly one word from the following 3 categories to label it: "in-favor", "against", "neutral-or-unclear".'
        prompt_list[0] = ".".join(prompt_list_0_split)
        return "\n".join(prompt_list)
    else:
        raise ValueError(f"Invalid type: {type}")
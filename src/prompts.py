# Prompts for the tweet classification task
import pandas as pd

prompts = {
    # Baseline prompt
    "baseline": 'What is the stance of the following tweet with respect to COVID-19 vaccine? Here is the tweet: "{tweet}" Please use exactly one word from the following 3 categories to label it: "in-favor", "against", "neutral-or-unclear".',
    # Role-based prompt
    "Role-based": 'You are a senior public health expert analyst who classifies social media posts about  COVID-19 vaccination. For a given tweet, decide whether the author is in-favor, against, or neutral-or-unclear of the vaccine. Here is the tweet: "{tweet}" Please use exactly one word from the following 3 categories to label it: "in-favor", "against", "neutral-or-unclear".'
}

def get_example_from_df(data, n_examples=1):
    """
    Get some examples from each stance in the dataframe for few-shot learning
    """
    result = {}

    # against stance
    against_df = data[data['label_majority'] == 'AGAINST']
    if len(against_df) > 0:
        result['AGAINST'] = against_df.sample(n_examples)

    # favor stance
    favor_df = data[data['label_majority'] == 'FAVOR']
    if len(favor_df) > 0:
        result['FAVOR'] = favor_df.sample(n_examples)

    # neutral-or-unclear stance
    none_df = data[data['label_majority'] == 'NONE']
    if len(none_df) > 0:
        result['NONE'] = none_df.sample(n_examples)

    return result

def get_few_shot_prompt(tweet):
    """
    Get a few-shot prompt for the tweet
    """

    df = pd.read_csv("data/Q2_20230202_majority.csv")
    examples = get_example_from_df(df, n_examples=1)

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

        "few_shot_role_based": f"""You are a senior public health expert analyst who classifies social media posts about  COVID-19 vaccination. For a given tweet, answer with exactly one word from the following 3 categories: "in-favor", "against", "neutral-or-unclear".

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
    Format the prompt for the tweet
    """
    return prompt.format(tweet=tweet)
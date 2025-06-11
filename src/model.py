# The model wrapper StanceClassifier for inference and finetuning
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

class StanceClassifier:
    def __init__(self, model_name = "google/flan-t5-large", max_new_tokens = 10, device = None, max_length = 512, adapter_path = None):
        """
        Initialize the stance classifier to classify the stance of a tweet about COVID-19 vaccine as in-favor, against, or neutral-or-unclear
        Args:
            model_name: hf model name or local path to the model, default is google/flan-t5-large
            max_new_tokens: maximum number of new tokens to generate, default is 10
            device: device to use for inference, default is cuda if available, otherwise cpu
            max_length: maximum length of the input text
            adapter_path: local path to the adapter model if finetuned with LoRA
        """
        # set device, if not specified use the most powerful device available
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        # load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # if provided path to LoRA adapter, load it
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        # move the model to the device
        self.model.to(self.device)
        self.model.eval() # set model to eval mode for generation
        print(f"Model loaded successfully")

    def clean_prediction(self, raw_response):
        """
        Clean the prediction in case model doesn't output the instructed format of "in-favor", "against", or "neutral-or-unclear"
        Args:
            raw_response: the raw response from the model
        Returns:
            The cleaned response as a string from the model, one of "in-favor", "against", or "neutral-or-unclear"
        """
        resp = raw_response.lower().strip()
        # if matches directly with the expected format, return it
        if resp in ["in-favor", "against", "neutral-or-unclear"]:
            return resp
        
        # handle some variation of the expected format
        if "in-favor" in resp or "favor" in resp or "support" in resp:
            return "in-favor"
        elif "against" in resp or "oppose" in resp or "opposed" in resp:
            return "against"
        elif "neutral" in resp or "unclear" in resp or "none" in resp:
            return "neutral-or-unclear"
        else:
            # default to neutral-or-unclear if no match
            return "neutral-or-unclear" 

    def predict_one(self, prompt):
        """
        Generate a prediction for a single prompt
        Args:
            prompt: the prompt to generate a prediction for
        Returns:
            The prediction as a string from the model
        """
        # tokenize input and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        # generate output
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_prediction(response)

    def predict_batch(self, prompts, batch_size = 16):
        """
        Generate predictions per batch for a list of prompts (useful for large datasets)
        Args:
            prompts: a list of prompts to generate predictions for
            batch_size: the batch size for inference, default is 16
        Returns:
            A list of predictions as strings from the model
        """
        all_predictions = []
        # generate predictions per batch
        for i in range(0, len(prompts), batch_size):
            prompts_batch = prompts[i:i+batch_size]
            
            inputs = self.tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            
            pred_batch = []
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                response = self.clean_prediction(response)
                pred_batch.append(response)
            
            all_predictions.extend(pred_batch)
        
        return all_predictions
    
if __name__ == "__main__":
    # quick test for the model
    import prompts
    base_prompt = prompts.prompts["baseline"]
    model = StanceClassifier()
    print("Prediction:")
    print(model.predict_one(prompts.format_prompt(base_prompt, "I am in favor of the vaccine")).strip())
    print(model.predict_one(prompts.format_prompt(base_prompt, "I am against the vaccine")).strip())
    print(model.predict_one(prompts.format_prompt(base_prompt, "I am neutral about the vaccine")).strip())
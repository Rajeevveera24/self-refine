from transformers import LlamaModel, LlamaTokenizer, AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM

def load_model(model_name, token):
    # Load model and tokenizer with authentication token
    # model = LlamaModel.from_pretrained(model_name, use_auth_token=token)
    # tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    return model, tokenizer

def load_model_from_local(model_file_path):
    # Load model and tokenizer from local cache
    tokenizer = AutoTokenizer.from_pretrained(model_file_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_file_path, local_files_only=True)
    model = None
    return model, tokenizer

# Specify the model name and your Hugging Face access token
model_name = "reciprocate/llama2-7b-gsm8k"
your_token = "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"  # Replace with your actual token
model_file_path = "/home/raj/.cache/huggingface/hub/models--reciprocate--llama2-7b-gsm8k/snapshots/a99b9c5a7e7b6c37dc6fd81cbc4fd2f2015b2967"
model, tokenizer = load_model_from_local(model_file_path)

# You can now continue with the rest of your inference code

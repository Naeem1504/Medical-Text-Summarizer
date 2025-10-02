from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline  

# Replace with your Hugging Face repo ID
repo_id = "Naeem92/medtext-bart-finetuned"  

# If your model is private, put your HF token here
token = None

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(repo_id, use_auth_token=token)
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, use_auth_token=token)

# Create summarization pipeline (device=-1 for CPU, 0 for GPU)
pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

# Test with a dummy clinical note
text = "The patient was admitted with chest pain. ECG showed abnormalities and treatment with aspirin was initiated."
summary = pipe(text, max_length=50, min_length=15, do_sample=False)[0]['summary_text']

print("\n--- Original Text ---")
print(text)
print("\n--- Generated Summary ---")
print(summary)

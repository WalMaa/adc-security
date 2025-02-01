from transformers import pipeline

def call_transformer(prompt, model_id):
    generator = pipeline('text-generation', model=model_id, do_sample=True, max_new_tokens=100, min_new_tokens=20)
    return generator(prompt)


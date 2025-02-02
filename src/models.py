import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, pipeline

# Loading a pre-trained RAG model and its tokenizer.
model_name = "facebook/rag-token-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)

# Using the custom passages file for the retriever.
passages_path = "/path/to/custom_passages_file.json"

# using a custom retriever to integrate the RAG model with the contextual data.
retriever = RagRetriever.from_pretrained(
    model_name,
    index_name="custom",         # use a custom index name
    passages_path=passages_path,   # point to your own passages file
    index_path=None              # if you have a pre-built index, provide its path; otherwise, leave as None
)

model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)

# if GPU is available, move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# text-generation pipeline using the RAG model.
# This pipeline will internally use the retriever to fetch additional context.
rag_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device.type=="cuda" else -1)

def call_transformer(prompt):
    
    # Step 4: Generate a response.
    # Adjust generation parameters (e.g., num_beams, max_new_tokens, temperature) for better diversity and reasoning.
    return rag_generator(
    prompt,
    num_beams=3,           # Use beam search to improve output quality
    max_new_tokens=300,    # Set an upper bound for the generated response
    do_sample=True,        # Allow sampling for more diverse outputs
    temperature=0.7,       # Lower temperature for more coherent reasoning
    )


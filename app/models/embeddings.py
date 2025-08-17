from sentence_transformers import SentenceTransformer

def generate_embeddings(text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text)
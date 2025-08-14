import numpy as np
from app.services.embeddings import embed_texts

def test_embed_shape():
    emb = embed_texts(["hello", "world"])
    assert emb.shape[0] == 2
    assert emb.shape[1] > 0
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

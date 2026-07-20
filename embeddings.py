from fastembed import TextEmbedding
import numpy as np


# ------------------------------------------------------------
# FIX: replaced sentence-transformers (torch backend) with
# fastembed (ONNX Runtime backend). torch + transformers alone
# commonly use 400-600MB of RAM once loaded, which is enough by
# itself to exceed Render's 512MB free-tier limit. fastembed has
# no torch dependency and uses a much smaller memory footprint.
#
# This wrapper class keeps the SAME .encode() interface that
# sentence-transformers had, so nothing else in the codebase
# (app.py, vector_store.py) needs to change.
# ------------------------------------------------------------
class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # fastembed loads the model lazily on first .embed() call,
        # so importing this module stays cheap.
        self._model = TextEmbedding(model_name=model_name)

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = list(self._model.embed(texts))
        embeddings = np.array(embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # avoid divide-by-zero
            embeddings = embeddings / norms

        return embeddings


# Load model only once, at import time (Python caches this - it
# will not reload on every Streamlit rerun since module-level code
# only executes once per process).
model = EmbeddingModel()


def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts.

    Parameters:
        texts (list): List of sentences/chunks

    Returns:
        embeddings (numpy array)
    """
    return model.encode(texts, normalize_embeddings=True)
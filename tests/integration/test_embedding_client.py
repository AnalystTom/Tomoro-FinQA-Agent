from app.clients.embedding_client import EmbeddingClient


def test_embedding_client_processes_text():
    client = EmbeddingClient(dim=8)
    vecs = client.embed(["hello", "world"])
    assert vecs.shape == (2, 8)

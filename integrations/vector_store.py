"""Dual vector store: Pinecone (managed cloud) + Qdrant (self-hosted/cloud)."""
import os
import uuid
from typing import Any


# ── Pinecone ──────────────────────────────────────────────────────────────────

def _pinecone_index(index_name: str = "grokadile-outputs"):
    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)


def upsert_pinecone(embedding: list[float], metadata: dict[str, Any]) -> str:
    vid = str(uuid.uuid4())
    _pinecone_index().upsert(vectors=[{"id": vid, "values": embedding, "metadata": metadata}])
    return vid


def query_pinecone(embedding: list[float], top_k: int = 5) -> list[dict]:
    res = _pinecone_index().query(vector=embedding, top_k=top_k, include_metadata=True)
    return res.matches


# ── Qdrant ────────────────────────────────────────────────────────────────────

_QDRANT_COLLECTION = "grokadile"
_QDRANT_DIM = 768


def _qdrant_client():
    from qdrant_client import QdrantClient  # type: ignore

    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def _ensure_qdrant_collection() -> None:
    from qdrant_client.models import Distance, VectorParams  # type: ignore

    client = _qdrant_client()
    names = [c.name for c in client.get_collections().collections]
    if _QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=_QDRANT_COLLECTION,
            vectors_config=VectorParams(size=_QDRANT_DIM, distance=Distance.COSINE),
        )


def upsert_qdrant(embedding: list[float], payload: dict[str, Any]) -> str:
    from qdrant_client.models import PointStruct  # type: ignore

    _ensure_qdrant_collection()
    pid = str(uuid.uuid4())
    _qdrant_client().upsert(
        collection_name=_QDRANT_COLLECTION,
        points=[PointStruct(id=pid, vector=embedding, payload=payload)],
    )
    return pid


def search_qdrant(embedding: list[float], top_k: int = 5) -> list:
    _ensure_qdrant_collection()
    return _qdrant_client().search(
        collection_name=_QDRANT_COLLECTION,
        query_vector=embedding,
        limit=top_k,
    )

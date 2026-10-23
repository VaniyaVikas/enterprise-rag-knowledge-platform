
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Chunk text must be a string.")

        if not isinstance(self.source, str):
            raise TypeError("Chunk source must be a string.")

        if isinstance(self.chunk_id, bool) or not isinstance(self.chunk_id, int):
            raise TypeError("chunk_id must be an integer.")

        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty.")

        if not self.source.strip():
            raise ValueError("Chunk source cannot be empty.")

        if self.chunk_id < 0:
            raise ValueError("chunk_id cannot be negative.")

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "chunk_size": len(self.text),
        }

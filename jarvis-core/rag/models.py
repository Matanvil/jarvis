from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    file: str
    start_line: int
    score: float = 0.0
    chunk_type: str = "block"  # "block" | "function" | "class"

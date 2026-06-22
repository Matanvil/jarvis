import json
import re
from pathlib import Path
from typing import List, Iterator
from rag.models import Chunk

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".swift",
    ".md", ".txt", ".json", ".yaml", ".yml",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", "dist", "build", ".chroma", "rag_store",
}
CHUNK_SIZE = 50

# tree-sitter node types that represent top-level definitions per extension
_TS_NODE_TYPES = {
    ".py":  {"function_definition", "class_definition", "decorated_definition"},
    ".js":  {"function_declaration", "class_declaration", "export_statement", "lexical_declaration"},
    ".jsx": {"function_declaration", "class_declaration", "export_statement", "lexical_declaration"},
    ".ts":  {"function_declaration", "class_declaration", "export_statement", "lexical_declaration",
             "interface_declaration", "type_alias_declaration"},
    ".tsx": {"function_declaration", "class_declaration", "export_statement", "lexical_declaration",
             "interface_declaration", "type_alias_declaration"},
    ".swift": {"function_declaration", "class_declaration", "struct_declaration",
               "extension_declaration", "protocol_declaration", "enum_declaration"},
}


def _load_ts_language(ext: str):
    """Lazy-load tree-sitter Language for a file extension. Returns None on ImportError."""
    try:
        from tree_sitter import Language
        if ext == ".py":
            import tree_sitter_python as m
            return Language(m.language())
        elif ext in (".js", ".jsx"):
            import tree_sitter_javascript as m
            return Language(m.language())
        elif ext == ".ts":
            import tree_sitter_typescript as m
            return Language(m.language_typescript())
        elif ext == ".tsx":
            import tree_sitter_typescript as m
            return Language(m.language_tsx())
        elif ext == ".swift":
            import tree_sitter_swift as m
            return Language(m.language())
    except (ImportError, AttributeError):
        return None
    return None


def scan_files(repo_path: str) -> Iterator[Path]:
    root = Path(repo_path)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def chunk_file_naive(file_path: Path, repo_root: str, chunk_size: int = CHUNK_SIZE) -> List[Chunk]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = text.splitlines()
    relative_path = str(file_path.relative_to(repo_root))
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_text = "\n".join(lines[i: i + chunk_size]).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, file=relative_path, start_line=i + 1))
    return chunks


def _chunk_markdown(file_path: Path, repo_root: str) -> List[Chunk]:
    """Split Markdown by ATX header lines (# through ######). Content before first header = one chunk."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return chunk_file_naive(file_path, repo_root)
    lines = text.splitlines()
    relative_path = str(file_path.relative_to(repo_root))
    header_re = re.compile(r'^#{1,6}\s+')
    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_start = 1
    for i, line in enumerate(lines, 1):
        if header_re.match(line) and current_lines:
            chunk_text = "\n".join(current_lines).strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, file=relative_path, start_line=current_start))
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)
    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, file=relative_path, start_line=current_start))
    return chunks or chunk_file_naive(file_path, repo_root)


def _chunk_json(file_path: Path, repo_root: str) -> List[Chunk]:
    """Split JSON objects by top-level keys. Falls back to naive for arrays or invalid JSON."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        data = json.loads(text)
    except Exception:
        return chunk_file_naive(file_path, repo_root)
    if not isinstance(data, dict):
        return chunk_file_naive(file_path, repo_root)
    relative_path = str(file_path.relative_to(repo_root))
    chunks = []
    for i, (key, value) in enumerate(data.items()):
        chunk_text = json.dumps({key: value}, indent=2)
        chunks.append(Chunk(text=chunk_text, file=relative_path, start_line=i + 1))
    return chunks or chunk_file_naive(file_path, repo_root)


def _chunk_tree_sitter(file_path: Path, repo_root: str, ext: str) -> List[Chunk]:
    """Use tree-sitter to split by top-level function/class definitions.
    Falls back to naive chunking if the grammar is unavailable or the file fails to parse."""
    language = _load_ts_language(ext)
    if language is None:
        return chunk_file_naive(file_path, repo_root)
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return chunk_file_naive(file_path, repo_root)
    try:
        from tree_sitter import Parser
        parser = Parser(language)
        tree = parser.parse(bytes(source, "utf-8"))
    except Exception:
        return chunk_file_naive(file_path, repo_root)

    target_types = _TS_NODE_TYPES.get(ext, set())
    lines = source.splitlines()
    relative_path = str(file_path.relative_to(repo_root))
    chunks: List[Chunk] = []

    for node in tree.root_node.children:
        if node.type not in target_types:
            continue
        sp = node.start_point
        ep = node.end_point
        start_line = sp.row if hasattr(sp, "row") else sp[0]   # 0-indexed
        end_line = ep.row if hasattr(ep, "row") else ep[0]     # 0-indexed
        chunk_text = "\n".join(lines[start_line: end_line + 1]).strip()
        if not chunk_text:
            continue
        if any(k in node.type for k in ("class", "struct", "extension", "protocol", "enum", "interface")):
            chunk_type = "class"
        elif any(k in node.type for k in ("function", "method")):
            chunk_type = "function"
        else:
            chunk_type = "block"
        chunks.append(Chunk(
            text=chunk_text,
            file=relative_path,
            start_line=start_line + 1,  # convert to 1-indexed
            chunk_type=chunk_type,
        ))

    return chunks or chunk_file_naive(file_path, repo_root)


def chunk_file_semantic(file_path: Path, repo_root: str) -> List[Chunk]:
    """Dispatch to the right semantic chunker for the file type."""
    ext = file_path.suffix.lower()
    if ext == ".md":
        return _chunk_markdown(file_path, repo_root)
    elif ext == ".json":
        return _chunk_json(file_path, repo_root)
    elif ext in _TS_NODE_TYPES:
        return _chunk_tree_sitter(file_path, repo_root, ext)
    else:
        return chunk_file_naive(file_path, repo_root)


def index_repo(repo_path: str, embedder, store, chunk_size: int = CHUNK_SIZE, use_semantic: bool = True) -> int:
    """Index all files in repo_path into store. Clears existing index first. Returns total chunk count."""
    store.clear()
    total = 0
    for file_path in scan_files(repo_path):
        chunks = chunk_file_semantic(file_path, repo_path) if use_semantic else chunk_file_naive(file_path, repo_path, chunk_size)
        if not chunks:
            continue
        embeddings = [embedder.embed(c.text) for c in chunks]
        store.add(chunks, embeddings)
        total += len(chunks)
    return total

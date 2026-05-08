"""
One-off script: translate Chinese in # comments and docstrings to English.
Does not translate Chinese inside regular code strings (e.g. format templates).
"""
from __future__ import annotations

import ast
import sys
import time
import tokenize
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parent
translator = GoogleTranslator(source="zh-CN", target="en")


def has_chinese(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def translate(text: str) -> str:
    text = text.strip()
    if not text or not has_chinese(text):
        return text
    try:
        time.sleep(0.12)
        out = translator.translate(text)
        return out if out else text
    except Exception as e:
        print(f"  translate error: {e!r} for {text[:60]!r}...", file=sys.stderr)
        return text


def line_starts(source: str) -> list[int]:
    """Byte/char index at start of each 1-based line."""
    starts = [0]
    i = 0
    while i < len(source):
        if source[i] == "\n":
            starts.append(i + 1)
        i += 1
    return starts


def pos_to_index(line_starts: list[int], lineno: int, col: int) -> int:
    return line_starts[lineno - 1] + col


def collect_docstring_spans(source: str) -> list[tuple[int, int, str]]:
    """(start_char, end_char, original_text) for docstring nodes only."""
    spans: list[tuple[int, int, str]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  skip ast (syntax): {e}", file=sys.stderr)
        return spans

    def add_doc(node: ast.AST) -> None:
        if not node.body:
            return
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            return
        v = first.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            s = v.value
        elif isinstance(v, ast.Str):  # py<3.8 compat
            s = v.s
        else:
            return
        if not has_chinese(s):
            return
        if hasattr(first, "end_col_offset") and first.end_col_offset is not None:
            ls = line_starts(source)
            start = pos_to_index(ls, first.lineno, first.col_offset)
            end = pos_to_index(ls, first.end_lineno, first.end_col_offset)
            spans.append((start, end, source[start:end]))
        else:
            pass

    add_doc(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_doc(n)
    return spans


def translate_comments_tokenize(source: str) -> str:
    """Replace # ... comment bodies that contain Chinese."""
    lines = source.splitlines(keepends=True)

    def readline() -> str:
        readline.i += 1
        if readline.i <= len(lines):
            return lines[readline.i - 1]
        return ""

    readline.i = 0
    out_parts: list[str] = []
    prev_end_line, prev_end_col = 1, 0

    try:
        tokens = list(tokenize.generate_tokens(readline))
    except tokenize.TokenError:
        return source

    ls = line_starts(source)

    for tok in tokens:
        if tok.type in (tokenize.ENDMARKER,):
            continue
        if tok.start[0] < prev_end_line or (
            tok.start[0] == prev_end_line and tok.start[1] < prev_end_col
        ):
            continue
        # gap before token
        gap_start = pos_to_index(ls, prev_end_line, prev_end_col)
        gap_end = pos_to_index(ls, tok.start[0], tok.start[1])
        out_parts.append(source[gap_start:gap_end])

        if tok.type == tokenize.COMMENT and has_chinese(tok.string):
            # tok.string includes '#'
            if tok.string.startswith("#"):
                prefix = "#"
                body = tok.string[1:]
                if has_chinese(body):
                    tr = translate(body)
                    new_comment = prefix + (" " + tr if tr and not tr.startswith(" ") else tr)
                else:
                    new_comment = tok.string
            else:
                new_comment = tok.string
            out_parts.append(new_comment)
        else:
            out_parts.append(
                source[pos_to_index(ls, tok.start[0], tok.start[1]) : pos_to_index(
                    ls, tok.end[0], tok.end[1]
                )]
            )
        prev_end_line, prev_end_col = tok.end

    tail_start = pos_to_index(ls, prev_end_line, prev_end_col)
    out_parts.append(source[tail_start:])
    return "".join(out_parts)


def replace_spans(source: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace source[start:end] with English docstring preserving quotes."""
    if not spans:
        return source
    spans = sorted(spans, key=lambda x: x[0], reverse=True)
    s = source
    for start, end, orig in spans:
        quote = orig[:3] if orig[:3] in ('"""', "'''") else (orig[0] if orig[0] in "'\"" else '"""')
        if orig.startswith('"""') or orig.startswith("'''"):
            q = orig[:3]
            inner = orig[3:-3]
            closing = orig[-3:]
            tr = translate(inner)
            new_block = q + tr + closing
        elif orig.startswith('"') or orig.startswith("'"):
            qchar = orig[0]
            inner = orig[1:-1]
            tr = translate(inner)
            new_block = qchar + tr.replace(qchar, "\\" + qchar) + qchar
        else:
            new_block = orig
        s = s[:start] + new_block + s[end:]
    return s


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # CRLF can desync tokenizer line indices vs ast line numbers
    text = text.replace("\r\n", "\n")
    if not has_chinese(text):
        return False
    orig = text
    spans = collect_docstring_spans(text)
    text = replace_spans(text, spans)
    text = translate_comments_tokenize(text)
    if text != orig:
        path.write_text(text, encoding="utf-8", newline="")
        return True
    return False


SKIP_DIR_NAMES = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules"})


def discover_py_files_with_chinese() -> list[Path]:
    import subprocess

    try:
        r = subprocess.run(
            ["rg", "-l", r"[\p{Han}]", "--glob", "*.py", str(ROOT)],
            capture_output=True,
            text=True,
        )
        if r.returncode in (0, 1) and r.stdout.strip():
            return sorted(Path(line.strip()) for line in r.stdout.splitlines() if line.strip())
    except FileNotFoundError:
        pass
    out: list[Path] = []
    for p in ROOT.rglob("*.py"):
        if p.name == "translate_zh_comments.py":
            continue
        if SKIP_DIR_NAMES.intersection(p.parts):
            continue
        try:
            if has_chinese(p.read_text(encoding="utf-8", errors="ignore")):
                out.append(p)
        except OSError:
            continue
    return sorted(out)


def main() -> None:
    files = sorted(
        {Path(p) for p in sys.argv[1:]} if len(sys.argv) > 1 else []
    )
    if not files:
        files = discover_py_files_with_chinese()

    n = 0
    for p in files:
        if not p.exists():
            continue
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        try:
            if process_file(p):
                print(f"updated: {rel}")
                n += 1
        except Exception as e:
            print(f"ERROR {rel}: {e}", file=sys.stderr)
    print(f"Done. {n} files updated.")


if __name__ == "__main__":
    main()

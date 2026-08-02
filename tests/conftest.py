"""Test isolation and fixture generation.

TOKEN_DISTILLER_HOME must be set before token_distiller.config is imported, because
config binds HOME/DB_PATH at import time and storage imports DB_PATH from it. Setting it
here at conftest top level guarantees that ordering: pytest loads conftest before any
test module, and no token_distiller import happens above this line.
"""

import os
import tempfile

_TEST_HOME = tempfile.mkdtemp(prefix="token-distiller-tests-")
os.environ["TOKEN_DISTILLER_HOME"] = _TEST_HOME

import pytest  # noqa: E402


def _pdf_content_stream(lines: list[str]) -> bytes:
    ops = []
    y = 720
    for line in lines:
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"BT /F1 12 Tf 72 {y} Td ({safe}) Tj ET")
        y -= 20
    return "\n".join(ops).encode("latin-1")


def make_pdf(path, pages: list[list[str]]) -> str:
    """Minimal multi-page text-native PDF. Hand-built rather than pulled from a binary
    fixture so page content is explicit and diffable in the tests that use it."""
    objects: list[bytes] = [b"", b"", b""]  # 1=catalog, 2=pages, 3=font (filled below)
    kids = []
    for page_lines in pages:
        stream = _pdf_content_stream(page_lines)
        page_obj_num = len(objects) + 1
        content_obj_num = page_obj_num + 1
        kids.append(page_obj_num)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_obj_num} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )

    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids_ref = " ".join(f"{k} 0 R" for k in kids)
    objects[1] = f"<< /Type /Pages /Kids [{kids_ref}] /Count {len(kids)} >>".encode()
    objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += b"trailer\n" + f"<< /Size {n} /Root 1 0 R >>\n".encode()
    out += b"startxref\n" + f"{xref}\n".encode() + b"%%EOF"

    path = str(path)
    with open(path, "wb") as f:
        f.write(out)
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    counter = {"n": 0}

    def _make(pages: list[list[str]], name: str | None = None) -> str:
        counter["n"] += 1
        target = tmp_path / (name or f"doc{counter['n']}.pdf")
        return make_pdf(target, pages)

    return _make


@pytest.fixture(autouse=True)
def no_accidental_ocr(request, monkeypatch):
    """Fail fast if a test drifts onto the OCR path.

    Tesseract can take tens of seconds per page, so the suite deliberately exercises
    logic through the native-text PDF path. A fixture whose page text falls under
    MIN_NATIVE_TEXT_CHARS silently rasterizes instead, which turns a millisecond test
    into a minute-long one. Mark a test with @pytest.mark.ocr to opt in.
    """
    if request.node.get_closest_marker("ocr"):
        return

    def _refuse(*_args, **_kwargs):
        raise AssertionError(
            "test invoked OCR unintentionally — page text is probably shorter than "
            "MIN_NATIVE_TEXT_CHARS, so the page rasterized instead of using its text layer"
        )

    monkeypatch.setattr("token_distiller.ocr.ocr_image", _refuse)


@pytest.fixture(autouse=True)
def clean_db():
    """Each test starts with an empty database — the cache and session-read tables are
    stateful and would otherwise leak hits across tests."""
    from token_distiller.config import DB_PATH

    if DB_PATH.exists():
        DB_PATH.unlink()
    yield
    if DB_PATH.exists():
        DB_PATH.unlink()

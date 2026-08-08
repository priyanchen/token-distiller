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


def make_pdf_with_images(path, pages: list[list[str]], image_pages: set[int]) -> str:
    """Same minimal hand-built PDF as make_pdf, but pages listed in image_pages also
    get a tiny flat-color embedded image XObject (uncompressed DeviceRGB, no filter --
    the simplest image pdfplumber's page.images can still detect). Used to test
    PageResult.image_count / pages_with_uncaptured_images without pulling in a binary
    fixture file."""
    img_w, img_h = 4, 4
    img_data = bytes([180, 40, 40] * (img_w * img_h))

    objects: list[bytes] = [b"", b"", b""]  # 1=catalog, 2=pages, 3=font
    kids = []
    for page_num, page_lines in enumerate(pages):
        content_ops = []
        y = 720
        for line in page_lines:
            safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            content_ops.append(f"BT /F1 12 Tf 72 {y} Td ({safe}) Tj ET")
            y -= 20

        page_obj_num = len(objects) + 1
        content_obj_num = page_obj_num + 1
        kids.append(page_obj_num)

        resources = "/Font << /F1 3 0 R >>"
        if page_num in image_pages:
            content_ops.append(f"q {img_w * 20} 0 0 {img_h * 20} 300 100 cm /Im1 Do Q")
            img_obj_num = content_obj_num + 1
            resources += f" /XObject << /Im1 {img_obj_num} 0 R >>"

        stream = "\n".join(content_ops).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /Resources << {resources} >> "
            f"/MediaBox [0 0 612 792] /Contents {content_obj_num} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
        if page_num in image_pages:
            objects.append(
                f"<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {len(img_data)} >>"
                f"\nstream\n".encode()
                + img_data
                + b"\nendstream"
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


def make_pdf_with_legible_figure(path, page_text: str, figure_lines: list[str]) -> str:
    """A PDF whose embedded figure contains real rendered text, so the OCR/vision figure
    path has something to actually recover. The flat-colour image in
    make_pdf_with_images is enough to test detection, but not extraction."""
    from PIL import Image, ImageDraw, ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    font_file = next((c for c in candidates if os.path.isfile(c)), None)
    if font_file is None:
        pytest.skip("no usable TrueType font for rendering a legible figure")

    img_w, img_h = 480, 220
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_file, 30)
    y = 20
    for line in figure_lines:
        draw.text((18, y), line, fill="black", font=font)
        y += 42
    raw = img.tobytes()  # uncompressed DeviceRGB, no filter — simplest valid XObject

    content_ops = [f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET", "q 342 0 0 157 40 400 cm /Im1 Do Q"]
    stream = "\n".join(content_ops).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 3 0 R >> "
        b"/XObject << /Im1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        f"<< /Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} "
        f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Length {len(raw)} >>\nstream\n".encode()
        + raw
        + b"\nendstream",
    ]

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
def pdf_with_images_factory(tmp_path):
    counter = {"n": 0}

    def _make(pages: list[list[str]], image_pages: set[int], name: str | None = None) -> str:
        counter["n"] += 1
        target = tmp_path / (name or f"docimg{counter['n']}.pdf")
        return make_pdf_with_images(target, pages, image_pages)

    return _make


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

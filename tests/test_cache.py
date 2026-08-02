from token_distiller import cache
from token_distiller.models import DistillMethod, DistillResult, PageResult


def _result(path="/tmp/x.pdf", text="hello world", boilerplate=None):
    return DistillResult(
        source_path=path,
        source_type="pdf",
        pages=[
            PageResult(
                page_index=0,
                method=DistillMethod.NATIVE_TEXT,
                text=text,
                raw_tokens_est=100,
                distilled_tokens_est=3,
            )
        ],
        duration_ms=7,
        boilerplate=boilerplate or [],
    )


def test_hash_is_stable_for_identical_bytes(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert cache.content_hash(str(a)) == cache.content_hash(str(b))


def test_hash_changes_when_bytes_change(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"version one")
    first = cache.content_hash(str(f))
    f.write_bytes(b"version two")
    assert cache.content_hash(str(f)) != first


def test_miss_returns_none():
    assert cache.get("no-such-hash") is None


def test_put_then_get_round_trips_text():
    handle = cache.put("h1", _result(text="round trip me"))
    got_handle, restored = cache.get("h1")
    assert got_handle == handle
    assert restored.text == "round trip me"


def test_round_trip_preserves_page_metadata():
    cache.put("h2", _result())
    _, restored = cache.get("h2")
    page = restored.pages[0]
    assert page.method is DistillMethod.NATIVE_TEXT
    assert page.raw_tokens_est == 100
    assert page.distilled_tokens_est == 3


def test_round_trip_preserves_boilerplate_manifest():
    """Without this the manifest is lost on a cache hit and the collapsed lines really
    would become unrecoverable."""
    manifest = [{"line": "(c) ACME", "occurrences": 12}]
    cache.put("h3", _result(boilerplate=manifest))
    _, restored = cache.get("h3")
    assert restored.boilerplate == manifest
    assert "(c) ACME" in restored.rendered_text


def test_get_by_id_matches_get_by_hash():
    handle = cache.put("h4", _result(text="by id"))
    assert cache.get_by_id(handle).text == "by id"


def test_get_by_id_unknown_handle_is_none():
    assert cache.get_by_id(999999) is None


def test_reput_same_hash_keeps_one_row():
    cache.put("h5", _result(text="first"))
    cache.put("h5", _result(text="second"))
    entries = [e for e in cache.list_entries() if e["source_path"] == "/tmp/x.pdf"]
    assert len(entries) == 1
    assert cache.get("h5")[1].text == "second"


def test_session_read_tracking():
    assert cache.seen_in_session("S1", "hashA") is False
    cache.mark_seen("S1", "hashA")
    assert cache.seen_in_session("S1", "hashA") is True


def test_sessions_do_not_share_read_state():
    cache.mark_seen("S1", "hashB")
    assert cache.seen_in_session("S2", "hashB") is False


def test_different_content_is_never_seen_in_session():
    cache.mark_seen("S1", "hashC")
    assert cache.seen_in_session("S1", "hashC-modified") is False


def test_missing_session_id_is_never_tracked():
    """CLI invocations have no session; they must not be able to poison collapse state."""
    cache.mark_seen(None, "hashD")
    assert cache.seen_in_session(None, "hashD") is False


def test_list_entries_is_newest_first():
    cache.put("hx", _result(path="/tmp/older.pdf"))
    cache.put("hy", _result(path="/tmp/newer.pdf"))
    assert cache.list_entries()[0]["source_path"] == "/tmp/newer.pdf"

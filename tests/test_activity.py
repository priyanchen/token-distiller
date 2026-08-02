from token_distiller.activity import _bucket, classify, current_mode, record


def test_edits_bucket_as_edit():
    assert _bucket("Edit", {}) == "edit"
    assert _bucket("Write", {}) == "edit"


def test_reads_bucket_as_read():
    assert _bucket("Read", {}) == "read"
    assert _bucket("Grep", {}) == "read"


def test_test_runners_are_distinguished_from_other_bash():
    assert _bucket("Bash", {"command": "pytest tests/"}) == "bash_test"
    assert _bucket("Bash", {"command": "npm test"}) == "bash_test"


def test_git_commands_bucket_separately():
    assert _bucket("Bash", {"command": "git status"}) == "bash_git"


def test_other_bash_is_infra():
    assert _bucket("Bash", {"command": "ls -la /tmp"}) == "bash_infra"


def test_bash_without_command_does_not_crash():
    assert _bucket("Bash", {}) == "bash_infra"


def test_unknown_tools_bucket_as_other():
    assert _bucket("SomeFutureTool", {}) == "other"


def test_empty_history_is_general():
    assert classify([]) == "general"


def test_edits_plus_test_runs_is_debug():
    assert classify(["edit", "bash_test", "edit", "bash_test"]) == "debug"


def test_mostly_edits_is_code():
    assert classify(["edit", "edit", "edit", "read"]) == "code"


def test_reads_without_edits_is_review():
    assert classify(["read"] * 6) == "review"


def test_infra_commands_dominate_as_infra():
    assert classify(["bash_infra", "bash_infra", "bash_git", "bash_infra"]) == "infra"


def test_recorded_calls_drive_the_reported_mode():
    for _ in range(3):
        record("Edit", {}, session_id="S_ACT")
        record("Bash", {"command": "pytest"}, session_id="S_ACT")
    assert current_mode(session_id="S_ACT")["mode"] == "debug"


def test_sessions_are_tracked_independently():
    record("Edit", {}, session_id="S_ONE")
    record("Read", {}, session_id="S_TWO")
    assert current_mode(session_id="S_TWO")["bucket_counts"] == {"read": 1}

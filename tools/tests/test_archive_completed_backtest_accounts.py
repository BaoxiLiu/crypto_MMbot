import hashlib
import io
import json
import tarfile

import pytest

from tools.maintenance.archive_completed_backtest_accounts import COLLECTOR, archive_account, stream_files, verify_folder


def entry(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def packed(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, data in members:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    stream.seek(0)
    return stream


def test_bytes_only_no_metadata_copy(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise OSError("External filesystem metadata unsupported")
    monkeypatch.setattr("os.utime", forbidden)
    stream_files(packed([("./nested/data", b"hello")]), tmp_path,
                 {"nested/data": entry(b"hello")})
    assert (tmp_path / "nested/data").read_bytes() == b"hello"


@pytest.mark.parametrize("members,expected", [
    ([("../escaped", b"x")], {}),
    ([("/absolute", b"x")], {}),
    ([("data", b"x"), ("data", b"x")], {"data": entry(b"x")}),
    ([("data", b"changed")], {"data": entry(b"original")}),
    ([], {"missing": entry(b"x")}),
])
def test_unsafe_changed_duplicate_missing_refused(tmp_path, members, expected):
    with pytest.raises(ValueError):
        stream_files(packed(members), tmp_path, expected)


def test_symlink_refused(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    (tmp_path / "link").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        verify_folder(tmp_path, {})


def test_existing_exact_archive_reused_no_stream(tmp_path):
    folder = tmp_path / "coin--control"
    folder.mkdir()
    (folder / "summary.json").write_bytes(b"{}")
    account = dict(id=folder.name, summary_sha256=entry(b"{}")["sha256"],
                   files={"summary.json": entry(b"{}")})
    def forbidden(_):
        raise AssertionError("Existing exact archive should not copy again")
    result = archive_account(account, tmp_path, "frozen", forbidden)
    assert result["reused"]
    with pytest.raises(ValueError, match="registration identity"):
        archive_account(account, tmp_path, "changed", forbidden)


def test_existing_corrupted_archive_not_overwritten(tmp_path):
    folder = tmp_path / "coin--control"
    folder.mkdir()
    (folder / "data").write_bytes(b"wrong")
    account = dict(id=folder.name, summary_sha256="s", files={"data": entry(b"right")})
    with pytest.raises(ValueError):
        archive_account(account, tmp_path, "frozen", lambda _: None)
    assert (folder / "data").read_bytes() == b"wrong"


def test_only_valid_companion_appledouble_is_metadata(tmp_path):
    (tmp_path/'data').write_bytes(b'good')
    (tmp_path/'._data').write_bytes(bytes.fromhex('0005160700020000')+bytes(18))
    assert verify_folder(tmp_path,{'data':entry(b'good')})==['._data']
    (tmp_path/'._data').write_bytes(b'arbitrary unexpected content')
    with pytest.raises(ValueError):verify_folder(tmp_path,{'data':entry(b'good')})


def test_appledouble_without_expected_companion_is_not_ignored(tmp_path):
    (tmp_path/'._missing').write_bytes(bytes.fromhex('0005160700020000')+bytes(18))
    with pytest.raises(ValueError):verify_folder(tmp_path,{})


class Process:
    def __init__(self, stream, code=0):
        self.stdout = stream
        self.code = code
    def wait(self):
        return self.code
    def poll(self):
        return self.code


def test_new_archive_only_committed_after_full_verification(tmp_path):
    account = dict(id="coin--control", summary_sha256="s", files={"data": entry(b"good")})
    result = archive_account(account, tmp_path, "frozen",
                             lambda _: Process(packed([("data", b"good")])))
    assert not result["reused"]
    receipt = json.loads((tmp_path / account["id"] / "byte-archive-receipt.json").read_text())
    assert receipt["economic_adoption"] is False
    assert receipt["registration_sha256"] == "frozen"


def test_failed_copy_preserves_partial_without_complete_directory(tmp_path):
    account = dict(id="coin--control", summary_sha256="s", files={"data": entry(b"good")})
    with pytest.raises(ValueError):
        archive_account(account, tmp_path, "frozen",
                        lambda _: Process(packed([("data", b"bad")])))
    assert not (tmp_path / account["id"]).exists()
    assert len(list(tmp_path.glob(".coin--control.partial-*"))) == 1


def collector_fixture(tmp_path, monkeypatch):
    folder = tmp_path / "runs-v1" / "coin--control"
    folder.mkdir(parents=True)
    summary = dict(id="coin--control", status="complete-not-adoption",
                   economic_adoption=False, full_cost_adoption=False)
    raw = json.dumps(summary).encode()
    (folder / "summary.json").write_bytes(raw)
    contract = dict(tasks=[dict(id=folder.name, config=dict(output=str(folder)))])
    contract_raw = json.dumps(contract).encode()
    (tmp_path / "registration.json").write_bytes(contract_raw)
    progress = dict(completed=[dict(id=folder.name, run=str(folder),
                                   summary_sha256=entry(raw)["sha256"])],
                    active=[], pending=["other-running-task"])
    (tmp_path / "runs-v1" / "progress.json").write_text(json.dumps(progress))
    monkeypatch.setattr("sys.argv", ["collector", str(tmp_path),
                                     entry(contract_raw)["sha256"], summary["status"]])
    return progress


def test_collector_only_completed_accounts(tmp_path, monkeypatch, capsys):
    collector_fixture(tmp_path, monkeypatch)
    exec(COLLECTOR, {})
    snapshot = json.loads(capsys.readouterr().out)
    assert [x["id"] for x in snapshot["accounts"]] == ["coin--control"]


@pytest.mark.parametrize("change", ["active", "summary_changed", "duplicate"])
def test_collector_rejects_live_or_changed_completion(tmp_path, monkeypatch, change):
    progress = collector_fixture(tmp_path, monkeypatch)
    if change == "active":
        progress["active"] = ["coin--control"]
    elif change == "summary_changed":
        progress["completed"][0]["summary_sha256"] = "wrong"
    else:
        progress["completed"] *= 2
    (tmp_path / "runs-v1" / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError):
        exec(COLLECTOR, {})

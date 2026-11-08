"""Archive completed native accounts without copying filesystem metadata.

This host-side tool does not modify a registered experiment or remove its data.
It streams bytes, verifies source and destination SHA256, and reuses an exact
existing archive. A partial copy remains explicitly partial after a failure.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile


COLLECTOR = r'''
from pathlib import Path
import hashlib,json,sys
def require(ok,message):
    if not ok:raise ValueError(message)
root=Path(sys.argv[1]); expected=sys.argv[2]; status=sys.argv[3]
raw=(root/'registration.json').read_bytes()
require(hashlib.sha256(raw).hexdigest()==expected, 'Registration identity differs')
contract=json.loads(raw); tasks={t['id']:t for t in contract['tasks']}
require(len(tasks)==len(contract['tasks']), 'Duplicate registered task')
progress=json.loads((root/'runs-v1/progress.json').read_text()); accounts=[]; seen=set()
for item in progress['completed']:
    key=item['id']
    require(key in tasks and key not in seen and key not in progress['active'], 'Not an inactive completed registered account')
    seen.add(key); folder=Path(tasks[key]['config']['output'])
    require(folder==Path(item['run']) and folder==root/'runs-v1'/key and not folder.is_symlink(), 'Account path differs')
    summary_raw=(folder/'summary.json').read_bytes(); summary=json.loads(summary_raw)
    require(hashlib.sha256(summary_raw).hexdigest()==item['summary_sha256'], 'Completed summary identity differs')
    require(summary['id']==key and summary['status']==status, 'Account status differs')
    require(summary['economic_adoption'] is False and summary['full_cost_adoption'] is False, 'Unexpected adoption claim')
    files={}
    for p in sorted(folder.rglob('*')):
        require(not p.is_symlink(), 'Source symlink is unsupported')
        if p.is_dir():continue
        require(p.is_file(), 'Unsupported source object')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for b in iter(lambda:stream.read(1048576),b''):h.update(b)
        files[str(p.relative_to(folder))]=dict(sha256=h.hexdigest(),bytes=p.stat().st_size)
    accounts.append(dict(id=key,summary_sha256=item['summary_sha256'],files=files))
print(json.dumps(dict(registration_sha256=expected,accounts=accounts)))
'''


RECEIPT = "byte-archive-receipt.json"
LEGACY_RECEIPT = "archive-receipt.json"


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError("Unsafe archive path: " + name)
    return Path(*path.parts)


def digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Archive must contain regular files: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(sha256=h.hexdigest(), bytes=path.stat().st_size)


def verify_folder(folder, expected):
    if folder.is_symlink():
        raise ValueError("Archive directory is a symlink")
    actual = {}
    metadata = []
    for p in folder.rglob("*"):
        if p.is_symlink():
            raise ValueError("Archive contains a symlink")
        if p.is_file():
            relative = str(p.relative_to(folder))
            if relative in (RECEIPT, LEGACY_RECEIPT):
                continue
            if p.name.startswith('._'):
                companion = p.with_name(p.name[2:])
                name = str(companion.relative_to(folder))
                with p.open('rb') as stream:
                    header = stream.read(8)
                if (header == bytes.fromhex('0005160700020000') and p.stat().st_size >= 26
                    and not companion.is_symlink() and companion.is_file()
                    and name in set(expected) | {RECEIPT, LEGACY_RECEIPT}):
                    metadata.append(relative)
                    continue
            actual[relative] = digest(p)
    if actual != expected:
        raise ValueError("Existing or copied archive bytes/membership differ")
    return metadata


def stream_files(stream, destination, expected):
    copied = {}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if member.isdir():
                safe_relative(member.name) if member.name not in (".", "./") else None
                continue
            relative = safe_relative(member.name)
            key = relative.as_posix()
            if not member.isfile() or key not in expected or key in copied:
                raise ValueError("Unexpected or duplicate archive member: " + member.name)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            size = 0
            with archive.extractfile(member) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    h.update(block)
                    size += len(block)
            copied[key] = dict(sha256=h.hexdigest(), bytes=size)
    if copied != expected:
        raise ValueError("Source stream changed or omitted registered account files")
    verify_folder(destination, expected)


def archive_account(account, destination, registration_sha, stream_factory):
    key = account["id"]
    relative = safe_relative(key)
    if len(relative.parts) != 1:
        raise ValueError("Account id must be a single path component")
    expected = account["files"]
    for name in expected:
        safe_relative(name)
    if RECEIPT in expected or LEGACY_RECEIPT in expected:
        raise ValueError("Source collides with archive receipt names")
    receipt = dict(status="completed_account_bytes_verified_not_economic_adoption",
                   registration_sha256=registration_sha, **account,
                   economic_adoption=False, long_term_goal_complete=False)
    final = destination / relative
    reused = final.exists()
    if reused:
        verify_folder(final, expected)
        existing = final / RECEIPT
        if existing.exists() and json.loads(existing.read_text()) != receipt:
            raise ValueError("Existing archive registration identity differs")
    else:
        partial = Path(tempfile.mkdtemp(prefix="." + key + ".partial-", dir=destination))
        process = stream_factory(key)
        try:
            stream_files(process.stdout, partial, expected)
            if process.wait() != 0:
                raise RuntimeError("Native tar command failed")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise  # Keep the visibly partial directory; never mark it complete.
        (partial / RECEIPT).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        if final.exists():
            raise FileExistsError("Archive appeared during copy; preserve partial")
        partial.rename(final)
    if not (final / RECEIPT).exists():
        with (final / RECEIPT).open("x") as out:
            out.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    metadata = verify_folder(final, expected)
    return dict(id=key, reused=reused, files=len(expected),
                ignored_valid_appledouble_files=len(metadata),
                bytes=sum(x["bytes"] for x in expected.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", required=True)
    p.add_argument("--python", required=True, help="Native Python executable")
    p.add_argument("--root", required=True, help="Registered native experiment root")
    p.add_argument("--registration-sha", required=True)
    p.add_argument("--account-status", required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    snapshot = json.loads(subprocess.check_output([
        "docker", "exec", a.container, a.python, "-c", COLLECTOR,
        a.root, a.registration_sha, a.account_status]))
    a.destination.mkdir(parents=True, exist_ok=True)
    def open_stream(key):
        return subprocess.Popen(["docker", "exec", a.container, "tar", "-C",
                                 a.root + "/runs-v1/" + key, "-cf", "-", "."],
                                stdout=subprocess.PIPE)
    rows = [archive_account(account, a.destination, a.registration_sha, open_stream)
            for account in snapshot["accounts"]]
    print(json.dumps(dict(status="completed_only_byte_archives_verified",
                          accounts=rows, native_data_removed=False,
                          economic_adoption=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

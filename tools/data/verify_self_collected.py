#!/usr/bin/env python3
"""自采数据接收端校验:逐文件比对 .manifest.json 中的 sha256/bytes。

用法:
  python3 tools/data/verify_self_collected.py [--dir $QUANT_DATA_ROOT/crypto/raw/self_collected/binance_futures]

退出码:0=全部通过;1=存在缺失或校验失败;2=参数/目录错误。
校验依据与 doc/data.md 的转存规范一致:
只有"已封存(有 manifest)且 SHA256 一致"的文件才允许进入服务器端删除流程。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="$QUANT_DATA_ROOT/crypto/raw/self_collected/binance_futures")
    args = ap.parse_args()
    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在: {root}", file=sys.stderr)
        return 2

    manifests = sorted(root.rglob("*.manifest.json"))
    if not manifests:
        print(f"未找到任何 manifest: {root}", file=sys.stderr)
        return 2

    ok = missing = hash_bad = size_bad = 0
    failures: list[str] = []

    for mp in manifests:
        try:
            m = json.loads(mp.read_text())
        except Exception as e:
            failures.append(f"MANIFEST_UNREADABLE\t{mp}\t{e}")
            hash_bad += 1
            continue
        rel = m.get("path", "")
        data = root / rel
        if not data.is_file():
            missing += 1
            failures.append(f"MISSING\t{rel}")
            continue
        expect_bytes = m.get("bytes")
        if expect_bytes is not None and data.stat().st_size != expect_bytes:
            size_bad += 1
            failures.append(
                f"SIZE\t{rel}\tlocal={data.stat().st_size}\tmanifest={expect_bytes}"
            )
            continue
        digest = sha256_of(data)
        if digest != m.get("sha256"):
            hash_bad += 1
            failures.append(f"SHA256\t{rel}\tlocal={digest}\tmanifest={m.get('sha256')}")
            continue
        ok += 1

    total = len(manifests)
    print(f"manifest 总数: {total}")
    print(f"通过: {ok}  缺失: {missing}  字节不符: {size_bad}  SHA256不符: {hash_bad}")
    if failures:
        print("\n失败明细:")
        for line in failures:
            print(line)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

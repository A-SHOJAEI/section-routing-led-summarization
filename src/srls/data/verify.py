from __future__ import annotations

import argparse
from pathlib import Path

from srls.config import cfg_get, load_config
from srls.utils.hashing import compute_tree_checksums, read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg_get(cfg, "data.processed_dir"))
    ck_path = processed_dir / "checksums.json"
    if not ck_path.exists():
        raise FileNotFoundError(f"Missing checksums: {ck_path}. Run data preparation first.")

    recorded = read_json(ck_path)
    expected = recorded.get("sha256", {})
    actual = compute_tree_checksums(processed_dir)

    missing = sorted(set(expected.keys()) - set(actual.keys()))
    extra = sorted(set(actual.keys()) - set(expected.keys()))
    mismatched = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])

    if missing or extra or mismatched:
        lines = ["Checksum verification FAILED:"]
        if missing:
            lines.append(f"- missing files: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if extra:
            lines.append(f"- unexpected files: {extra[:20]}{' ...' if len(extra) > 20 else ''}")
        if mismatched:
            lines.append(f"- mismatched files: {mismatched[:20]}{' ...' if len(mismatched) > 20 else ''}")
        raise RuntimeError("\n".join(lines))


if __name__ == "__main__":
    main()


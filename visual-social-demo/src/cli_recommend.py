"""CLI helper for quick smoke tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.bootstrap_images import ensure_placeholder_images
    from src.io_profiles import load_profiles
    from src.match.explainer import explain_match
    from src.match.scoring import rank_candidates
    from src.profile.builder import build_all_personas
    from src.vision.extractor import VisionExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="query user_id e.g. u001")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    ensure_placeholder_images(root)
    users = load_profiles(root)
    personas = build_all_personas(users, VisionExtractor(), root)

    qid = args.user
    if qid not in personas:
        raise SystemExit(f"Unknown user_id: {qid}")

    query = personas[qid]
    rows = rank_candidates(query, personas, top_k=args.top)
    print(f"Query {qid} {query.nickname} mode={query.preference_mode}")
    for cid, sp in rows:
        print(f"\n--- {cid} {personas[cid].nickname} score={sp.total_display:.2f}")
        for line in explain_match(query, personas[cid], sp):
            print(line)


if __name__ == "__main__":
    main()

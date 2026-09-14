"""Compare three VLM binding criteria on SS and LSDA arms (criterion sensitivity)."""
import json
from pathlib import Path

R = Path(r"D:\Python\MMDIT\experiment\2026_9_12_EXP_3_KA_ME")


def load(p):
    d = {}
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            d[r["id"]] = r
    return d


cls = json.loads((R / "analysis/classification.json").read_text(encoding="utf-8"))
census = load(R / "ratings/census_scores.jsonl")
rep = load(R / "ratings/repair_scores.jsonl")
rep.update(load(R / "ratings/meattrs_scores.jsonl"))


def sides(row):
    out = []
    for r in ("gpt54", "gemini35"):
        d = (row or {}).get(r) or {}
        rv, lv = d.get("right_is_b"), d.get("left_is_a")
        if not isinstance(rv, (int, float)) and isinstance(lv, (int, float)):
            rv = 1.0 - lv
        if not isinstance(rv, (int, float)) or not isinstance(lv, (int, float)):
            return None
        out.append((float(lv), float(rv)))
    return out if len(out) == 2 else None


def c1_rater_strict(row):  # correct unless BOTH raters fail (any side)
    s = sides(row)
    if not s:
        return None
    return not all(lv < 0.5 or rv < 0.5 for lv, rv in s)


def c2_side_consensus(row):  # correct unless both raters fail the SAME side
    s = sides(row)
    if not s:
        return None
    right_fail = all(rv < 0.5 for _, rv in s)
    left_fail = all(lv < 0.5 for lv, _ in s)
    return not (right_fail or left_fail)


def c3_strict_both(row):  # correct only if both raters pass both sides
    s = sides(row)
    if not s:
        return None
    return all(lv >= 0.5 and rv >= 0.5 for lv, rv in s)


def arm_rows():
    for rec in cls.values():
        if rec.get("label") not in ("KA", "ME", "BC"):
            continue
        p, s = int(rec["pair"]), int(rec["seed"])
        ss = census.get(f"cen_p{p:03d}_s{s}_SS")
        if rec["label"] == "ME":
            ls = rep.get(f"lsda_meattrs_p{p:03d}_s{s}")
        elif rec["label"] == "KA":
            ls = rep.get(f"lsda_ka_p{p:03d}_s{s}")
        else:
            ls = ss
        yield ss, ls


for name, fn in (("c1_rater_strict", c1_rater_strict), ("c2_side_consensus", c2_side_consensus), ("c3_strict_both", c3_strict_both)):
    ss_ok = ss_n = ls_ok = ls_n = 0
    for ss, ls in arm_rows():
        a, b = fn(ss), fn(ls)
        if a is not None:
            ss_n += 1; ss_ok += a
        if b is not None:
            ls_n += 1; ls_ok += b
    print(f"{name:20s} SS drift={1-ss_ok/ss_n:.3f} (n={ss_n})   LSDA drift={1-ls_ok/ls_n:.3f} (n={ls_n})")

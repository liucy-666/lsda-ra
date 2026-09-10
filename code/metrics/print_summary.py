"""Print the unified analysis summary in a compact table."""
from __future__ import annotations

import json

s = json.load(
    open(r"D:\Python\MMDIT\experiment\2026_8_31_EXP_2\analysis\unified_summary.json", encoding="utf-8")
)

print(f"{'metric':10s} {'native':>8s} {'lsda':>8s} {'rel_red':>8s} {'restored':>9s} {'worsened':>8s} {'mcnemar_p':>10s}")
for name, m in list(s["metrics"].items()) + [("consensus", s["consensus"])]:
    print(
        f"{name:10s} {m['native_SS']['drift_rate']:8.1%} {m['lsda_clean']['drift_rate']:8.1%} "
        f"{m['relative_drift_reduction']:8.1%} {m['restored']:9d} {m['worsened']:8d} {m['mcnemar_p']:10.1e}"
    )
print()
print("consensus methods:", s.get("consensus_methods"))
print("kappa vs Qwen:", s["kappa_vs_qwen"])
print("kappa vs Gemini:", s["kappa_vs_gemini"])
print("kappa Qwen vs Gemini:", s["kappa_qwen_vs_gemini"])
print("kappa vs dual-vlm-fail:", s.get("kappa_vs_dual_vlm_fail"))
dv = s.get("dual_vlm_consensus", {})
if dv:
    print("dual-VLM native:", round(dv["native_SS"]["drift_rate"], 4), "lsda:", round(dv["lsda_clean"]["drift_rate"], 4))

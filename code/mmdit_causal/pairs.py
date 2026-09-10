"""Cultural pairs and prompt builders (no torch dependency)."""

PAIRS = {
    1: {"A": "Chinese blue-and-white porcelain vase", "B": "Italian maiolica vase"},
    2: {"A": "Japanese Imari porcelain vase", "B": "Dutch Delftware vase"},
    3: {"A": "Chinese cloisonne enamel vase", "B": "French Sevres porcelain vase"},
    4: {"A": "Turkish Iznik ceramic plate", "B": "Chinese famille rose plate"},
    5: {"A": "Mexican Talavera pottery plate", "B": "Turkish Canakkale ceramic plate"},
    6: {"A": "Japanese Kutani porcelain bowl", "B": "Russian Khokhloma bowl"},
}


def ss_prompt(pair_id: int) -> str:
    p = PAIRS[pair_id]
    return (
        f"Neutral studio background: a {p['A']} on the left, a {p['B']} on the right; "
        "both fully visible, separate, and similar in size."
    )


def b_donor_prompt(pair_id: int) -> str:
    p = PAIRS[pair_id]
    return f"Neutral studio background: a {p['B']} on the right; fully visible."


def a_only_prompt(pair_id: int) -> str:
    return f"a {PAIRS[pair_id]['A']}"


def b_only_prompt(pair_id: int) -> str:
    return f"a {PAIRS[pair_id]['B']}"

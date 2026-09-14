"""Extract first-page text of reference PDFs to a UTF-8 file."""
from pathlib import Path

from pypdf import PdfReader

REFS = Path(r"D:\Python\MMDIT\references")
OUT = Path(r"C:\Users\admin\AppData\Local\Temp\opencode\ref_first_pages.txt")
chunks = []
for name in ["LayerBind 2026CVPR.pdf", "SplitFlux 2026 CVPR.pdf", "DreamRenderer 2025 ICCV.pdf"]:
    chunks.append("=" * 80 + "\n" + name + "\n")
    try:
        reader = PdfReader(str(REFS / name))
        text = (reader.pages[0].extract_text() or "")
        chunks.append(text[:1800])
    except Exception as exc:  # noqa: BLE001
        chunks.append(f"ERR {exc}")
    chunks.append("\n")
OUT.write_text("\n".join(chunks), encoding="utf-8")
print("wrote", OUT)


from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

txt_path = Path("output/docai_text/merged_ALLfile.txt")

script_dir = Path(__file__).parent.resolve()
pdf_path = script_dir / "datafinal.pdf"

font_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]

font_path = next((p for p in font_candidates if Path(p).exists()), None)
if not font_path:
    raise FileNotFoundError(
        "Không tìm thấy font Unicode. Hãy cài: sudo apt install fonts-dejavu-core "
        "hoặc fonts-noto-core"
    )

pdfmetrics.registerFont(TTFont("VietFont", font_path))

c = canvas.Canvas(str(pdf_path), pagesize=A4)
width, height = A4

x_margin = 40
y_margin = 40
y = height - y_margin
line_height = 14

c.setFont("VietFont", 11)

with open(txt_path, "r", encoding="utf-8", errors="strict") as f:
    for line in f:
        line = line.rstrip("\n")
        if y < y_margin:
            c.showPage()
            c.setFont("VietFont", 11)
            y = height - y_margin

        # drawString không tự xuống dòng, nên mỗi line là 1 dòng
        c.drawString(x_margin, y, line)
        y -= line_height

c.save()
print(f"Đã tạo PDF (đúng tiếng Việt): {pdf_path}")


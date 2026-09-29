from pathlib import Path
from PIL import Image, ImageDraw

out = Path("data/sample/sample_medical_like.png")
out.parent.mkdir(parents=True, exist_ok=True)

img = Image.new("RGB", (512, 512), (35, 35, 35))
draw = ImageDraw.Draw(img)
draw.ellipse((120, 90, 392, 422), fill=(125, 125, 125))
draw.ellipse((205, 160, 310, 300), fill=(175, 175, 175))
draw.rectangle((20, 20, 120, 50), fill=(80, 80, 80))
img.save(out)
print(out)

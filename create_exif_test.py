from PIL import Image, ImageDraw

# Create test image
img = Image.new("RGB", (800, 500), "white")

draw = ImageDraw.Draw(img)

draw.rectangle(
    (60, 60, 740, 440),
    outline="black",
    width=4
)

draw.text(
    (120, 130),
    "EXIF TEST IMAGE",
    fill="black"
)

draw.text(
    (120, 190),
    "Digital Image Forensics",
    fill="black"
)

draw.text(
    (120, 250),
    "This image contains test EXIF metadata.",
    fill="black"
)

# Create EXIF metadata
exif = Image.Exif()

exif[271] = "TestCamera"
exif[272] = "Forensics Demo 1000"
exif[306] = "2026:09:22 19:00:00"
exif[36867] = "2026:09:22 19:00:00"
exif[305] = "AI Forensics EXIF Test"
exif[274] = 1

# Save image
img.save(
    "exif_test_image.jpg",
    format="JPEG",
    quality=95,
    exif=exif
)

print("EXIF test image created successfully.")
print("File: exif_test_image.jpg")
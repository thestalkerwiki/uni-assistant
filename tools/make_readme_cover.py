from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFilter

# Input files
SCREENSHOT_PATH = Path("static/assets/hero-banner.png")
LOGO_PATH = Path("static/assets/uni-assist-avatar.png")

# Output files
OUTPUT_COVER_PATH = Path("static/assets/readme-cover.png")
OUTPUT_LOGO_PATH = Path("static/assets/readme-logo.png")

# README cover ratio: wide Notion / YouTube-like banner
COVER_WIDTH = 1600
COVER_HEIGHT = 520

# Logo size for README title block
LOGO_SIZE = 96


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side

    return image.crop((left, top, right, bottom))


def add_round_corners(image: Image.Image, radius: int) -> Image.Image:
    image = image.convert("RGBA")

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, image.size[0], image.size[1]), radius=radius, fill=255)

    image.putalpha(mask)
    return image


def make_cover() -> None:
    screenshot = Image.open(SCREENSHOT_PATH).convert("RGB")

    # Crop screenshot into a wide cover.
    # centering=(0.5, 0.35) keeps more of the upper interface visible.
    cover = ImageOps.fit(
        screenshot,
        (COVER_WIDTH, COVER_HEIGHT),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.35),
    ).convert("RGBA")

    # Soft readable overlay, like a professional product cover.
    overlay = Image.new("RGBA", cover.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Light vignette / premium softness
    draw.rectangle((0, 0, COVER_WIDTH, COVER_HEIGHT), fill=(255, 255, 255, 26))

    # Subtle bottom fade
    for y in range(COVER_HEIGHT):
        alpha = int(35 * (y / COVER_HEIGHT))
        draw.line((0, y, COVER_WIDTH, y), fill=(246, 244, 238, alpha))

    cover = Image.alpha_composite(cover, overlay)

    # Rounded corners look better inside README.
    cover = add_round_corners(cover, radius=28)
    cover.save(OUTPUT_COVER_PATH)

    print(f"Saved cover: {OUTPUT_COVER_PATH}")


def make_logo() -> None:
    logo = Image.open(LOGO_PATH).convert("RGBA")

    # The original owl image is wide, so crop the central square.
    logo = center_crop_square(logo)
    logo = logo.resize((LOGO_SIZE, LOGO_SIZE), Image.Resampling.LANCZOS)
    logo = add_round_corners(logo, radius=18)

    logo.save(OUTPUT_LOGO_PATH)

    print(f"Saved logo: {OUTPUT_LOGO_PATH}")


if __name__ == "__main__":
    make_cover()
    make_logo()
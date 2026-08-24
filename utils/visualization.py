from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image: Image.Image,
    boxes,
    scores,
    labels,
    score_threshold: float,
) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for box, score, label in zip(boxes, scores, labels):
        if float(score) < score_threshold:
            continue
        xyxy = tuple(float(value) for value in box)
        draw.rectangle(xyxy, outline=(255, 0, 255), width=2)
        draw.text((xyxy[0] + 2, xyxy[1] + 2), f"ship {float(score):.3f}", fill=(255, 0, 255), font=font)
    return output


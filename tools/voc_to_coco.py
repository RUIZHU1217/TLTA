from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SSDD-style Pascal VOC XML to COCO JSON")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-file", type=Path, default=None, help="Optional list of image stems")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split_file:
        stems = [line.strip() for line in args.split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        xml_files = [args.annotations / f"{stem}.xml" for stem in stems]
    else:
        xml_files = sorted(args.annotations.glob("*.xml"))

    output = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "ship"}]}
    annotation_id = 1
    for image_id, xml_path in enumerate(xml_files, start=1):
        root = ElementTree.parse(xml_path).getroot()
        filename = root.findtext("filename")
        if not filename:
            candidates = list(args.images.glob(f"{xml_path.stem}.*"))
            if not candidates:
                raise FileNotFoundError(f"No image for {xml_path}")
            filename = candidates[0].name
        image_path = args.images / filename
        size_node = root.find("size")
        if size_node is not None:
            width = int(size_node.findtext("width", "0"))
            height = int(size_node.findtext("height", "0"))
        else:
            with Image.open(image_path) as image:
                width, height = image.size
        output["images"].append({"id": image_id, "file_name": filename, "width": width, "height": height})
        for object_node in root.findall("object"):
            if object_node.findtext("name", "ship").lower() != "ship":
                continue
            box = object_node.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            width_box, height_box = max(0.0, xmax - xmin), max(0.0, ymax - ymin)
            if width_box == 0 or height_box == 0:
                continue
            output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, width_box, height_box],
                    "area": width_box * height_box,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Converted {len(output['images'])} images and {len(output['annotations'])} boxes")


if __name__ == "__main__":
    main()


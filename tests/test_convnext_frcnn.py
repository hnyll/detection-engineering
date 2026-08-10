import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.convnext_frcnn import (ARCH, FAMILY, YoloBoxDataset, validate_config)
from src.expmeta import inference_identity


class ConvNeXtFRCNNTest(unittest.TestCase):
    def _config(self):
        return {
            "meta": {"model_family": FAMILY},
            "evaluation": {
                "model_pipeline": {
                    "schema_version": 1,
                    "family": FAMILY,
                }
            },
            "train": {
                "model": "convnext_tiny",
                "architecture": ARCH,
                "imgsz": 960,
                "batch": 1,
                "anchor_sizes": [8, 16, 32, 64, 128],
            },
        }

    def test_config_and_protocol_identity(self):
        train = validate_config(self._config())
        self.assertEqual(train["anchor_sizes"], (8, 16, 32, 64, 128))
        self.assertEqual(train["batch"], 1)
        self.assertEqual(
            inference_identity(self._config())["model_pipeline"]["family"], FAMILY
        )

    def test_yolo_boxes_become_native_xyxy_and_foreground_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image_path = image_dir / "sample.jpg"
            Image.new("RGB", (100, 50), "white").save(image_path)
            (label_dir / "sample.txt").write_text("2 0.5 0.5 0.2 0.4\n")
            dataset = YoloBoxDataset([image_path], augment=False)
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 50, 100))
            torch.testing.assert_close(
                target["boxes"], torch.tensor([[40.0, 15.0, 60.0, 35.0]])
            )
            self.assertEqual(target["labels"].tolist(), [3])


if __name__ == "__main__":
    unittest.main()

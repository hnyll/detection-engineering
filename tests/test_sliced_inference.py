import json
import tempfile
import unittest
from pathlib import Path

from src.sliced_inference import (_merge_candidates, _shift_clip_box, _slice_boxes,
                                  native_size_recall, normalized_tiling)


class SlicedInferenceTest(unittest.TestCase):
    def test_normalization_and_native_recall(self):
        proto = {"conf": 0.001, "iou": 0.7, "max_det": 300}
        cfg = {
            "evaluation": {
                "tiling": {
                    "enabled": True,
                    "tile_height": 640,
                    "tile_width": 640,
                    "overlap_height_ratio": 0.2,
                    "overlap_width_ratio": 0.2,
                    "per_tile_conf": 0.001,
                    "per_tile_nms_iou": 0.7,
                    "merge_iou": 0.7,
                    "global_max_det": 300,
                }
            }
        }
        tiling = normalized_tiling(cfg, proto)
        self.assertEqual(tiling["tile_width"], 640)
        self.assertEqual(tiling["merge"], "class_aware_nms")
        self.assertEqual(tiling["schema_version"], 2)

        cfg["evaluation"]["tiling"]["include_full_image"] = True
        fused = normalized_tiling(cfg, proto)
        self.assertTrue(fused["include_full_image"])
        self.assertEqual(_slice_boxes(540, 960, tiling), _slice_boxes(540, 960, fused))
        self.assertEqual(len(_slice_boxes(540, 960, fused)), 2)
        self.assertEqual(len(_slice_boxes(765, 1360, fused)), 6)
        self.assertEqual(len(_slice_boxes(1080, 1920, fused)), 8)

        mapped = _shift_clip_box([-5, 2, 20, 30], 100, 50, 110, 70)
        self.assertEqual(mapped, [95.0, 52.0, 110.0, 70.0])
        self.assertIsNone(_shift_clip_box([20, 2, 20, 30], 100, 50, 110, 70))

        merged, post_merge = _merge_candidates(
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]],
            [0.9, 0.8, 0.7], [0, 0, 1], merge_iou=0.5, max_det=10,
        )
        self.assertEqual(post_merge, 2)
        self.assertEqual([(row["cls"], round(row["score"], 1)) for row in merged], [(0, 0.9), (1, 0.7)])
        capped, _ = _merge_candidates(
            [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.7], [0, 1], 0.5, 1,
        )
        self.assertEqual(len(capped), 1)

        cross_view, _ = _merge_candidates(
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]],
            [0.9, 0.8, 0.7], [0, 0, 1], 0.5, 10,
            sources=["tile", "full_frame", "full_frame"],
        )
        self.assertEqual(
            [(row["cls"], row["source"]) for row in cross_view],
            [(0, "tile"), (1, "full_frame")],
        )

        gt = {
            "images": [{"id": 0, "file_name": "x.jpg"}],
            "categories": [{"id": 1, "name": "x"}],
            "annotations": [
                {"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 6, 6]},
                {"id": 2, "image_id": 0, "category_id": 1, "bbox": [20, 0, 20, 20]},
            ],
        }
        detections = [
            {"image_id": 0, "category_id": 1, "bbox": [0, 0, 6, 6], "score": 0.8},
            {"image_id": 0, "category_id": 1, "bbox": [20, 0, 5, 5], "score": 0.7},
        ]
        with tempfile.TemporaryDirectory() as directory:
            gt_path = Path(directory) / "gt.json"
            gt_path.write_text(json.dumps(gt))
            report = native_size_recall(gt_path, detections)
        self.assertEqual(report["overall"]["matched"], 1)
        self.assertEqual(report["by_native_short_side"]["lt8"]["recall"], 1.0)
        self.assertEqual(report["by_native_short_side"]["16_to_31"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()

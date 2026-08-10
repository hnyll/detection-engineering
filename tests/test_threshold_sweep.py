import unittest

from src.threshold_sweep import select_best, sweep_records


class ThresholdSweepTest(unittest.TestCase):
    def test_class_aware_greedy_matching_and_selection(self):
        gt = {
            "images": [{"id": 0, "file_name": "synthetic.jpg"}],
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "annotations": [
                {"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
                {"id": 2, "image_id": 0, "category_id": 1, "bbox": [20, 0, 10, 10], "iscrowd": 0},
                {"id": 3, "image_id": 0, "category_id": 2, "bbox": [0, 20, 10, 10], "iscrowd": 0},
            ],
        }
        predictions = [
            {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.90},
            # Wrong class: it must be an FP for b, not a match for GT class a.
            {"image_id": 0, "category_id": 2, "bbox": [20, 0, 10, 10], "score": 0.80},
            {"image_id": 0, "category_id": 2, "bbox": [0, 20, 10, 10], "score": 0.40},
            # Duplicate on the first class-a GT.
            {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.30},
            # Exact-threshold inclusion should recover the second class-a GT.
            {"image_id": 0, "category_id": 1, "bbox": [20, 0, 10, 10], "score": 0.20},
        ]

        rows = sweep_records(gt, predictions, [0.20, 0.25], iou_threshold=0.5)
        low, high = rows
        self.assertEqual(tuple(low["overall"]["micro"][k] for k in ("tp", "fp", "fn")), (3, 2, 0))
        self.assertEqual(tuple(high["overall"]["micro"][k] for k in ("tp", "fp", "fn")), (2, 2, 1))
        self.assertEqual(low["per_class"]["a"]["tp"], 2)
        self.assertEqual(high["per_class"]["a"]["fn"], 1)

        selected = select_best(rows)
        self.assertEqual(selected["recommended_global"]["threshold"], 0.20)
        self.assertEqual(selected["best_per_class"]["a"]["threshold"], 0.20)
        # Class b is unchanged, so the documented exact-tie rule chooses 0.25.
        self.assertEqual(selected["best_per_class"]["b"]["threshold"], 0.25)


if __name__ == "__main__":
    unittest.main()

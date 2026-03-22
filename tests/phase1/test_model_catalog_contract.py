import unittest
from pathlib import Path

import yaml


CATALOG_PATH = Path("config/models/catalog.yaml")
SUPPORTED_MODEL_IDS = {
    "facebook/mbart-large-50-many-to-many-mmt",
    "google/mt5-base",
    "facebook/nllb-200-distilled-600M",
}


class TestModelCatalogContract(unittest.TestCase):
    def test_catalog_contains_exact_supported_model_ids(self) -> None:
        self.assertTrue(CATALOG_PATH.exists(), "catalog.yaml must exist")

        with CATALOG_PATH.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        approved_models = payload.get("approved_models", [])
        actual_ids = {entry.get("model_id") for entry in approved_models}

        self.assertEqual(SUPPORTED_MODEL_IDS, actual_ids)
        self.assertEqual(
            3, len(approved_models), "catalog must contain exactly 3 approved models"
        )


if __name__ == "__main__":
    unittest.main()

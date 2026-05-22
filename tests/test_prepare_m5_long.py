import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_m5_long import prepare_m5_long
from vara.forecast_data import load_panel_data


class PrepareM5LongTests(unittest.TestCase):
    def test_prepare_m5_long_writes_long_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "m5"
            out = root / "long"
            raw.mkdir()

            pd.DataFrame({
                "date": pd.date_range("2011-01-29", periods=3, freq="D"),
                "d": ["d_1", "d_2", "d_3"],
            }).to_parquet(raw / "calendar.parquet", index=False)
            pd.DataFrame({
                "id": ["A_validation", "B_validation"],
                "item_id": ["A", "B"],
                "dept_id": ["D1", "D1"],
                "cat_id": ["C", "C"],
                "store_id": ["S1", "S1"],
                "state_id": ["ST", "ST"],
                "d_1": [1, 4],
                "d_2": [2, 5],
                "d_3": [3, 6],
            }).to_parquet(raw / "sales_train_validation.parquet", index=False)

            metadata = prepare_m5_long(
                raw_dir=raw,
                output_dir=out,
                splits=("validation",),
                chunk_series=1,
            )

            summary = metadata["summaries"]["validation"]
            self.assertEqual(summary["n_rows"], 6)
            self.assertEqual(summary["n_series"], 2)
            self.assertEqual(summary["date_start"], "2011-01-29")
            self.assertEqual(summary["date_end"], "2011-01-31")
            long = pd.read_parquet(summary["output_path"])
            self.assertEqual(list(long.columns), ["date", "series_id", "value", "d", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
            self.assertEqual(set(long["series_id"]), {"A_validation", "B_validation"})
            self.assertEqual(long["value"].sum(), 21)

            wide = load_panel_data(Path(summary["output_path"]), data_format="long")
            self.assertEqual(list(wide.columns), ["date", "A_validation", "B_validation"])
            self.assertEqual(wide["A_validation"].tolist(), [1, 2, 3])
            self.assertTrue((out / "metadata.json").exists())
            self.assertTrue((out / "sales_train_validation_series_metadata.parquet").exists())


if __name__ == "__main__":
    unittest.main()

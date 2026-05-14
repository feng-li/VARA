import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.prepare_m4_yearly import prepare_m4_yearly


class PrepareM4YearlyTests(unittest.TestCase):
    def test_prepare_m4_yearly_writes_long_and_tail_wide_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "out"
            raw.mkdir()

            pd.DataFrame({
                "V1": ["Y1", "Y2", "Y3"],
                "V2": [10.0, 20.0, 30.0],
                "V3": [11.0, 21.0, 31.0],
                "V4": [12.0, 22.0, None],
                "V5": [None, 23.0, None],
            }).to_csv(raw / "Yearly-train.csv", index=False)
            pd.DataFrame({
                "V1": ["Y1", "Y2", "Y3"],
                "V2": [13.0, 24.0, 32.0],
                "V3": [14.0, 25.0, 33.0],
            }).to_csv(raw / "Yearly-test.csv", index=False)
            pd.DataFrame({
                "M4id": ["Y1", "Y2", "Y3"],
                "category": ["Macro", "Finance", "Other"],
                "Frequency": ["Yearly", "Yearly", "Yearly"],
                "Horizon": [2, 2, 2],
                "SP": [1, 1, 1],
                "StartingDate": ["2000-01-01", "1999", "2010-01-01"],
            }).to_csv(raw / "M4-info.csv", index=False)

            metadata = prepare_m4_yearly(raw_dir=raw, output_dir=out, tail_length=3, max_series=0)

            train_long = pd.read_csv(out / "m4_yearly_train_long.csv")
            test_long = pd.read_csv(out / "m4_yearly_test_long.csv")
            full_long = pd.read_csv(out / "m4_yearly_full_long.csv")
            info = pd.read_csv(out / "m4_yearly_series_info.csv")
            tail_wide = pd.read_csv(out / "m4_yearly_tail3_wide.csv")
            tail_info = pd.read_csv(out / "m4_yearly_tail3_series_info.csv")

            self.assertEqual(metadata["n_series"], 3)
            self.assertEqual(metadata["tail_n_series"], 2)
            self.assertEqual(metadata["test_horizon_values"], [2])
            self.assertEqual(len(train_long), 9)
            self.assertEqual(len(test_long), 6)
            self.assertEqual(len(full_long), 15)
            self.assertEqual(set(info["series_id"]), {"Y1", "Y2", "Y3"})
            self.assertEqual(list(tail_wide.columns), ["date", "Y1", "Y2"])
            self.assertEqual(tail_wide.shape, (5, 3))
            self.assertEqual(tail_wide["Y1"].tolist(), [10.0, 11.0, 12.0, 13.0, 14.0])
            self.assertEqual(tail_wide["Y2"].tolist(), [21.0, 22.0, 23.0, 24.0, 25.0])
            self.assertEqual(set(tail_info["series_id"]), {"Y1", "Y2"})
            self.assertEqual(train_long.loc[train_long["series_id"] == "Y1", "date"].iloc[0], "2000-01-01")
            self.assertEqual(test_long.loc[test_long["series_id"] == "Y1", "t"].tolist(), [4, 5])


if __name__ == "__main__":
    unittest.main()

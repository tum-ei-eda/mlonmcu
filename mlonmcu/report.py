from pathlib import Path
import pandas as pd


class Report:
    def __init__(self):
        self.df = pd.DataFrame()

    def export(self, path):
        ext = Path(path).suffix[1:]
        assert ext in ["csv"], f"Unsupported report format: {ext}"
        if ext == "csv":
            parent = Path(path).parent
            if not parent.is_dir():
                parent.mkdir()
            self.df.to_csv(path, index=False)
        else:
            raise RuntimeError()

    # def append(self, *args, **kwargs):
    #     self.df = self.df.append(*args, **kwargs, ignore_index=True)

    def set(self, data):
        self.df = pd.DataFrame.from_records(data)

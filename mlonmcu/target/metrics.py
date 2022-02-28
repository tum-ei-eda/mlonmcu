import io
import csv
import ast


class Metrics:
    def __init__(self):
        self.data = {}
        self.optional_keys = []

    @staticmethod
    def from_csv(text):
        # TODO: how to find out data types? -> pandas?
        lines = text.splitlines()
        assert len(lines) == 2, "Metrics should only have two lines"
        # headers, data = lines[0].split(","), lines[1]
        reader = csv.DictReader(lines)
        data = list(reader)[0]
        ret = Metrics()
        ret.data = data
        return ret

    def add(self, name, value, optional=False):
        assert name not in self.data, "Collumn with the same name already exists in metrics"
        self.data[name] = value
        if optional:
            self.optional_keys.append(name)

    def get_data(self, include_optional=False):
        return {
            key: ast.literal_eval(value) if isinstance(value, str) else value
            for key, value in self.data.items()
            if key not in self.optional_keys or include_optional
        }

    def to_csv(self, include_optional=False):
        data = self.get_data(include_optional=include_optional)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        return output.getvalue()

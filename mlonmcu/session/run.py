
class Run:

    def __init__(self, model=None, backend=None, num=1):
        self.model = model
        self.backend = backend
        self.artifacts = {}
        self.num = num

    def __repr__(self):
        return f"Run({self.model},{self.backend},{self.num})"


class Target:

    def __init__(self, name, features=[], config={}):
        self.name = name
        self.features = features
        self.config = config
        self.inspectProgram = "readelf"
        self.inspectprogramArgs = []

    def __repr__(self):
        return f"Target({self.name})"

    def exec(self, program, *args):
        os.system(...)

    def inspect(self, program):
        os.system(...)

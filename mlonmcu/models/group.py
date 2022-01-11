class ModelGroup:
    def __init__(self, name, models, description=""):
        self.name = name
        self.models = models
        self.description = description

    def __repr__(self):
        return f"ModelGroup({self.name},models={self.models})"

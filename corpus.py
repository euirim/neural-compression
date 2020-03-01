class Corpus:
    def __init__(self, name, filename, preprocess_func):
        self.name = name
        self.filename = filename
        self.preprocess_func = preprocess_func
        self.docs = None

    def load(self):
        with open(self.filename, "r") as f:
            self.docs = self.preprocess_func(f.read().strip())

    def __iter__(self):
        return iter(self.docs)

    def __str__(self):
        return self.name

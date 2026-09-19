import json


def load(path):
    with open(path) as handle:
        return json.load(handle)


def save(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle)


class Store:
    def __init__(self):
        self.items = {}

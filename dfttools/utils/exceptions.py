class ItemNotFoundError(Exception):
    def __init__(self, item):
        self.key = item[0]
        self.value = item[1]
        super().__init__(f"'{self.key}': '{self.value}' item not found in dictionary")

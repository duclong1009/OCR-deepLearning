import os

class LogTrain(object):
    def __init__(self, path, name="log.txt"):
        self.path = path
        self.name = name
        file_path = os.path.join(path, name)
        self.file = open(file_path, "w+")

    def write(self, data, end_line=True):
        if not isinstance(data, str):
            data = str(data)
        self.file.write(data + "\n") if end_line else self.file.write(data)
        self.file.flush()

    def close(self):
        self.file.close()

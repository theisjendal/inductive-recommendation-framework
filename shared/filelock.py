import fcntl


class FileLock:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, 'a')
        fcntl.flock(self.file, fcntl.LOCK_EX)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()
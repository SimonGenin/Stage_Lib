import os


class RecursiveBigPythonFileGenerator:

    def __init__(self, directory_path, destination_file_path):
        self.directory_path = directory_path
        self.destination_file_path = destination_file_path

    def create(self, limit=None):
        with open(self.destination_file_path, 'w') as dest:
            for folder, subs, files in os.walk(self.directory_path):
                print(folder)
                for index, filename in enumerate(files):
                    if limit is not None and index > limit:
                        break
                    if filename.endswith(".py"):
                        with open(os.path.join(folder, filename), 'r', encoding='UTF-8') as src:
                            dest.write(src.read())
                            dest.write('\n')

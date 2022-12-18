from os import listdir
from os.path import isfile, join


def get_filenames(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    print(onlyfiles)
    return onlyfiles
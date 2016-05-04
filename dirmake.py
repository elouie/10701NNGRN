from os import makedirs
from os import path
import errno

def dirmake(dirpath):
    try:
        makedirs(dirpath)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(dirpath):
            print "The directory {} already exists! You might be overwriting existing data! Are you sure!? (Y/N)".format(dirpath)
            answer = raw_input().lower()
            if not (answer == "y" or answer == "yes"):
                print "Stopping!"
                exit()
            else:
                pass
        else:
            raise


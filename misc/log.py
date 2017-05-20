import time, os
from misc.utils import create_folders_if_not_exist
from time import strftime, gmtime


# A general purpose class for logging
class Log():
    def __init__(self, folder):
        self.file_path = os.path.join(folder, "log_"+strftime("%b_%d_%H_%M_%S")+'.txt')
        create_folders_if_not_exist(self.file_path)

    def write(self, string, also_print=True, include_timestamp=True):
        if include_timestamp:
            string = "%s [INFO]: %s" % (strftime("%H:%M:%S"), string)
        if also_print:
            print(string)
        with open(self.file_path, "a") as f:
            f.write(string+" \n")

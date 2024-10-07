from fileinput import filename
import os
import sys
import datetime

def configure(dir=None,info=None):
    if dir == None: 
        dir = os.path.abspath(r"..")

    dir = os.path.join(dir,"log/")
    os.makedirs(dir, exist_ok=True)
    file_name = datetime.datetime.now().strftime(f"Log-[ {info} ]-%m%d-%H:%M:%S") + ".txt"
    path = dir + file_name
    Logger.CURRENT = Logger(path=path)

def get_current():
    if Logger.CURRENT is None:
        configure()
    return Logger.CURRENT

def log(content):
        get_current().log(content)    

class Logger():
    CURRENT = None

    def __init__(self,path):
        self.path = path

    def log(self,content):
        with open(self.path, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            f.write('content: ' + f'{content}' + ',Time: ' +f'{time}' + '---END\n')

    
    


import configparser

class ConfigProcessor:
    def __init__(self):
        pass

    def __enter__(self):
        return self
    
    def __exit__(self,*exc):
        del self
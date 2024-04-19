from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log(self, **kwargs):
        pass
    
    @abstractmethod
    def close(self):
        pass
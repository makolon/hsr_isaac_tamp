from abc import ABC, abstractmethod

class BaseHandler(ABC):
    def __init__(self, ):
        pass

    @abstractmethod
    def import_robot_assets(self):
        pass

    @abstractmethod
    def post_reset(self):
        pass

    @abstractmethod
    def apply_actions(self):
        pass

    @abstractmethod
    def reset(self):
        pass
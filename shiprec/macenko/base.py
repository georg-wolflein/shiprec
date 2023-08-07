import abc


class HENormalizer(abc.ABC):
    @abc.abstractmethod
    def fit(self, target):
        pass

    @abc.abstractmethod
    def normalize(self, I, **kwargs):
        pass

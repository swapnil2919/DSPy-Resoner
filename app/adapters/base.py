from abc import ABC, abstractmethod
class BenchmarkAdapter(ABC):
    @abstractmethod
    async def run(self, config: dict, scenario: dict):
        pass
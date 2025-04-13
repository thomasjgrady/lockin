import json
from pathlib import Path
from torch.utils.data import Dataset
from ..tokenizers import ChatMessage


class MATHDataset(Dataset[list[ChatMessage]]):
    
    def __init__(self, root_dir: str | Path) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.data_paths = list(self.root_dir.rglob("*.json"))

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index) -> list[ChatMessage]:
        with open(self.data_paths[index], "r") as f:
            data = json.load(f)
            return [
                ChatMessage(role="system", content=r"Think carefully and answer the given problem step by step. At the end of your response, output your answer in LaTeX like $\boxed{your answer}$"),
                ChatMessage(role="user", content=data["problem"]),
                ChatMessage(role="assistant", content=data["solution"]),
            ]
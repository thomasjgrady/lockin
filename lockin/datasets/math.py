from pathlib import Path
from torch.utils.data import Dataset
from lockin.tokenizers import ChatMessage


class MATHDataset(Dataset[list[ChatMessage]]):

    def __init__(self, root_dir: str | Path) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)

    def __len__(self) -> int:
        ...

    def __getitem__(self, index) -> list[ChatMessage]:
        ...
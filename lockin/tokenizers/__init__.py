from typing import Literal, Protocol
from pydantic import BaseModel
from torch import Tensor
import torch

from lockin.utils import pack


class Tokenizer(Protocol):

    def encode(self, text: str, bos: bool, eos: bool) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...

Role = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel, extra="forbid"):
    role: Role
    content: str

class ChatFormat(Protocol):

    def encode(self, dialog: list[ChatMessage], add_final_eot: bool) -> tuple[list[int], list[bool]]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def eot_id(self) -> int: ...

def tokenize_chat_data(batch: list[list[ChatMessage]], chat_format: ChatFormat, add_final_eot: bool) -> tuple[list[Tensor], list[Tensor]]:
    tups = [chat_format.encode(x, add_final_eot) for x in batch]
    tokens, masks = map(list, zip(*tups))
    tokens = [torch.tensor(x, dtype=torch.long) for x in tokens]
    masks = [torch.tensor(x, dtype=torch.bool) for x in masks]
    return tokens, masks
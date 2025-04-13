import os
from pathlib import Path
from typing import Iterator

import tiktoken
from . import ChatFormat, ChatMessage, Tokenizer
from tiktoken.load import load_tiktoken_bpe


class LlamaTokenizer(Tokenizer):

    special_tokens: dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, tokenizer_path: str | Path):
        
        assert os.path.isfile(tokenizer_path), tokenizer_path

        mergeable_ranks = load_tiktoken_bpe(str(tokenizer_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(tokenizer_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(self, text: str, bos: bool, eos: bool) -> list[int]:

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                text[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: list[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=set(),
                    disallowed_special=(),
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, tokens: list[int]) -> str:
        return self.model.decode(tokens)
    
    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    
class LlamaChatFormat(ChatFormat):

    def __init__(self, tokenizer: LlamaTokenizer):
        self.tokenizer = tokenizer
        self.always_mask = [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>"
        ]

    def encode_header(self, message: ChatMessage) -> tuple[list[int], list[bool]]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message.role, bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        mask = [False for _ in range(len(tokens))]
        return tokens, mask
    
    def encode_message(
        self,
        message: ChatMessage,
        add_eot: bool = True
    ) -> tuple[list[int], list[bool]]:
        
        header_tokens, header_mask = self.encode_header(message)
        message_tokens = self.tokenizer.encode(message.content, bos=False, eos=False)
        if add_eot:
            message_tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        message_mask = [True if message.role == "assistant" else False for _ in range(len(message_tokens))]

        return header_tokens + message_tokens, header_mask + message_mask

    def encode(self, dialog: list[ChatMessage], add_final_eot: bool) -> tuple[list[int], list[bool]]:
        tokens = [self.tokenizer.special_tokens["<|begin_of_text|>"]]
        mask = [False]
        for i, message in enumerate(dialog):
            add_eot = True
            if not add_final_eot and i == len(dialog) - 1:
                add_eot = False
            message_tokens, message_mask = self.encode_message(message, add_eot=add_eot)
            tokens.extend(message_tokens)
            mask.extend(message_mask)
        assert len(tokens) == len(mask)
        return tokens, mask
    
    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    def eot_id(self) -> int:
        return self.tokenizer.special_tokens["<|eot_id|>"]
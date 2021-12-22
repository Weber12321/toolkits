from dataclasses import dataclass, astuple, field
from datetime import datetime
from typing import Optional, List, Union


@dataclass
class InputTextExample:
    id_: str
    content: str
    label: Optional[str] = field(default=None)

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'input text data class'

@dataclass
class InputNumericExample:
    id_: int
    content: Union[List[Union[int, float]], int, float]
    label: Optional[Union[float, int]] = field(default=None)

    def __iter__(self):
        return iter(astuple(self))

    def __repr__(self):
        return f'input number data class'
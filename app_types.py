from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    YES_NO = 0
    INPUT = 1
    MULTIPLE_CHOICE = 2


@dataclass
class Question:
    type: QuestionType
    question: str
    response: str | dict[str, str | None]
    incompatible_with: list[str]

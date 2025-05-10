from aiogram.filters import Filter
from aiogram.types import Message, ContentType


class TextFilter(Filter):
    def __init__(self, my_text: str) -> None:
        self.my_text = my_text

    async def __call__(self, message: Message) -> bool:
        return message.text == self.my_text


class TypeFilter(Filter):
    def __init__(self, my_type: ContentType) -> None:
        self.my_type = my_type

    async def __call__(self, message: Message) -> bool:
        return message.content_type == self.my_type
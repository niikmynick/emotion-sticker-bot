import logging

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from bot.source import send_message_with_retry


start_router = Router(name="start")


@start_router.message(CommandStart())
async def start_handler(message: Message) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    logging.info(f'User {user_id} :: {username} sent command start')

    await send_message_with_retry(
        user_id,
        "Привет, я умею распознавать эмоции людей по фото / видео и находить максимально похожие стикеры\n\n"
        "Чтобы попробовать просто пришли мне файл"
    )

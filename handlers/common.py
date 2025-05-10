import logging

from aiogram import Router
from aiogram.types import Message

from bot.source import send_message_with_retry


common_router = Router(name="common")


@common_router.message()
async def message_handler(message: Message) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    logging.info(f'User {user_id} :: {username} sent a message: {message.text}')

    await send_message_with_retry(user_id, "Извини, я тебя не понимаю")

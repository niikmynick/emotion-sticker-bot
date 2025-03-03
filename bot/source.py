import logging
from asyncio import sleep

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramRetryAfter
from aiogram.client.default import DefaultBotProperties
from aiogram.types import ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup

from properties import BOT_TOKEN


bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


async def start_bot():
    logging.debug('Starting bot polling')
    await dp.start_polling(bot)


async def send_message_with_retry(
        chat_id: int,
        text: str,
        reply_markup: ReplyKeyboardMarkup | ReplyKeyboardRemove | InlineKeyboardMarkup = None,
        parse_mode: ParseMode = ParseMode.HTML
    ):
    while True:
        try:
            await bot.send_message(chat_id, text, reply_markup=reply_markup, parse_mode=parse_mode)
            return
        except Exception as e:
            if isinstance(e, TelegramRetryAfter):
                _chat_id = e.method.chat_id
                _msg = e.message
                _help = e.url
                logging.warning(f"Error description: {_msg} in chat {_chat_id} ... Seek help: {_help}")
                await sleep(e.retry_after)
                await bot.send_message(chat_id, text, reply_markup=reply_markup, parse_mode=parse_mode)
        return

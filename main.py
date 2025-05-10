import asyncio
import json
import logging.config

from aiogram import Router

from bot.source import dp, start_bot
from database.source import initialize_db, close_db
from handlers.common import common_router
from handlers.file import file_router
from handlers.start import start_router
from handlers.sticker import sticker_router
from middleware.anti_flood import AntiFloodMiddleware


def register_middleware():
    dp.message.middleware(AntiFloodMiddleware(limit=5, time_window=10, timeout=10))


def combine_routers():
    router = Router(name="main")
    router.include_routers(
        start_router,
        sticker_router,
        file_router,
        common_router,
    )

    dp.include_router(router)


async def main() -> None:
    register_middleware()
    combine_routers()

    await asyncio.gather(
        start_bot()
    )


if __name__ == "__main__":
    dict_config = json.load(open('logging.conf.json', 'r'))
    logging.config.dictConfig(
        config=dict_config
    )

    initialize_db()
    asyncio.run(main())
    close_db()

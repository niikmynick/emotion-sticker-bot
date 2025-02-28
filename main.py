import asyncio
import json
import logging.config

from aiogram import Router

from bot.source import dp, start_bot
from handlers.common import common_router
from middleware.anti_flood import AntiFloodMiddleware


def register_middleware():
    dp.message.middleware(AntiFloodMiddleware(limit=5, time_window=10, timeout=10))


def combine_routers():
    router = Router(name="main")
    router.include_routers(
        common_router,
    )

    dp.include_router(router)


async def main() -> None:
    register_middleware()
    combine_routers()

    await asyncio.run(
        start_bot()
    )


if __name__ == "__main__":
    dict_config = json.load(open('logging.conf.json', 'r'))
    logging.config.dictConfig(
        config=dict_config
    )

    asyncio.run(main())

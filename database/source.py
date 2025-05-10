import logging

from database.model import *


logger = logging.getLogger(__name__)


def create_tables():
    with db:
        db.create_tables([User, Sticker, Rating], safe=True)
        logger.info("Tables created successfully.")


def drop_tables():
    with db:
        db.drop_tables([User, Sticker, Rating], safe=True)
        logger.info("Tables dropped successfully.")


def initialize_db():
    db.connect()
    logger.info("Database connection established.")
    create_tables()
    logger.info("Database initialized.")


def close_db():
    db.close()
    logger.info("Database connection closed.")


def add_user_if_not_exists(user_id: int, user_name: str) -> None:
    if not User.select().where(User.user_id == user_id).exists():
        User.create(user_id=user_id, user_name=user_name)


def add_sticker_if_not_exists(sticker_id: str, emotion: str) -> None:
    if not Sticker.select().where(Sticker.telegram_id == sticker_id).exists():
        Sticker.create(telegram_id=sticker_id, emotion=emotion)


def get_all_stickers_by_emotion(emotion: str) -> list[Sticker]:
    return Sticker.select().where(Sticker.emotion == emotion)


if __name__ == '__main__':
    initialize_db()
    # drop_tables()
    # create_tables()
    close_db()
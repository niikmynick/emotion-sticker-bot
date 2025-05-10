from peewee import *


db = SqliteDatabase('db.db')


class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    user_id = IntegerField(primary_key=True, null=False, unique=True)
    user_name = CharField(unique=True, null=True)


class Sticker(BaseModel):
    id = IntegerField(primary_key=True, null=False, unique=True)
    telegram_id = CharField(null=False)
    emotion = CharField(null=False)


class Rating(BaseModel):
    id = IntegerField(primary_key=True, null=False, unique=True)
    user_id = ForeignKeyField(User, backref='ratings', null=False)
    file_telegram_id = CharField(null=False)
    sticker_telegram_id = ForeignKeyField(Sticker, backref='ratings', null=False)
    rating = IntegerField(null=True, constraints=[Check('rating BETWEEN 1 AND 5')])

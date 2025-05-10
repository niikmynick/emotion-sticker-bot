from aiogram.fsm.state import State, StatesGroup


class UserState(StatesGroup):
    accepting_sticker = State()
    choosing_emotion = State()
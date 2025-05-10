import json
import logging
import os

import cv2
from aiogram import Router
from aiogram.enums import ContentType
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, KeyboardButton, ReplyKeyboardRemove
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from ai.animal import get_animal_emotion
from bot.source import send_message_with_retry, save_file_by_id
from database.model import Sticker
from properties import ADMINS
from util.filters import TypeFilter, TextFilter
from util.states import UserState
from util.video import get_frames


sticker_router = Router(name="sticker")


yes_button = KeyboardButton(text="Yes")
no_button = KeyboardButton(text="No")

sticker_keyboard = ReplyKeyboardBuilder([
    [yes_button, no_button]
]).as_markup(resize_keyboard=True)


with open('human_class_indices.json', 'r') as f:
    class_indices = json.load(f)

emotions = [k for k in class_indices.keys()]

emotions_keyboard = ReplyKeyboardBuilder([
    [KeyboardButton(text=emotion)] for emotion in emotions
]).as_markup(resize_keyboard=True)


@sticker_router.message(TypeFilter(ContentType.STICKER), StateFilter(None))
async def photo_handler(message: Message, state: FSMContext) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    if user_id not in ADMINS:
        return

    sticker_id = message.sticker.file_id

    logging.info(f'Admin {user_id} :: {username} sent a sticker: {sticker_id}')

    await state.set_state(UserState.accepting_sticker)
    await state.update_data(sticker_id=sticker_id)

    disk_file_path = f'./data/downloads/{sticker_id}.webm'
    await save_file_by_id(sticker_id, disk_file_path)

    frames = get_frames(disk_file_path)

    if frames:
        await send_message_with_retry(user_id, f"Extracted {len(frames)} frames from the sticker")
    else:
        await send_message_with_retry(user_id, "No frames extracted from the sticker")
        os.remove(disk_file_path)
        return

    emotions = set()
    for frame_path in frames:
        image = cv2.imread(frame_path)
        emotion = get_animal_emotion(image)
        if emotion:
            emotions.add(emotion)

    for frame_path in frames:
        os.remove(frame_path)
    os.remove(disk_file_path)

    if emotions:
        await send_message_with_retry(user_id, f"Detected emotions: {', '.join(emotions)}", reply_markup=sticker_keyboard)
    else:
        await send_message_with_retry(user_id, "No faces detected or emotions unknown")

    await state.update_data(emotions=list(emotions))


@sticker_router.message(TextFilter(yes_button.text), StateFilter(UserState.accepting_sticker))
async def yes_handler(message: Message, state: FSMContext) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    if user_id not in ADMINS:
        return

    state_data = await state.get_data()
    sticker_id = state_data.get('sticker_id')
    emotions = state_data.get('emotions')

    if not sticker_id:
        return

    for emotion in emotions:
        Sticker(
            telegram_id=sticker_id,
            user_id=user_id,
            emotion=emotion,
        ).save()

    await state.set_data(data={})
    await send_message_with_retry(user_id, f'Saved sticker with emotions {emotions}')
    await state.set_state(None)


@sticker_router.message(TextFilter(no_button.text), StateFilter(UserState.accepting_sticker))
async def no_handler(message: Message, state: FSMContext) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    if user_id not in ADMINS:
        return

    await send_message_with_retry(user_id, 'Choose the correct emotion', reply_markup=emotions_keyboard)
    await state.set_state(UserState.choosing_emotion)


@sticker_router.message(StateFilter(UserState.choosing_emotion))
async def choose_handler(message: Message, state: FSMContext) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    if user_id not in ADMINS:
        return

    if message.text not in emotions:
        return

    state_data = await state.get_data()
    sticker_id = state_data.get('sticker_id')

    if sticker_id:
        Sticker(
            telegram_id=sticker_id,
            user_id=user_id,
            emotion=message.text,
        ).save()

    await state.set_data(data={})
    await send_message_with_retry(user_id, f'Saved sticker with emotion {message.text}', reply_markup=ReplyKeyboardRemove())
    await state.set_state(None)

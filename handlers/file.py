import logging
import os
from random import choice

import cv2

from aiogram import Router
from aiogram.enums import ContentType
from aiogram.types import Message

from ai.human import get_human_emotion
from bot.source import send_message_with_retry, save_file_by_id, send_sticker_with_retry
from database.source import get_all_stickers_by_emotion
from util.filters import TypeFilter
from util.video import get_frames


file_router = Router(name="file")


@file_router.message(TypeFilter(ContentType.PHOTO))
async def photo_handler(message: Message) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    photo_id = message.photo[-1].file_id
    logging.info(f'User {user_id} :: {username} sent a photo: {photo_id}')

    disk_file_path = f'./data/downloads/{photo_id}.jpg'
    await save_file_by_id(photo_id, disk_file_path)

    image = cv2.imread(disk_file_path)
    emotion = get_human_emotion(image)

    if emotion:
        await send_message_with_retry(user_id, f"Detected emotion: {emotion}")
    else:
        await send_message_with_retry(user_id, "No face detected or emotion unknown")

    os.remove(disk_file_path)

    stickers = get_all_stickers_by_emotion(emotion)
    if stickers:
        await send_sticker_with_retry(user_id, choice(stickers).telegram_id)



@file_router.message(TypeFilter(ContentType.VIDEO))
async def video_handler(message: Message) -> None:
    username = message.from_user.username
    user_id = message.from_user.id

    video_id = message.video.file_id
    logging.info(f'User {user_id} :: {username} sent a video: {video_id}')

    disk_file_path = f'./data/downloads/{video_id}.mp4'
    await save_file_by_id(video_id, disk_file_path)

    frames = get_frames(disk_file_path)

    if frames:
        await send_message_with_retry(user_id, f"Extracted {len(frames)} frames from the video")
    else:
        await send_message_with_retry(user_id, "No frames extracted from the video")
        os.remove(disk_file_path)
        return

    emotions = set()
    for frame_path in frames:
        image = cv2.imread(frame_path)
        emotion = get_human_emotion(image)
        if emotion:
            emotions.add(emotion)

    for frame_path in frames:
        os.remove(frame_path)

    os.remove(disk_file_path)

    if emotions:
        await send_message_with_retry(user_id, f"Detected emotions: {', '.join(emotions)}")
    else:
        await send_message_with_retry(user_id, "No faces detected or emotions unknown")

    for emotion in emotions:
        stickers = get_all_stickers_by_emotion(emotion)
        if stickers:
            await send_sticker_with_retry(user_id, choice(stickers).telegram_id)


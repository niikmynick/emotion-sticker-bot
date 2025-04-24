import os
import cv2
import logging

from properties import VIDEO_DIR_PATH, FRAME_DIR_PATH

logger = logging.getLogger(__name__)


def get_frames(filename: str) -> list:
    full_path = os.path.join(VIDEO_DIR_PATH, filename)

    if not os.path.exists(full_path):
        logger.error(f'File {full_path} does not exist')

    if not os.path.isfile(full_path):
        logger.error(f'{full_path} is not a file')

    logger.debug(f'Reading frames from {full_path}')
    temp_frames_path = FRAME_DIR_PATH + f'video_{len(os.listdir(FRAME_DIR_PATH))}/'

    capture = cv2.VideoCapture(full_path)

    frame_number = 0
    frames = []

    while True:
        success, frame = capture.read()

        if success:
            frame_path = temp_frames_path + f'frame_{frame_number}.jpg'
            frames.append(frame_path)

            cv2.imwrite(frame_path, frame)

        else:
            break

        frame_number += 1

    capture.release()

    logger.info(f'Frames read from {full_path} - {len(frames)} items')

    return frames

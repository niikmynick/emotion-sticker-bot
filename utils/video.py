import os
import cv2
import logging


logger = logging.getLogger(__name__)
videos_path = './data/video/'
frames_path = './data/frames/'


def get_frames(filename: str) -> list:
    full_path = os.path.join(videos_path, filename)

    if not os.path.exists(full_path):
        logger.error(f'File {full_path} does not exist')

    if not os.path.isfile(full_path):
        logger.error(f'{full_path} is not a file')

    logger.debug(f'Reading frames from {full_path}')
    temp_frames_path = frames_path + f'video_{len(os.listdir(frames_path))}/'

    capture = cv2.VideoCapture(full_path)

    frameNr = 0
    frames = []

    while (True):
        success, frame = capture.read()

        if success:
            frame_path = temp_frames_path + f'frame_{frameNr}.jpg'
            frames.append(frame_path)

            cv2.imwrite(frame_path, frame)

        else:
            break

        frameNr = frameNr + 1

    capture.release()

    logger.info(f'Frames read from {full_path} - {len(frames)} items')

    return frames

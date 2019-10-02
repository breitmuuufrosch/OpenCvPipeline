from random import randint
import numpy as np
import cv2


def random_mask(height, width, channels=3):
    """
    Generates a random irregular mask with lines, circles and ellipses.
    """
    img = np.zeros((height, width, channels), np.uint8)

    # Set size scale
    size = int((width + height) * 0.03)
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(3, size)
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, size)
        cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

    # Draw random ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, size)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    return 1 - img


def random_mirror(height, width, channels=3):
    use_mirror = randint(0, 10) <= 1
    size_h = height // 2

    if use_mirror:
        map_mask = np.ones((height, width, channels), dtype=np.uint8)

        # Map where the mirror is
        r = 4191
        x = size_h + randint(-1.5 * height, 1.5 * height)
        y = size_h + r + randint(-100, 100)

        p = y - r - randint(50, 80)

        map_circle = np.ones((height, width), dtype=np.uint8)
        cv2.circle(map_circle, (x, y), r, 0, -1)

        idx_circle = np.where(map_circle == 0)
        for x, y in zip(*idx_circle):
            map_mask[x, y, :] = 0

        map_mask[:p, :, :] = 0

    else:
        map_mask = np.zeros((height, width, channels), dtype=np.uint8)
        x1, x2 = 0, width
        y1 = randint(1, height)
        y2 = randint(y1 - size_h, y1 + size_h)
        thickness = randint(50, 130)
        cv2.line(map_mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

    return 1 - map_mask

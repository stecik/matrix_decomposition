import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform


def rotate_img(img: np.array, angle: float) -> np.array:
    angle = np.radians(angle)
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    center = np.array(img.shape) / 2
    offset = center - R @ center
    return affine_transform(img, R, offset=offset)


image = np.array(cv2.imread("dog.jpeg", cv2.IMREAD_GRAYSCALE))
plt.imshow(rotate_img(image, -90), cmap="gray")
plt.axis("off")
plt.show()

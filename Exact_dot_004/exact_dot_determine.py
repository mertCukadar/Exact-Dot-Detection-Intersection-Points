## importing modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tk as tk
import cv2 as cv


# loading image and convert it gray scaled image
original_image  = cv.imread("BW_image.jpeg")
# plt.imshow(original_image)
# plt.show()

# We have converted the gray image
gray_image = cv.blur(original_image , (10,10))
gray_image = cv.cvtColor(gray_image , cv.COLOR_BGR2GRAY)


plt.imshow(original_image)
plt.show()


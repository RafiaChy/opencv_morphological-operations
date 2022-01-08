import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('smarties.png', 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.array((5, 5), np.uint8)
_, threshold = cv2.threshold(img, 200, 250, cv2.THRESH_BINARY_INV)
erosion = cv2.erode(threshold, kernel, iterations=5)
dilation = cv2.dilate(threshold, kernel, iterations= 2)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(threshold, cv2.MORPH_GRADIENT, kernel)

# cv2.imshow('Threshold', threshold)
titles = ['original', 'threshold', 'erosion', 'dilation', 'opening', 'closing ', 'gradient']
images = [img, threshold, erosion, dilation, opening, closing, gradient ]

for i in range(0, len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
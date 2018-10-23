import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
img12 = mpimage.imread("im12.png")
img11 = mpimage.imread("im11.png")
plt.imshow(img12)
print("Please click")
x = plt.ginput(3)
plt.imshow(img11)
print("Please click")
y = plt.ginput(3)
print("clicked", x)
print("clicked", y)
x = np.array(x)
y = np.array(y)
transformation = np.linalg.lstsq(x, y)
print("transformations", transformation)
plt.show()



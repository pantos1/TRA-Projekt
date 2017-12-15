import matplotlib.pyplot as plt
import os
from skimage.feature import hog
from skimage import data, exposure, io, color

filename = os.path.join('C:/Users/Piotr/Documents/Studia/7. semestr/TRA/Projekt', 'vertical.bmp')
image = io.imread(filename)

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.Greys_r)
ax1.set_title('Obraz')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.Greys_r)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
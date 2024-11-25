from shared import *


def load_image(file_name: str) -> CvImg:
   img = cv.imread(os.path.join('images', file_name))
   assert img is not None, f"{file_name} does not exist"
   return img


def map_to_rgb(img: CvImg) -> CvImg:
   return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def blur_image(img: CvImg, ksize: int) -> CvImg:
   kernel = (ksize, ksize)
   blurred = cv.GaussianBlur(img, kernel, 0)
   return map_to_rgb(blurred)


def sharpen_image(img: CvImg) -> CvImg:
   kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
   sharpened = cv.filter2D(img, -1, kernel)
   return map_to_rgb(sharpened)


# original = load_image('0AA0A2.jpg')
# blurred = blur_image(original, 5)
# sharpened = sharpen_image(original)
# original = map_to_rgb(original)
#
# plt.subplot(2, 1, 1), plt.imshow(original), plt.title('Original')
# plt.subplot(2, 2, 3), plt.imshow(blurred), plt.title('Blurred')
# plt.subplot(2, 2, 4), plt.imshow(sharpened), plt.title('Sharpened')
# plt.show()

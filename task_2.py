from shared import *


def detect_face(img: CvImg) -> Sequence[Sequence[int]] | None:
   gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   classifier = cv.CascadeClassifier(
      cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
   )
   face = classifier.detectMultiScale(
      gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(130, 130)
   )

   if len(face) == 0:
      return None

   return face


# img = cv.imread(os.path.join(IMG_FOLDER_PATH, '2FC4FD.jpg'))
#
# for (x, y, w, h) in face:
#    cv.rectangle(img, (x, y), (x + w, y + h), (0x00, 0x00, 0xff), 1)
#
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()

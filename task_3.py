from shared import *
from task_1 import load_image
from task_2 import detect_face


def is_grayscale(img: CvImg, threshold=2) -> bool:
   if len(img.shape) < 3 or img.shape[2] == 1:
      return True

   # get all 3 channels
   b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

   # if all the pixels have equal colors of are very close
   return (abs(b - g) < threshold).all() and (abs(b - r) < threshold).all() and (abs(g - r) < threshold).all()


def is_portrait_or_square(img: CvImg) -> bool:
   w, h, c = img.shape
   return w <= h


def detect_eyes(img: CvImg) -> Sequence[Sequence[int]] | None:
   gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   classifier = cv.CascadeClassifier(
      cv.data.haarcascades + 'haarcascade_eye.xml'
   )
   eyes = classifier.detectMultiScale(
      gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10)
   )

   if len(eyes) == 0:
      return None

   return eyes


def validate_eyes(file_name: str, eyes: Sequence[Sequence[int]]) -> bool:
   if eyes is None:
      print(f'No eyes detected for {file_name}')
      return False
   if len(eyes) != 2:
      print(f'Wrong number of eyes detected for {file_name}')
      return False
   eyes_diff = abs(eyes[0][1] - eyes[1][1])
   if eyes_diff > 5:
      print(f'Eyes are on different levels ({eyes_diff}px) for {file_name}')
      return False
   return True


def validate_face(file_name: str, face: Sequence[Sequence[int]], img) -> bool:
   if face is None:
      print(f'No face detected for {file_name}')
      return False
   if len(face) != 1:
      print(f'No face or more than 1 face detected for {file_name} ({len(face)})')
      return False
   ratio = (face[0][2] * face[0][3]) / (img.shape[0] * img.shape[1])
   if ratio < 0.2:
      print(f'Face surface is smaller than 20% of image for {file_name}')
      return False
   if ratio > 0.5:
      print(f'Face surface is larger than 50% of image for {file_name}')
      return False
   return True


file_names = ['33C8EE.jpg', '4CA327.jpg', '7E0875.jpg', '7E0875.jpg']

for file_name in file_names:
   img = load_image(file_name)

   if is_grayscale(img, 2):
      print (f'{file_name} is grayscale')
      continue

   if not is_portrait_or_square(img):
      print(f'{file_name} is not of portrait of scale ratio')
      continue

   face = detect_face(img)
   if not validate_face(file_name, face, img):
      continue

   eyes = detect_eyes(img)
   if not validate_eyes(file_name, eyes):
      continue

   for (x, y, w, h) in eyes:
      cv.rectangle(img, (x, y), (x + w, y + h), (0x00, 0x00, 0xff), 1)

   for (x, y, w, h) in face:
      cv.rectangle(img, (x, y), (x + w, y + h), (0xff, 0x00, 0x00), 1)

   img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
   plt.imshow(img_rgb)
   plt.axis('off')
   plt.show()

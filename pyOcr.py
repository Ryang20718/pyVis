import tesserocr
from PIL import Image, ImageEnhance, ImageFilter
import sys

image = Image.open(sys.argv[1])
image = image.convert('1', dither=Image.NONE)

# enhancer = ImageEnhance.Contrast(image)
# image = enhancer.enhance(2)
# image = image.convert('1')
image.save('temp2.jpg')


#maybe separate this image to different components and then perform ocr on each of these components

print(tesserocr.image_to_text(image))  # print ocr text from image
# or
#print(tesserocr.file_to_text('temp2.jpg'))
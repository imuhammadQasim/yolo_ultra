import easyocr
import cv2

!pip install easyocr

# Initialize the EasyOCR reader
# Using the same languages and GPU setting as previously shown in the notebook
reader = easyocr.Reader(['ch_tra', 'en'], gpu=True)

# Perform OCR on the image 'EFU.jpeg'
# Assuming 'EFU.jpeg' is available in the current environment
result = reader.readtext('EFU.jpeg')

# Iterate through the result and print only the extracted text
print("--- Extracted Text ---")
for (bbox, text, prob) in result:
    print(text)
print("--------------------")
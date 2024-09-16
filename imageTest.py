import numpy as np
from skimage import io
from matplotlib import pyplot as plt

imgPath = "G:/HMVUD/anh/hue.jpg"
#đọc ảnh màu
img = io.imread(imgPath)
#in kích thước ảnh màu
print(img.shape)
# print("Ma trận ảnh màu")
# print(img)
print("Lấy ma trận ảnh màu Red")
imgRed=img[:,:,0]
print(imgRed)
io.imshow(imgRed)
plt.show()
#đọc ảnh xám
img = io.imread(imgPath, as_gray=True)
#in kích thước ảnh xám
print(img.shape)
#hiển thị ảnh xám
# io.imshow(img)
# plt.show()
#in giá trị của ma trận ảnh
# print(img)
#in vector thứ 7 trong ma trận ảnh
# print(img[6])
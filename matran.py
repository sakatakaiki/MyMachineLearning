import numpy as np
#Sinh số ngẫu nhiên
np.random.seed(123)
#tạo ma trận số nguyễn ngẫu nhiên kích thước 3x5
A1 = np.random.randint(low=-5, high=5, size=15).reshape((3,5))
print(A1)
#Tạo 1 ma trận các số thực ngẫu nhiên có kích thước 3x5, giá trị ngẫu nhiên trong [a,b]
a = float(input('Nhập a: '))
b = float(input('Nhập b: '))
if a>b:
    a,b=b,a
A2 = np.random.uniform(low=a, high=b,size=15).reshape((3,5))
print(A2)
#Tạo 1 ma trận 3x5 các số nguyên có giá trị tăng dần từ -3 đến 12
A3 = np.arange(start=-3, stop = 12, dtype = int).reshape((3,5))
print(A3)
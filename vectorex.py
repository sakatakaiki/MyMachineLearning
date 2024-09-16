import numpy as np

# Tạo 1 vector 15 phần tử gồm các số nguyên có các giá trị tăng dần
v = np.arange(15, dtype=int)
v2 = np.arange(start=20, stop=34, dtype=int)
print(v)
print(v2)

#Viết chương trình nhập 1 số nguyên n từ bàn phím
n = int(input('Nhập số nguyên n: '))
n=abs(n)
#và sinh ra 1 vector 𝑣 ∈ 𝑅𝑛: Vector các số nguyên có n phần tử và có giá trị tăng dần từ a đến b;
a =-50
b = 50
v3 = np.random.randint(low=a,high=b, size=n)
v3= np.sort(v3)
print(v3)


#- Vector các số thực có n phần tử và có giá trị tăng dần từ c đến d;
c =-10
d = 10
#v4 = np.random.uniform(low=c,high=d,size=n)
v4=(d-c)*np.random.random_sample(n) + c
v4= np.sort(v4)
print(v4)
#- Vector ngẫu nhiên các số nguyên n phần tử có giá trị trong [a,b];
v5 = np.random.randint(low=a,high=b,size=n)
print(v5)
#- Vector ngẫu nhiên các số thực n phần tử có giá trị trong [c,d].
v6 = (d-c)*np.random.random_sample(n) + c
print(v6)
#Kiểm tra kích thước
print(v6.shape)
#Kiểm tra hạng
print(v6.ndim)
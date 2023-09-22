# 경사 하강법 (Gradien Decent)

import numpy as np
import matplotlib.pyplot as plt

# 공부 시간 x와 성적 y의 넘파이 배열을 만든다.
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# 데이터의 분포를 그래프로 나타낸다.
plt.scatter(x, y)
plt.show()

# 기울기 a의 값과 절편 b의 값을 초기화.
a = 0
b = 0

# 학습률을 정한다.
lr = 0.03

# 몇 번 반복될지 설정.
epochs = 2001

# x 값이 총 몇 개인지 센다..
n = len(x)

# 경사 하강법을 시작
for i in range(epochs):
    y_pred = a * x + b
    error = y - y_pred
    
    a_diff = (2/n) * sum(-x * (error))
    b_diff = (2/n) * sum(-(error))

    a = a - lr * a_diff
    b = b - lr * b_diff
    
    if i % 100 == 0: 
        print("epoch = %.f, 기울기 = %.04f, 절편 = %.04f" % (i, a, b))
        
# 앞서 구한 최종 a 값을 기울기, b 값을 y절편에 대입해 그래프를 그린다.
y_pred = a * x + b

# 그래프를 출력.
plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.show()    
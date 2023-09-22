# 다중 선형 회귀 (Multiple Linear Regression)

import numpy as np
import matplotlib.pyplot as plt

# 공부 시간 x1과 과외 시간 x2, 성적 y의 넘파이 배열을 만든다.
x1 = np.array([2, 4, 6, 8])
x2 = np.array([0, 4, 2, 3])
y = np.array([81, 93, 91, 97])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x1, x2, y)
plt.show()

# 기울기 a의 값과 절편 b의 값을 초기화.
a1 = 0
a2 = 0
b = 0

# 학습률을 정한다.
lr = 0.01

# 몇 번 반복될지 설정..
epochs = 2001

# x 값이 총 몇 개인지 센다. x1과 x2의 수가 같으므로 x1만 센다.
n = len(x1)

# 경사 하강법 시작.
for i in range(epochs):
    y_pred = a1 * x1 + a2 * x2 + b
    error = y - y_pred
    
    a1_diff = (2/n) * sum(-x1 * (error))
    a2_diff = (2/n) * sum(-x1 * (error))
    b_diff = (2/n) * sum(-(error))

    a1 = a1 - lr * a1_diff
    a2 = a2 - lr * a2_diff
    b = b - lr * b_diff
    
    if i % 100 == 0 :
        print("epoch = %.f, 기울기1 = %.04f, 기울기2 = %.04f, 절편 = %.04f" % (i, a1, a2, b))
        
# 실제 점수와 예측된 점수를 출력.
print("실제 점수: ", y)
print("예측 점수: ", y_pred)


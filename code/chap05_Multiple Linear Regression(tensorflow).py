# 다중 선형 회귀 (Multiple Linear Regression)

import numpy as np
import matplotlib.pyplot as plt

# 텐서플로의 케라스 API에서 필요한 함수들을 불러온다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[2, 0], [4, 4], [6, 2], [8, 3]])
y = np.array([81, 93, 91, 97])

model = Sequential()

# 입력 변수가 두 개(학습시간, 과외시간)이므로 input_dim에 2를 입력.
model.add(Dense(1, input_dim = 2, activation = 'linear'))


# 오차 수정을 위해 경사 하강법(sgd)을,
# 오차의 정도를 판단하기 위해 평균 제곱 오차(mse)를 사용한다.
model.compile(optimizer = 'sgd', loss = 'mse')

# 오차를 최소화하는 과정을 2000번 반복.
model.fit(x, y, epochs = 2000)

# plt.scatter(x, y)
# plt.plot(x, model.predict(x), 'r')
# plt.show()

# 임의의 시간을 집어넣어 점수를 예측하는 모델을 테스트 해본다.
hour = 7
private_class = 4
prediction = model.predict([[hour, private_class]])
print("%.f시간을 공부하고 %.f시간의 과외를 받 경우의 예상 점수는 %.02f점입니다." % (hour, private_class, prediction))

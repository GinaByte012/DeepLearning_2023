# 평균 제곱 오차
import numpy as np


# 가상의 기울기 a와 y절편 b를 정한다.
fake_a = 3
fake_b = 76

# 공부시간 x와 성적 y의 넘파이 배열을 만든다.
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])

# y = ax + b 에 가상의 a, b 값을 대입한 결과를 출력하는 함수
def predict(x):
    return fake_a * x + fake_b

# 예측값이 들어갈 빈 list를 만든다
predict_result = []

# 모든 x값을 한 번씩 대입해 predict_result 리스르를 완성한다.
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간 = %.f, 실제점수 = %.f, 예측점수 = %.f" % (x[i], y[i], predict(x[i])))
    
# 평균 제곱 오차 함수를 각 y값에 대입해 최종 값을 구하는 함수
n = len(x)
def mse(y, y_pred):
    return (1/n) * sum((y - y_pred) ** 2)

# 평균 제곱 오차 값을 출력
print("평균 제곱 오차: " + str(mse(y, predict_result)))
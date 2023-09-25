# 딥러닝 개괄 (Deep Learning Generalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


Data_set = np.loadtxt("./../data/ThoraricSurgery3.csv", delimiter=",")

X = Data_set[:,0:16]
Y = Data_set[:, 16]


model = Sequential()    # Sequential() 함수를 model로 사용. 
# Sequential(): 딥러닝의 층들을 model.add() 함수를 사용해 간단히 추가시켜줌.
model.add(Dense(30, input_dim=16, activation='relu'))   # +1 층
model.add(Dense(1, activation='sigmoid'))   # +1 층
# Dense(): 각 층의 입력과 출력을 촘촘하게 모두 연결하라는 것.


# model.compile(): 괄호 안의 설정대로 모델을 실행
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(): 모델의 층들을 몇 번 오갈지, 그때마다 몇 개의 데이터를 사용할 것인지 정하는 함수.
history = model.fit(X, Y, epochs=30, batch_size=16)


print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))

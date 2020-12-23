# Загружаем пакет модели по Гауссу
from sklearn.naive_bayes import GaussianNB
import numpy as np

# создаем данные для теста и тренировки
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# Запускаем классификатор
model = GaussianNB()

# Тренируем модель 
model.fit(x, Y)

#Прогнозируем вывод
predicted= model.predict([[1,2],[3,4]])
print predicted

# Выводом будет [3,4]
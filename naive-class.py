# Импортируем необходимые пакеты библиотек
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Импортируем данные для теста и тренировки модели
# Здесь я нашел открытый гитхабовский датасет
training = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/iris_test.csv')


# Создаем переменные для осей X, Y, тренировки и теситрования модели 
xtrain = training.drop('Species', axis=1)
ytrain = training.loc[:, 'Species']
xtest = test.drop('Species', axis=1)
ytest = test.loc[:, 'Species']


# Запускаем Гауссовский классификатор
model = GaussianNB()

# Тренируем модель
model.fit(xtrain, ytrain)

# Прогнозируем вывод 
pred = model.predict(xtest)

# Строим матрицу путаницы
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
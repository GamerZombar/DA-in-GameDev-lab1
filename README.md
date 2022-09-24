# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил:
- Плешивцев Денис Владимирович
- РИ-210948

Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами языка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity.
- Python

![image](screenshots/Colab.png)
- Unity

![image](screenshots/Unity.png)

## Задание 2
### В разделе «ход работы» пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.

- Производим подготовку данных для работы с алгоритмом линейной
регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в
формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```
![image](screenshots/0.png)

- Определим связанные функции. Функция модели: определяет модель
линейной регрессии wx+b. Функция потерь: функция потерь
среднеквадратичной ошибки. Функция оптимизации: метод
градиентного спуска для нахождения частных производных w и b.

```py

#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+b
def model(a, b, x):
  return a*x + b

#The most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)
  return (0.5/num) * (np.square(prediction-y)).sum()

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a, b, x, y):
  num = len(x)
  prediction = model(a, b, x)
  # Update the values of A and B finding the partial derivatives of the loss function on a and b
  da = (1.0/num) * ((prediction - y)*x).sum()
  db = (1.0/num) * ((prediction - y).sum())
  a = a - Lr*da
  b = b - Lr*db
  return a, b

#Iterated function, return a and b
def iterate(a, b, x, y, times):
  for i in range(times):
    a, b = optimize(a, b, x, y)
  return a, b

```

- Инициализация и первая итерация

```py

#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

#For the first iteration, the parameter values, loses, and visualization after the iteration are displayed
a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
![image](screenshots/1.png)

- Вторая итерация

```py

a, b = iterate(a, b, x, y, 2)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
![image](screenshots/2.png)

- Третья итерация

```py

a, b = iterate(a, b, x, y, 3)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
![image](screenshots/3.png)

- Четвертая итерация

```py

a, b = iterate(a, b, x, y, 4)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
![image](screenshots/4.png)

- Пятая итерация

```py

a, b = iterate(a, b, x, y, 5)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
![image](screenshots/5.png)

- 10000-я итерация

```py

a, b = iterate(a, b, x, y, 10000)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```
![image](screenshots/10000.png)


## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.
- Величина loss с каждой итерацией стремится к нулю, так как модель постепенно "обучается". Наглядный пример — это разница в значении loss на первой и на 10000-ой итерации.

Первая:
```py

a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```

![image](screenshots/31.png)

10000-я:
```py

a, b = iterate(a, b, x, y, 10000)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```

![image](screenshots/10000.png)

### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

- За "обучение" модели как раз и отвечает параметр Lr (Learning rate). Мы можем увеличить значение этого параметра и увидеть, что он заставил модель "обучаться" быстрее, о чем свидетельствует уменьшенная величина loss.


```py

a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)

```

![image](screenshots/31.png)

И при Lr = 0.0001: 

![image](screenshots/0.0001.png)


## Выводы

Я познакомился с новыми фреймворками для программирования на Python, сделал первые шаги в Unity и узнал основные операторы языка Python на примере реализации линейной регрессии.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

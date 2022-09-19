# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Романов Вадим Юрьевич
- РИ210950
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 | // СДЕЛАЮ НА СЛЕД. ДЕНЬ (мало ли...)

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
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity
Ход работы:
- **Для Python в отчёте привести скриншоты с демонстрацией сохранения документа google.colab на свой диск с запуском программы, выводящей сообщение Hello World**

```py
print("Hello World")
```

![Screenshot_2](https://user-images.githubusercontent.com/58142149/191076474-710db566-46bc-46c4-a0fd-ef54f434ffc9.png)
____
![Screenshot_4](https://user-images.githubusercontent.com/58142149/191076526-ca53a5ce-68cb-43f0-95a7-e7ee8f5010a3.png)

*Ну и Jupyter Notebook на всякий :)*
![Screenshot_1](https://user-images.githubusercontent.com/58142149/191077046-feb19c3c-00fd-486c-bb26-d41e9ad6f239.png)
____

- **Для Unity в отчете привести скриншоты вывода сообщения Hello World на консоль**
```cs
using UnityEngine;

public class HelloWorld : MonoBehaviour
{
    void Start()
    {
        Debug.Log("Hello World");
    }
}
```
![Screenshot_5](https://user-images.githubusercontent.com/58142149/191079989-64116ab8-10b6-4a09-b75c-dde30a1a175b.png)
____


## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

![Screenshot_6](https://user-images.githubusercontent.com/58142149/191106725-f489cd37-8956-450b-9968-ffc0ad6a7aa3.png)


- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.
```py
import numpy as np
import matplotlib.pyplot as plt


def model(a, b, x):
    return a*x+b


def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5/num) * (np.square(prediction - y)).sum()


def optimize(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0/num) * ((prediction - y)*x).sum()
    db = (1.0/num) * ((prediction - y).sum())
    a -= Lr*da
    b -= Lr*db
    return a, b


def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)
    return a, b


a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

x = np.array([3, 21, 22, 34, 54, 34, 55, 67, 89, 99])
y = np.array([2, 22, 24, 65, 79, 82, 55, 130, 150, 199])

a, b = iterate(a, b, x, y, 1000000)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
plt.show()
```

- **Начать итерацию**

Поскольку итеративная модель учится на случайно подоваемом значении, то и результаты всегда разные. Ниже кучу скриншотов подтвержающее это. В моём случае модель показала лучшие результаты на 2-й и 6-й итерации, но если провернуть эти же действия по новой, то картина будет совсем иная.
![Screenshot_11](https://user-images.githubusercontent.com/58142149/191103347-2a9ae2d7-39f2-479a-b055-5074ae987636.png)
![Screenshot_12](https://user-images.githubusercontent.com/58142149/191103358-10d14abb-4066-4f6c-b452-43c32a32e15a.png)
![Screenshot_13](https://user-images.githubusercontent.com/58142149/191103364-22a68fdd-bc62-4f93-a8fe-d94d57d937ac.png)
![Screenshot_14](https://user-images.githubusercontent.com/58142149/191103638-2ba3df92-1377-407c-a58c-d2dcc450b089.png)
![Screenshot_15](https://user-images.githubusercontent.com/58142149/191103645-3d2233bd-f803-4316-88e0-4bd689da9ffb.png)
![Screenshot_16](https://user-images.githubusercontent.com/58142149/191103675-26244ae7-91e2-47b3-ba70-977f7ab8ca55.png)

Примерно на 500-й итерации потери перестают сильно "скакать", и ближе к 1000-ой итерации потери изменяются в приделах ±10, эффект рандома уже почти не ощутим.
*Это можно проверить эксперементальным путём, захламлять отчёт однотипными скриншотами не вижу смысла*

![500](https://user-images.githubusercontent.com/58142149/191100998-8c8c1367-e794-4d12-9e73-ce6a154254bf.png)
![1000](https://user-images.githubusercontent.com/58142149/191101307-e5a68acb-98ee-40c5-a867-6e5d45fc2bdb.png)
![10000](https://user-images.githubusercontent.com/58142149/191104593-2428b03f-8f65-42ac-8da8-5573e7ac6b47.png)


Ну и дальнейшие итерации, начиная с 2000-ой, ощутимого эффекта не дают: хоть десятитысячная, хоть стотысячная -- потери составляют ≈ 190. 
*От миллиона и дальше --> ≈180*

___По итогу, лучшей функцией, полученной в ходе данного эксперемента, (ссылаясь на то, что конечной должна была стать 10000-ая итерация) является___ **Y =1.74X + 0.48**

____

## Задание 3
### Ответы на вопросы
- **Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.**

- Перечисленные в этом туториале действия могут быть выполнены запуском на исполнение скрипт-файла, доступного [в репозитории](https://github.com/Den1sovDm1triy/hfss-scripting/blob/main/ScreatingSphereInAEDT.py).
- Для запуска скрипт-файла откройте Ansys Electronics Desktop. Перейдите во вкладку [Automation] - [Run Script] - [Выберите файл с именем ScreatingSphereInAEDT.py из репозитория].

```py

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.Rename("C:/Users/denisov.dv/Documents/Ansoft/SphereDIffraction.aedt", True)
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateSphere(
	[
		"NAME:SphereParameters",
		"XCenter:="		, "0mm",
		"YCenter:="		, "0mm",
		"ZCenter:="		, "0mm",
		"Radius:="		, "1.0770329614269mm"
	], 
)

```

- **Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.**

- Перечисленные в этом туториале действия могут быть выполнены запуском на исполнение скрипт-файла, доступного [в репозитории](https://github.com/Den1sovDm1triy/hfss-scripting/blob/main/ScreatingSphereInAEDT.py).
- Для запуска скрипт-файла откройте Ansys Electronics Desktop. Перейдите во вкладку [Automation] - [Run Script] - [Выберите файл с именем ScreatingSphereInAEDT.py из репозитория].

```py

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.NewProject()
oProject.Rename("C:/Users/denisov.dv/Documents/Ansoft/SphereDIffraction.aedt", True)
oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Terminal Network", "")
oDesign = oProject.SetActiveDesign("HFSSDesign1")
oEditor = oDesign.SetActiveEditor("3D Modeler")
oEditor.CreateSphere(
	[
		"NAME:SphereParameters",
		"XCenter:="		, "0mm",
		"YCenter:="		, "0mm",
		"ZCenter:="		, "0mm",
		"Radius:="		, "1.0770329614269mm"
	], 
)

```

## Выводы

P.s. Провтыкал с нумерацией заданий (ибо в методичке были ещё задания), но думаю ничего страшного, поскольку выполнено всё...

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

**Agora Hackathon**
===
In this repository you will find a solution to the case provided from Agora Hack 19-20 August 2022 by team 'Cat Troop'.

[comment]: <> <img src="https://github.com/advasileva/WalletSummary/blob/master/img/CometDefinition.jpg" width="500" />

Index
---
+ [**Problem**](#problem)  
+ [**Solution**](#solution)   
+ [**Use Cases**](#use-cases)  
  + [Подключение кошелька](#1-подключение-кошелька)  
  + [Просмотр summary](#2-просмотр-summary)  
  + [Поделиться summary](#3-поделиться-summary)  
+ [**Software architecture**](#architecture)  
  + [Backend](#бэкенд)
  + [ML model](#model)
+ [**Demonstration**](#demostration)
+ [**Further development**](#development)  

Problem
---
The goal is to develop a product matcher for product's name and description to correct labels. 

Solution
---
Using a Concurrent Neural Network our programm predicts a product's label.

Use Cases 
---

### **(1)** Подключение кошелька
**Название:** Подключение ко

**Описание:** Пользователь подключает свой кошелёк, чтобы получить доступ к функционалу системы. 

**Предусловия:** Приложение установлено. 

**Результат:** Кошелёк подключен. 

**Триггер:** Пользователь открывает приложение. 

**Успешный сценарий:**

1. Пользователь вводит токен кошелька и нажимает на кнопку “->”. 

2. Система отправляет запрос на сервер. 

3. Система получает ответ и выводит информацию о подключенном кошельке. 

### **(2)** Просмотр Summary
**Название:** Просмотр Summary

**Описание:** Пользователь просматривает Summary по своему кошельку. 

**Предусловия:** Кошелёк подключен. 

**Результат:** Пользователь просмотрел summary. 

**Триггер:** Пользователь переходит на страницу summary. 

**Успешный сценарий:**

1. Пользователь перешёл на страницу summary. 

2. Система отправляет запрос на сервер. 

3. Система получает ответ и выводит summary по кошельку. 


**Успешный сценарий:**

1. Пользователь нажал на кнопку. 

2. Система предложила одну из установленных социальных сетей / приложений с подобным функционалом в мобильном приложении.
2. Система предложила перейти на ВКонтакте / Facebook / Instagram / Twitter в веб-версии  

3. Пользователь выбран нужное приложение.

4. Система перенаправила его в выбранное приложение

Software architecture
---
### 
Our web service was developed using flask.

### Backend
Consists of connecting 

### Что мы сделали за хакатон
+ В мобильном приложении и веб-версии:
  + Подключение кошелька
  + Просмотр summary о кошельке
  + Возможность поделиться summary
+ В бекенде:
  + Подключение к API Zerion
  + Обработка полученной информации по кошельку

Демонстрация решения
---
[Демка мобильного приложения](https://github.com/advasileva/WalletSummary/blob/master/img/Demo.mp4)

[Презентация](https://github.com/advasileva/WalletSummary/blob/master/img/WalletSummaryPresentation.pdf)

Направления дальнейшей разработки
---
Проект вырос в социальную сеть для крипто-инвесторов и продолжит жизнь в другом репозитории 

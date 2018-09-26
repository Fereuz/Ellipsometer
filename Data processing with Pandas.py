# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', np.RankWarning)


def file_processing(name):
    """ Получает данные из .dat файла и записывает их в переменную DataFrame
    с именем raw_data"""

    tic = time.time()
    
    raw_data = pd.read_table('data/{}.dat'.format(name),
                             sep='\s+', header=None, skiprows=5)

    tac = time.time()
    print('Время на открытие и чтение файла = {:.3f} сек'.format(tac - tic))
    
    return raw_data


def preparing_data(raw_data):
    """ Обрабатывает данные из файла, распределяя их по координатам предметного
    столика и записывает их в DataFrame с именем data.
                
        data['wavelengths'] = float - Длины волн на которых проводились
        измерения.
        data['i'] = float - Интенсивности в i-ой Х коорденате образца.
    """
    
    tic = time.time()

    data = pd.DataFrame()

    # Получаем уникальные значения длин волн и Х координат
    wavelengths = raw_data[1].unique()
    x_coordinates = raw_data[5].unique()
    
    data['wavelengths'] = wavelengths.astype('int64')

    # Переводим координаты предметного столика в координаты образца (центрируем)
    zero_x = round(x_coordinates.astype('float64').mean(), 2)
    for i, value in enumerate(x_coordinates):
        x_coordinates[i] = round(float(value) - float(zero_x), 2)

    # Заполняем переменную data
    for i, value in enumerate(x_coordinates):
        data[value] = raw_data[3][i * len(wavelengths) : \
                                  i * len(wavelengths) + \
                                  len(wavelengths)].reset_index()\
                                  [3].astype('float64')

    tac = time.time()
    print('Время на обработку данных = {:.3f} сек'.format(tac - tic))

    return data


def recording_data_to_file(name, data):
    """ Записывает обработанные данные в .xls для работы с ними в сторонних
    программах."""

    tic = time.time()

    with pd.ExcelWriter(name + '.xls') as writer:
        data.to_excel(writer, 'Лист1', index=False)
        writer.save()

    tac = time.time()
    print('Время на запись данных = {:.3f} сек'.format(tac - tic))


def original_approximation(data):
    """ Аппроксимирует данные методом наименьших квадратов и записывает их
    в переменную approximated_data, по структуре идентичной переменной data."""

    tic = time.time()
    approximated_data = pd.DataFrame()
    approximated_data['wavelengths'] = data['wavelengths']

    # Ищем наименьшую степень полинома для лучшей аппроксимации
    for x in data.columns[1 :]:
        sums_of_squares = []

        # Увеличиваем степень полинома, для нахождения оптимального уравнения 
        for exponent in range(100):
            
            # Получаем уравнение аппроксимирующей кривой степени-exponent
            approximating_function = np.poly1d(np.polyfit(data['wavelengths'],
                                                          data[x],
                                                          exponent))

            # Записываем аппроксимационные данные во временную переменную
            temporary_appr_data = approximating_function(data['wavelengths'])

            # Сумма квадратов отклонений аппроксим. данных от оригин. данных
            sum_of_squares = 0
            for i, value in enumerate(data[x]):
                sum_of_squares = sum_of_squares + \
                                 (temporary_appr_data[i] - value) ** 2

            # Записываем сумму квадратов отклонений для этой степени полинома
            sums_of_squares.append(sum_of_squares)
            
        # Находим степень полинома при которой отклонение минимально
        best_exponent = sums_of_squares.index(min(sums_of_squares))
        
        # Получаем уравнение полинома с наименьшим отклонением
        approximating_function = np.poly1d(np.polyfit(data['wavelengths'],
                                                      data[x],
                                                      best_exponent))

        # Аппроксимируем данные и записываем их в approximated_data
        approximated_data[x] = approximating_function(data['wavelengths'])
        
    tac = time.time()
    print('Время на оригинальную аппроксимацию = {:.3f} сек'.format(tac - tic))

    return approximated_data


def new_approximation(data):
    """ Аппроксимирует данные методом наименьших квадратов по их центральной
    части, и записывает их в переменную new_approximated_data, по структуре
    идентичной переменной data.
    """
    
    tic = time.time()
    new_approximated_data = pd.DataFrame()

    # Формируем параметры для выделения центральной части
    length = len(data['wavelengths'])
    start = int(length / 4)
    end = int(length - start)

    x_x = data['wavelengths'][start : end]
    new_approximated_data['wavelengths'] = x_x
    
    # Ищем наименьшую степень полинома для лучшей аппроксимации
    for x in data.columns[1 :]:
        sums_of_squares = []

        # Получаем данные с нужной формой
        y_y = data[x][start : end]
        
        # Увеличиваем степень полинома, для нахождения оптимального уравнения 
        for exponent in range(100):
            
            # Получаем уравнение аппроксимирующей кривой степени-exponent
            approximating_function = np.poly1d(np.polyfit(x_x,
                                                          y_y,
                                                          exponent))

            # Записываем аппроксимационные данные во временную переменную
            temporary_appr_data = approximating_function(x_x)

            # Сумма квадратов отклонений аппроксим. данных от оригин. данных
            sum_of_squares = 0
            for i, value in enumerate(y_y):
                sum_of_squares = sum_of_squares + \
                                 (temporary_appr_data[i] - value) ** 2

            # Записываем сумму квадратов отклонений для этой степени полинома
            sums_of_squares.append(sum_of_squares)

        # Находим степень полинома при которой отклонение минимально
        best_exponent = sums_of_squares.index(min(sums_of_squares))
        
        # Получаем уравнение полинома с наименьшим отклонением
        approximating_function = np.poly1d(np.polyfit(x_x,
                                                      y_y,
                                                      best_exponent))

        # Аппроксимируем данные и записываем их в approximated_data
        new_approximated_data[x] = approximating_function(x_x)

    tac = time.time()
    print('Время на новую аппроксимацию = {:.3f} сек'.format(tac - tic))

    return new_approximated_data


def finding_of_peaks(approximated_data, new_approximated_data):
    """ Находит координаты пиков на аппроксимированных данных и записывает их
    в переменные origin_peaks и new_peaks, которые имеют одинаковую структуру.

    new_peaks = [[float, ..., float], ..., [...]]
        new_peaks[i] - значение(я) длин волн пиков в i-ой Х координате образца.
    """

    tic = time.time()
    
    # Пустые массивы-контейнеры.
    origin_peaks = [[] for i in range(len(approximated_data.columns) - 1)]
    new_peaks = [[] for i in range(len(new_approximated_data.columns) - 1)]

    # Ищем пики с помощью полной аппроксимации
    for x_ind, x in enumerate(approximated_data.columns[1 :]):
        for i, value in enumerate(approximated_data[x]):
            
            # Пропускает первое и последнее значение кривой.
            if i == 0 or i == len(approximated_data[x]) - 1:
                continue
            
            # Если значение больше предыдущего и следующего - пик.
            if value > approximated_data[x][i - 1] and \
               value > approximated_data[x][i + 1]:
                origin_peaks[x_ind].append(approximated_data['wavelengths'][i])

    # Ищем пики с помощью частичной аппроксимации
    for x_ind, x in enumerate(new_approximated_data.columns[1 :]):
        for i, value in enumerate(new_approximated_data[x]):
            
            # Пропускает первое и последнее значение кривой.
            if i == 0 or i == len(new_approximated_data[x]) - 1:
                continue
            
            # Если значение больше предыдущего и следующего - пик.
            if value > new_approximated_data[x].values[i - 1] and \
               value > new_approximated_data[x].values[i + 1]:
                new_peaks[x_ind].append(new_approximated_data['wavelengths']\
                                        .values[i])
      
    tac = time.time()
    print('Время на нахождение пиков = {:.3f} сек'.format(tac - tic))
    
    return origin_peaks, new_peaks


def calculation_of_delta(origin_peaks, new_peaks):
    """ Находим Дельту - процентное отклонения пика в Х координате образца от
    положения пика в центре образца.

    origin_delta = [float, ..., float]
    new_delta = [float, ..., float]
    """

    tic = time.time()
    origin_delta = []
    new_delta = []

    # Находим пики в центре образца
    origin_central_peaks = origin_peaks[int(len(origin_peaks) / 2)]
    new_central_peaks = new_peaks[int(len(new_peaks) / 2)]

    # Получаем процент отклонения пика в Х координате от центрального пика
    for i in origin_peaks:
        origin_delta.append((i[0] / origin_central_peaks[0] - 1) * 100)

    for i in new_peaks:
        new_delta.append((i[0] / new_central_peaks[0] - 1) * 100)
    
    tac = time.time()
    print('Время на расчет Дельты = {:.3f} сек'.format(tac - tic))

    return origin_delta, new_delta


def drawing_graphs(name, data, approximated_data, new_approximated_data):
    """ Строит графики экспериментальных и аппроксимированных данных"""
    
    for x in data.columns[1 :]:
        #plt.figure(figsize = (20, 15))
        plt.title('{} в точке {}'.format(name, x))
        plt.grid(True)
        plt.plot(data['wavelengths'], data['{}'.format(x)], '-',
                 data['wavelengths'], approximated_data['{}'.format(x)], '--',
                 new_approximated_data['wavelengths'], new_approximated_data['{}'.format(x)], '--')
        plt.legend(['Данные', 'Старая аппроксимация', 'Новая аппроксимация'])
        plt.xlabel('Длина волны, [нм]')
        plt.ylabel('Интенсивность')
        #plt.savefig('{} в точке {}.{}'.format(name, x, 'png'))
        plt.show()

       
def drawing_delta(name, data, origin_delta, new_delta):
    """ Строит график Дельта от Х"""
    #plt.figure(figsize = (20, 15))
    plt.title('{} - Дельта'.format(name))
    plt.grid(True)
    plt.plot(data.columns[1 :], origin_delta,
             data.columns[1 :], new_delta)
    plt.legend(['origin_delta', 'new_delta'])
    plt.xlabel('X координата, [см]')
    plt.ylabel('Дельта, [%]')
    #plt.savefig('{} - дельта от х'.format(name))
    plt.show()


def main():
##    name = input("Введите имя файла для обработки \n")
    name = '1094'

    data = preparing_data(file_processing(name))

    #recording_data_to_file(name, data)

    approximated_data = original_approximation(data)    
    new_approximated_data = new_approximation(data)

    origin_peaks, new_peaks = finding_of_peaks(approximated_data,
                                               new_approximated_data)

    origin_delta, new_delta = calculation_of_delta(origin_peaks, new_peaks)

##    drawing_graphs(name, data, approximated_data, new_approximated_data)
    drawing_delta(name, data, origin_delta, new_delta)



if __name__ == '__main__':
    main()

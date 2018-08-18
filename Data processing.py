# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', np.RankWarning)


def file_processing(name):
    """ Получает данные из .dat файла и записывает их в переменную data.

        data = [[str, str, str], ..., [...]]
            data[0][0] - длина волны.
            data[0][1] - интенсивность.
            data[0][2] - Х координата предметного столика.
    """

    tic = time.time()
    data = []

    with open('{}.dat'.format(name), 'r') as f_r:
        while True:
            line = f_r.readline()
            if line == '':
                break

            line_r = line.split('\t')

            for i in line_r:
                if len(i) == 0:
                    line_r.remove(i)

            if len(line_r) > 5:
                data.append(line_r[1::2])

    tac = time.time()
    print('Время на открытие и чтение файла = {:.6f} сек'.format(tac - tic))

    return data

	
def preparing_data(data):
    """ Разделяет данные из переменной data на переменные:

        x_coordinate = [float, ..., float] - список координат Х образца, где
            проводились измерения.

        wavelengths = [float, ..., float] - список длин волн на которых
            проводились измерения.

        prepared_data = [[float, ..., float], ..., [...]] - массив, где
            prepared_data[i] - список значений интенсивности от длины волны
            в координате x_coordinate[i].
    """

    tic = time.time()

    x_coordinate = []
    wavelengths = []

    # Выделяем и записываем из data координаты и длины волн.
    for i in data:
        if float(i[2]) not in x_coordinate:
            x_coordinate.append(float(i[2]))
        if float(i[0]) not in wavelengths:
            wavelengths.append(float(i[0]))

    # Преобразует Х координату предметного столика в Х координату образца.
    zero_x = x_coordinate[int(len(x_coordinate) / 2)]
    for i in range(len(x_coordinate)):
        x_coordinate[i] = round(x_coordinate[i] - zero_x, 2)

    # Нулевой массив-контейнер.
    prepared_data = [[0] * int(len(data) / len(x_coordinate))
                     for i in range(len(x_coordinate))]

    # Записываем данные из переменной data в массив.
    k = 0
    for i in range(len(x_coordinate)):
        for j in range(len(wavelengths)):
            prepared_data[i][j] = float(data[k + j][1])
        k = k + j + 1

    tac = time.time()
    print('Время на подготовку данных = {:.6f} сек'.format(tac - tic))
    
    return x_coordinate, wavelengths, prepared_data

	
def recording_data_to_file(name, x_coordinate, wavelengths, prepared_data):
    """ Записывает оригинальные данные из переменной prepared_data и
    wavelengths в файл .txt для удобной работы с ними в Origin.
    """

    tic = time.time()
    
    with open('{}.txt'.format(name), 'w') as f_w:
        f_w.write('Wavelength\\X\t')
        for i in x_coordinate:
            f_w.write(str(i) + '\t')
        f_w.write('\n\n')

        for i in range(len(wavelengths)):
            f_w.write(str(wavelengths[i]) + '\t')
            for j in range(len(x_coordinate)):
                f_w.write(str(prepared_data[j][i]) + '\t')
            f_w.write('\n')

    tac = time.time()
    print('Время на запись данных = {:.6f} сек'.format(tac - tic))

	
def approximation(wavelengths, prepared_data):
    """ Аппроксимирует данные и записывает их в переменную approximated_data.

        approximated_data = [[float, ..., float], ..., [...]]
    """

    tic = time.time()
    approximated_data = []

    # Ищет наименьшую степень полинома для лучшей аппроксимации.
    for y in prepared_data:
        sums_of_squares = []
        _ = []
        for exponent in range(100):
            approximating_function = np.poly1d(np.polyfit(wavelengths, y,
                                                          exponent))
            _.append(approximating_function(wavelengths))

            sum_of_squares = 0
            for i in range(len(y)):
                sum_of_squares = sum_of_squares + (_[0][i] - y[i]) ** 2

            sums_of_squares.append(sum_of_squares)
            _ = []

        # Аппроксимирует данные с использованием наименьшей степени полинома.
        approximating_function = np.poly1d(np.polyfit(wavelengths, y,
                                                      sums_of_squares.index(min(sums_of_squares)),))
        approximated_data.append(approximating_function(wavelengths))

    tac = time.time()
    print('Время на аппроксимацию данных = {:.6f} сек'.format(tac - tic))
    
    return approximated_data

	
def finding_of_peaks(wavelengths, approximated_data):
    """ Находит координаты пиков на аппроксимированных данных и записывает их
    в переменную coordinates_of_peaks.

    coordinates_of_peaks = [[float, ..., float], ..., [...]]
        coordinates_of_peaks[i] - значения длин волн пиков в x_coordinate[i]
        координате образца.
    """

    tic = time.time()
    # Пустой массив-контейнер.
    coordinates_of_peaks = [[] for i in range(len(prepared_data))]

    k = 0
    for appr_data in approximated_data:
        j = 0
        for i in appr_data:
            # Пропускает первое и последнее значение кривой.
            if i == appr_data[0] or i == appr_data[len(appr_data) - 1]:
                j = j + 1
                continue
            # Если значение больше предыдущего и следующего - пик.
            if i > appr_data[j - 1] and i > appr_data[j + 1]:
                coordinates_of_peaks[k].append(float(wavelengths[j]))
            j = j + 1
        k = k + 1

    tac = time.time()
    print('Время на нахождение пиков = {:.6f} сек'.format(tac - tic))
    
    return coordinates_of_peaks

	
def calculation_of_delta(coordinates_of_peaks):
    """ Расчитывает Дельту, последовательно деля положение пиков в разных
    Х'ах на положение центрального пика. Стоит [0] т.к. не во всех случаях
    есть больше одного пика.
    """

    tic = time.time()
    delta = []

    # Находим пики в центре образца.
    central_peaks = coordinates_of_peaks[int(len(coordinates_of_peaks) / 2)]

    for i in coordinates_of_peaks:
        delta.append((i[0] / central_peaks[0] - 1) * 100)

    tac = time.time()
    print('Время на расчет Дельты = {:.6f} сек'.format(tac - tic))

    return delta

	
def drawing_graphs(name, x_coordinate, wavelengths,
                   prepared_data, approximated_data):
    """ Строит графики экспериментальных и аппроксимированных данных."""

    for i in range(len(x_coordinate)):
        #plt.figure(figsize = (20, 15))
        plt.title('{} в точке {}'.format(name, x_coordinate[i]))
        plt.grid(True)
        plt.plot(wavelengths, prepared_data[i], '-',
                 wavelengths, approximated_data[i], '--')
        plt.legend(['Данные', 'Аппроксимация'])
        plt.xlabel('Длина волны, [нм]')
        plt.ylabel('Интенсивность')
        #plt.savefig('{} в точке {}.{}'.format(name, x_coordinate[i], 'png'))
        plt.show()
  
  
def drawing_delta(name, x_coordinate, delta):
    """ РСтроит графики расчитанной Дельты."""

    #plt.figure(figsize = (20, 15))
    plt.grid(True)
    plt.plot(x_coordinate, delta)
    plt.xlabel('X координата, [см]')
    plt.ylabel('Дельта, [%]')
    #plt.savefig('{} - дельта от х'.format(name))
    plt.show()



#----------Main----------#
name = input("Введите имя файла для обработки \n")
data = file_processing(name)
x_coordinate, wavelengths, prepared_data = preparing_data(data)
#recording_data_to_file(name, x_coordinate, wavelengths, prepared_data)
approximated_data = approximation(wavelengths, prepared_data)
coordinates_of_peaks = finding_of_peaks(wavelengths, approximated_data)
delta = calculation_of_delta(coordinates_of_peaks)
##drawing_graphs(name, x_coordinate, wavelengths,
##               prepared_data, approximated_data)
drawing_delta(name, x_coordinate, delta)

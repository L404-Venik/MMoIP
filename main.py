import argparse
import numpy as np
import skimage.io

#########################################
# Task 1
def mirror(img: np.ndarray, axis:str ='h') -> np.ndarray:
    '''
    Отразить изображение.

    Параметры:
    ----------
    img : np.ndarray 
        Входное изображение.

    axis : str - отразить относительно:\\
        h  - горизонтальной оси;
        v  - вертикальной оси;
        d  - главной диагонали;
        cd - побочной диагонали;

    Возвращает:
    ----------
    out: np.ndarray\\
        Повёрнутое изображение.
    '''

    height, width = img.shape[:2]
    if axis in ['h', 'v']:
        new_shape = (height, width,3)  # создать tuple из 2 элементов
    else:
        new_shape = (width, height,3)
        
    res = np.zeros(new_shape, dtype=float)  # массив из нулей
    
    match axis:
        case 'h': # относительно горизонтальной оси
            for i in range (0,height):
                for j in range (0,width):
                    res[i][j] = img[height - i - 1][j]
        case 'v': # относительно Вертикальной оси
            for i in range (0,height):
                for j in range (0,width):
                    res[i][j] = img[i][width - j - 1]
        case 'd': # относительно главной диагонали
            for i in range (0,height):
                for j in range (0,width):
                    res[j][i] = img[i][j]
        case 'cd': # относительно побочной диагонали
            for i in range (0,height):
                for j in range (0,width):
                    res[j][i] = img[height - i - 1][width - j - 1]
        
    return res


def extract(img: np.ndarray, left_x: int, top_y: int, width: int, height: int) -> np.ndarray:
    """
    Извлечение прямоугольного фрагмента из изображения.

    Параметры:
    ----------
    img : np.ndarray\\
        Входное изображение.
        
    left_x, top_y: int\\
        Координаты левого верхнего угла прямоугольного фрагмента.
        
    width, height: int\\
        Размеры фрагмента.

    Возвращает:
    ----------
    res: np.ndarray\\
        извлечённый из изображения img прямоугольный фрагмент 
        (с чёрными полями, если координаты фрагмента вышли за пределы img)
    """
    
    res = np.zeros((height, width,3), dtype=float) # массив из нулей
    img_height, image_width = img.shape

    for i in range (0,height):
        for j in range (0,width):
            x = left_x + j
            y = top_y + i
            if(x >= 0 and y >= 0 and x < image_width and y < img_height ): # если координаты попали в пределы изображения - копируем пиксель
                res[i][j] = img[y][x]
                
    return res


def rotate(img: np.ndarray, direction: str, angle: int) -> np.ndarray:
    """
    Поворот изображения на целое число градусов кратное 90.

    Параметры:
    ----------
    img : np.ndarray\\
        Входное изображение.
     direction : str\\
        Направление поворота:
        cw - по часовой стрелки;
        ccw - против часовой стрелки.
    angle : int\\
        Угол поворота.

    Возвращает:
    ----------
    res: np.ndarray\\
        Повёрнутое изображение.
    """

    angle %= 360;

    # повороты реализованы через последовательности отражений
    if angle == 180:
        return mirror(mirror(img, 'h'), 'v')
    
    if angle == 0:
        return img
    
    match direction:
        case 'cw': # по часовой
            if angle == 90:
                return mirror(mirror(img, 'd'), 'v')
            if angle == 270:
                return mirror(mirror(img, 'd'), 'h')
        case 'ccw': # против часовой
            if angle == 90:
                return mirror(mirror(img, 'd'), 'h')
            if angle == 270:
                return mirror(mirror(img, 'd'), 'v')
        case _:
            raise NotImplementedError('Некорректное направление. Допустимы: "cw", "ccw"')
    
    return res


def autocontrast(img: np.ndarray) -> np.ndarray:
    '''
    Привести диапазон значений яркостей изображения в диапазон [0,255].

    Параметры:
    ----------
    img : np.ndarray 
        Входное изображение.

    Возвращает:
    ----------
    out: np.ndarray\\
        Результирующее изображение.
    '''
    res = np.zeros_like(img)  # массив из нулей такой же формы и типа
    
    max_brightness = img.max()
    min_brightness = img.min()

    res = (img - min_brightness) / (max_brightness - min_brightness) # формула автоконтраста
    return res


def fixinterlace(img: np.ndarray) -> np.ndarray:
    '''
    Обнаружить и исправить интерлейсинг (артефакт чересстрочной развёртки).

    Параметры:
    ----------
    img : np.ndarray 
        Входное изображение.

    Возвращает:
    ----------
    out: np.ndarray\\
        Исправленное изображение.
    '''
    res = img.copy()
    
    def calc_variation(img: np.ndarray): 
        # подсчёт метрики "непохожести" соседних строк
        height, width = img.shape[:2]
        var = 0
        for i in range (0,height-1):
            for j in range (0,width):
                var += abs(img[i+1][j][0] - img[i][j][0]) # считаем для одного канала

        return var
    
    img_height, image_width = img.shape[:2]
    for i in range (0,img_height-1,2): # создаём изображение с поменянными местами чётными и нечётными строками
        res[[i, i + 1]] = res[[i + 1, i]]
    
    if calc_variation(res) < calc_variation(img): # оцениваем, какое из них "меньше интерлейсинг"
        return res
    else:
        return img

#########################################
# Task 2

def compute_mse (input_file_1, input_file_2):
    """
    :param input_file_1:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param input_file_2:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, MSE metric
    """

    mse = np.mean((input_file_1 - input_file_2) ** 2)
    
    return mse


def compute_psnr(input_file_1, input_file_2):
    """
    :param input_file_1:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param input_file_2:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    input_file_1 = input_file_1[:,:,0].astype(np.float64)
    input_file_2 = input_file_2[:,:,0].astype(np.float64)
    
    mse = np.mean((input_file_1 - input_file_2) ** 2)
    
    if mse == 0:
        raise ValueError("MSE is zero, images are identical")
    
    max_pixel_value = input_file_2.max()
    
    psnr = 10 * np.log10(max_pixel_value**2 / mse)
    
    return psnr # psnr for first channel 


def compute_ssim(input_file_1, input_file_2):
    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure

    h,w,d = input_file_1.shape
    ssim_values = []

    L = np.max(input_file_1)
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    for i in range(d):  # Для каждого канала (R, G, B)

        mean1 = np.mean(input_file_1[:,:,i])
        mean2 = np.mean(input_file_2[:,:,i])
        var1 = np.var(input_file_1[:, :, i])
        var2 = np.var(input_file_2[:, :, i])
        covariance = np.mean((input_file_1[:, :, i] - mean1) * (input_file_2[:, :, i]- mean2))

        ssim_value = (2 * mean1 * mean2 + c1)/(mean1 **2 + mean2 ** 2 + c1) * (2 * covariance ** 2 + c2) / (var1 ** 2 + var2 ** 2 + c2)

        ssim_values.append(ssim_value)

    # Возвращаем среднее значение SSIM для всех каналов
    return ssim_values


def median_filter(radius, input_file):
    
    h,w,_ = input_file.shape
    result = np.zeros_like(input_file)
    input_file = np.pad(input_file,((radius,radius),(radius,radius),(0,0)), mode='edge')

    for i in range(h):
        min_i = i
        max_i = i + 2 * radius + 1
        for j in range(w):
            window = input_file[min_i: max_i, j : j + 2 * radius + 1]
            result[i,j] = np.median(window) 

    return result


def gauss_filter (sigma_d, input_file):
    
    h,w,_ = input_file.shape
    result = np.zeros_like(input_file)

    radius = int(np.ceil(3 * sigma_d))
    input_file = np.pad(input_file,((radius,radius),(radius,radius),(0,0)), mode='edge')

    # Создание гауссового ядра
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    gauss_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))
    gauss_kernel /= gauss_kernel.sum()  # Нормализуем ядро

    for i in range(h):
        min_i = i
        max_i = i + 2 * radius + 1
        for j in range(w):
            window = input_file[min_i: max_i, j : j + 2 * radius + 1]
            result[i, j] = np.tensordot(window, gauss_kernel, axes=((0, 1), (0, 1)))

    return result

def bilateral_filter(sigma_d, sigma_r, input_file):

    h,w,_ = input_file.shape
    result = np.zeros_like(input_file)

    radius = int(np.ceil(3 * sigma_d))
    input_file = np.pad(input_file,((radius,radius),(radius,radius),(0,0)), mode='edge')

    # Создание гауссового ядра
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    gauss_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))

    for i in range(h):
        min_i = i
        max_i = i + 2 * radius + 1
        for j in range(w):
            window = input_file[min_i: max_i, j : j + 2 * radius + 1]

            # Вычисляем разностное ядро на основе разности интенсивности
            center_pixel = input_file[i + radius, j + radius]
            intensity_diff = window - center_pixel
            # Вычисляем ядро яркостей
            intensity_kernel = np.exp(-(np.sum(intensity_diff ** 2, axis=-1)))
            intensity_kernel /= intensity_kernel.sum()  # Нормализуем ядро яркостей
            intensity_kernel /= (2 * sigma_r**2)

            # Умножаем пространственное ядро на ядро яркостей
            bilateral_kernel = gauss_kernel * intensity_kernel
            bilateral_kernel /= bilateral_kernel.sum()  # Нормализуем

            result[i, j] = np.tensordot(window, bilateral_kernel, axes=((0, 1), (0, 1)))

    return result

from numpy.fft import fft2, fftshift
from skimage.transform import warp_polar
def compare(input_file_1, input_file_2):
    # 1. Применяем преобразование Фурье и берем амплитуду
    f_transform1 = np.abs(fftshift(fft2(input_file_1[...,0])))
    f_transform2 = np.abs(fftshift(fft2(input_file_2[...,0])))

    # 2. Переводим спектры в полярные координаты
    polar_img1 = warp_polar(f_transform1, radius=min(f_transform1.shape) // 2, scaling='log')
    polar_img2 = warp_polar(f_transform2, radius=min(f_transform2.shape) // 2, scaling='log')

    # 3. Применяем Фурье к полярному изображению по углу (оси)
    polar_ft1 = np.abs(fft2(polar_img1, axes=(0,)))
    polar_ft2 = np.abs(fft2(polar_img2, axes=(0,)))

    # 4. Сравнение
    if compute_mse(polar_ft1, polar_ft2) < 0.00001:
        return 1

    return 0


if __name__ == '__main__':  # если файл выполняется как отдельный скрипт (python script.py), то здесь будет True. Если импортируется как модуль, то False. Без этой строки весь код ниже будет выполняться и при импорте файла в виде модуля (например, если захотим использовать эти функции в другой программе), а это не всегда надо.
    # получить значения параметров командной строки
    parser = argparse.ArgumentParser(  # не все параметры этого класса могут быть нужны; читайте мануалы на docs.python.org, если интересно
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help',  # в конце списка параметров и при создании list, tuple, dict и set можно оставлять запятую, чтобы можно было удобно комментить или добавлять новые строчки без добавления и удаления новых запятых
    )
    parser.add_argument('command', help='Command description')  # add_argument() поддерживает параметры вида "-p 0.1", может сохранять их как числа, строки, включать/выключать переменные True/False ("--activate-someting"), поддерживает задание значений по умолчанию; полезные параметры: action, default, dest - изучайте, если интересно
    parser.add_argument('parameters', nargs='*')  # все параметры сохранятся в список: [par1, par2,...] (или в пустой список [], если их нет)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    # Можете посмотреть, как распознаются разные параметры. Но в самом решении лишнего вывода быть не должно.
    # print('Распознанные параметры:')
    # print('Команда:', args.command)  # между 2 выводами появится пробел
    # print('Её параметры:', args.parameters)
    # print('Входной файл:', args.input_file)
    # print('Выходной файл:', args.output_file)

    # Загрузка первого изображения
    img1= skimage.io.imread(args.input_file)  # прочитать изображение
    img1 = img1 / 255  # перевести во float и диапазон [0, 1]
    #if len(img1.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
    #    img = img1[:, :, 0]

    NeedToSave = False
    if args.command == 'mirror':
        res = mirror(img1, args.parameters[0])
        NeedToSave = True

    elif args.command == 'extract':
        left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
        res = extract(img1, left_x, top_y, width, height)
        NeedToSave = True

    elif args.command == 'rotate':
        direction = args.parameters[0]
        angle = int(args.parameters[1])
        res = rotate(img1, direction, angle)
        NeedToSave = True

    elif args.command == 'autocontrast':
        res = autocontrast(img1)
        NeedToSave = True

    elif args.command == 'fixinterlace':
        res = fixinterlace(img1)
        NeedToSave = True



    # Task 2 functions
    if args.command == 'mse':

        # Загрузка второго изображения
        img2 = skimage.io.imread(args.output_file)  # прочитать изображение
        img2 = img2 / 255  # перевести во float и диапазон [0, 1]

        res = compute_mse(input_file_1 = img1[...,0], input_file_2 = img2[...,0])
        #print('MSE = ', res)
        print(res)

    elif args.command == 'psnr':

        # Загрузка второго изображения
        img2 = skimage.io.imread(args.output_file)  # прочитать изображение
        img2 = img2 / 255  # перевести во float и диапазон [0, 1]
    
        #left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
        res = compute_psnr(input_file_1 = img1, input_file_2 = img2)
        #print('PSNR = ', res)
        print(res)

    elif args.command == 'ssim':

        # Загрузка второго изображения
        img2 = skimage.io.imread(args.output_file)  # прочитать изображение
        img2 = img2 / 255  # перевести во float и диапазон [0, 1]

        res = compute_ssim(input_file_1 = img1, input_file_2 = img2)
        #print('SSIM = ', res[0])
        print(res[0])

    elif args.command == 'median':

        radius = [int(x) for x in args.parameters]
        radius = radius[0]

        res = median_filter(radius, input_file = img1)
        NeedToSave = True


    elif args.command == 'gauss':

        sigma_d = [int(x) for x in args.parameters]
        sigma_d = sigma_d[0]

        res = gauss_filter(sigma_d, input_file = img1)
        NeedToSave = True


    elif args.command == 'bilateral':

        sigma_d, sigma_r = [int(x) for x in args.parameters]

        res = bilateral_filter(sigma_d, sigma_r, input_file = img1)
        NeedToSave = True

    elif args.command == 'compare':
        # Загрузка второго изображения
        img2 = skimage.io.imread(args.output_file)  # прочитать изображение
        img2 = img2 / 255  # перевести во float и диапазон [0, 1]
        res = compare(input_file_1 = img1, input_file_2 = img2)
        print(res)


    if(NeedToSave):
        # сохранить результат
        res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
        res = np.round(res * 255).astype(np.uint8)  # конвертация в байты
        skimage.io.imsave(args.output_file, res)


    # Ещё некоторые полезные штуки в Питоне:
    
    # l = [1, 2, 3]  # list
    # l = l + [4, 5]  # сцепить списки
    # l = l[1:-2]  # получить кусок списка (slice)
    
    # Эти тоже можно сцеплять и т.п. - читайте мануалы
    # t = (1, 2, 3)  # tuple, элементы менять нельзя, но можно сцеплять и т.д.
    # s = {1, 'a', None}  # set
    
    # d = {1: 'a', 2: 'b'}  # dictionary
    # d = dict((1, 'a'), (2, 'b'))  # ещё вариант создания
    # d[3] = 'c'  # добавить или заменить элемент словаря
    # value = d.get(3, None)  # получить (get) и удалить (pop) элемент словаря, а если его нет, то вернуть значение по умолчанию (в данном случае - None)
    # for k, v in d.items()    for k in d.keys() (или просто "in d")    for v in d.values() - варианты прохода по словарю
    
    # if 6 in l:  # проверка на вхождение в list, tuple, set, dict
    #     pass
    # else:
    #     pass

    # print(f'Какое-то число: {1.23}. \nОкруглить до сотых: {1.2345:.2f}. \nВывести переменную: {args.input_file}. \nВывести список: {[1, 2, "a", "b"]}')  # f-string позволяет создавать строки со значениями переменных
    # print('Вывести текст с чем-нибудь другим в конце вместо перевода строки.', end='1+2=3')
    # print()  # 2 раза перевести строку
    # print()
    # print('  Обрезать пробелы по краям строки и перевести всё в нижний РеГиСтР.   \n\n\n'.strip().lower())

    # import copy
    # tmp = copy.deepcopy(d)  # глубокая, полная копия объекта
    
    # Можно передавать в функцию сколько угодно параметров, если её объявить так:
    # def func(*args, **kwargs):
    # Тогда args - это list, а kwargs - это dict
    # При вызове func(1, 'b', c, par1=2, par2='d') будет: args = [1, 'b', c], а kwargs = {'par1': 2, 'par2': 'd'}.
    # Можно "раскрывать" списки и словари и подавать их в функции как последовательность параметров: some_func(*[l, i, s, t], **{'d': i, 'c': t})
    
    # p = pathlib.Path('/home/user/Documents') - создать объект Path
    # p2 = p / 'dir/file.txt' - добавить к нему ещё уровени
    # p.glob('*.png') и p.rglob('*.png') - найти все файлы нужного вида в папке, только в этой папке и рекурсивно; возвращает не list, а generator (выдаёт только по одному элементу за раз), поэтому если хотите получить сразу весь список файлов, то надо обернуть результат в "list(...)".

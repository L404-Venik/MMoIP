import argparse  # модуль (библиотека) для обработки параметров коммандной строки
import numpy as np  # модуль для работы с массивами и векторных вычислений
import skimage.io  # модуль для обработки изображений, подмодуль для чтения и записи

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


if __name__ == '__main__':  # если файл выполняется как отдельный скрипт (python script.py), то здесь будет True. Если импортируется как модуль, то False. Без этой строки весь код ниже будет выполняться и при импорте файла в виде модуля (например, если захотим использовать эти функции в другой программе), а это не всегда надо.
    # получить значения параметров командной строки
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help', 
    )
    parser.add_argument('command', help='Command description') 
    parser.add_argument('parameters', nargs='*')  # все параметры сохранятся в список: [par1, par2,...] (или в пустой список [], если их нет)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    img = skimage.io.imread(args.input_file)  # прочитать изображение
    img = img / 255  # перевести во float и диапазон [0, 1]
    #if len(img.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
    #    img = img[:, :, 0]

    # получить результат обработки для разных команд
    if args.command == 'mirror':
        res = mirror(img, args.parameters[0])

    elif args.command == 'extract':
        left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
        res = extract(img, left_x, top_y, width, height)

    elif args.command == 'rotate':
        direction = args.parameters[0]
        angle = int(args.parameters[1])
        res = rotate(img, direction, angle)

    elif args.command == 'autocontrast':
        res = autocontrast(img)

    elif args.command == 'fixinterlace':
        res = fixinterlace(img)

    # сохранить результат
    res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
    res = np.round(res * 255).astype(np.uint8)  # конвертация в байты
    skimage.io.imsave(args.output_file, res)
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

def compute_mse(input_file_1: np.ndarray, input_file_2: np.ndarray) -> float:
    """
    :param input_file_1:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        predicted image
    :param input_file_2:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, MSE metric for single channel
    """
    input_1 = input_file_1.astype(np.float64)
    input_2 = input_file_2.astype(np.float64)
    mse = np.mean((input_1 - input_2) ** 2)
    
    return mse


def compute_psnr(input_file_1: np.ndarray, input_file_2: np.ndarray) -> float:
    """
    :param input_file_1:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        predicted image
    :param input_file_2:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric for single channel
    """
    mse = compute_mse(input_file_1, input_file_2)
    
    if mse == 0:
        raise ValueError("MSE is zero, images are identical")
    
    max_pixel_value = 255
    
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr # psnr for first channel 


def compute_ssim(input_file_1: np.ndarray, input_file_2: np.ndarray) -> float:
    """
    :param input_file_1:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        predicted image
    :param input_file_2:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric for single channel
    """

    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure

    L = 255
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    input_1 = input_file_1.astype(np.float64)
    input_2 = input_file_2.astype(np.float64)


    mean1 = np.mean(input_1)
    mean2 = np.mean(input_2)
    var1 = np.var(input_1 - mean1 )
    var2 = np.var(input_2 - mean2)
    covariance = abs(np.mean((input_1 - mean1) * (input_2- mean2)))

    ssim_value = (2 * mean1 * mean2 + c1)/(mean1 **2 + mean2 ** 2 + c1) * (2 * covariance + c2) / (var1 + var2 + c2)

    return max(0, min(ssim_value, 1))


def median_filter(radius, input_file: np.ndarray):
    
    h,w = input_file.shape[:2]
    result = np.zeros_like(input_file)
    input_file = np.pad(input_file,((radius,radius),(radius,radius)), mode='edge')

    for i in range(h):
        min_i = i
        max_i = i + 2 * radius + 1
        for j in range(w):
            window = input_file[min_i: max_i, j : j + 2 * radius + 1]
            result[i,j] = np.median(window) 

    return result


def gauss_filter (sigma_d: float, input_file: np.ndarray):
    
    h,w = input_file.shape[:2]
    result = np.zeros_like(input_file)

    radius = int(np.ceil(3 * sigma_d))
    input_file = np.pad(input_file,((radius,radius),(radius,radius)), mode='edge')

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

def bilateral_filter(sigma_d: float, sigma_r: float, input_file: np.ndarray):

    h,w = input_file.shape[:2]
    result = np.zeros_like(input_file,np.float32)

    radius = int(np.ceil(3 * sigma_d))
    input_file = np.pad(input_file,((radius,radius),(radius,radius),(0,0)), mode='edge')

    # Создание гауссового ядра
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    gauss_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_d**2))

    for i in range(h):
        min_i = i
        max_i = i + 2 * radius + 1
        for j in range(w):
            window = input_file[min_i: max_i, j : j + 2 * radius + 1].astype(np.float32)

            # Вычисляем разностное ядро на основе разности интенсивности
            center_pixel = window[radius,radius]
            intensity_diff = window - center_pixel
            # Вычисляем ядро яркостей
            
            intensity_kernel = np.exp(-(np.sum(intensity_diff ** 2, axis=-1))/(3 * 2 * sigma_r**2))

            # Умножаем пространственное ядро на ядро яркостей
            bilateral_kernel = gauss_kernel * intensity_kernel
            bilateral_kernel /= bilateral_kernel.sum()  # Нормализуем

            result[i, j] = np.tensordot(window, bilateral_kernel, axes=((0, 1), (0, 1)))

    return result

from numpy.fft import fft2, fftshift
from skimage.transform import warp_polar
from scipy.ndimage import gaussian_filter
def compare(input_file_1: np.ndarray, input_file_2: np.ndarray):
    # 1. Применяем преобразование Фурье и берем амплитуду
    f_transform1 = np.abs(fftshift(fft2(input_file_1)))
    f_transform2 = np.abs(fftshift(fft2(input_file_2)))


    f_transform1 = gaussian_filter(f_transform1, sigma=1)
    f_transform2 = gaussian_filter(f_transform2, sigma=1)

    # 2. Переводим спектры в полярные координаты
    polar_img1 = warp_polar(f_transform1, radius=min(f_transform1.shape) // 2, scaling = 'log')
    polar_img2 = warp_polar(f_transform2, radius=min(f_transform2.shape) // 2, scaling = 'log')

    # 3. Применяем Фурье к полярному изображению по углу (оси)
    polar_ft1 = np.abs(fft2(polar_img1, axes=(0,)))
    polar_ft2 = np.abs(fft2(polar_img2, axes=(0,)))

    polar_ft1 = gaussian_filter(polar_ft1, sigma=1)
    polar_ft2 = gaussian_filter(polar_ft2, sigma=1)

    polar_ft1 /= np.sum(polar_ft1)
    polar_ft2 /= np.sum(polar_ft2)

    # vis1 = np.log(polar_ft1 )
    # vis1 = ((vis1 - vis1.min()) / (vis1.max() - vis1.min()) * 255).astype(np.uint8)
    # skimage.io.imsave("transform1.png", vis1)
    # vis2 = np.log(polar_ft2 )
    # vis2 = ((vis2 - vis2.min()) / (vis2.max() - vis2.min()) * 255).astype(np.uint8)
    # skimage.io.imsave("transform2.png", vis2)

    # 4. Сравнение
    mse = compute_mse(polar_ft1, polar_ft2)
    if  mse < 7e-10:
        return 1

    return 0


#########################################
# Task 3

from scipy.ndimage import convolve
def grad(sigma, img: np.ndarray) -> np.ndarray:

    radius = int(np.ceil(3 * sigma))

    # Производные двумерной функции Гаусса
    y, x = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing='ij')
    gauss_dx = -x * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    gauss_dy = -y * np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)

    # Нормализация ядер
    gauss_dx -= gauss_dx.mean()
    gauss_dy -= gauss_dy.mean()

    img = img.astype(np.float32)

    grad_x = convolve(img, gauss_dx, mode='nearest')
    grad_y = convolve(img, gauss_dy, mode='nearest')

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    gmax = grad_magnitude.max()
    if gmax > 0:
        grad_magnitude = np.round( grad_magnitude * (255.0 / gmax) )

    return grad_magnitude.astype(np.uint8)


def image_gradients(sigma, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Ядра Собеля для вычисления горизонтальной и вертикальной производных
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=float)

    # Применение фильтра Гаусса для сглаживания
    smoothed_img = gauss_filter(sigma, img).astype(np.float32)

    # Вычисление горизонтальной (Gx) и вертикальной (Gy) компонент
    Gx = convolve(smoothed_img, sobel_x, mode='reflect')
    Gy = convolve(smoothed_img, sobel_y, mode='reflect')
    
    return Gx, Gy

def nonmax(sigma, img: np.ndarray) -> np.ndarray:

    smoothed_img = gauss_filter(sigma, img)
    gx = np.gradient(smoothed_img, axis=1)
    gy = np.gradient(smoothed_img, axis=0)
    magnitude = grad(sigma, img)# np.sqrt(gx**2 + gy**2)
    
    direction = np.arctan2(gy, gx) * (180 / np.pi)
    direction = (direction + 180) % 180
    direction = np.round(direction / 45) * 45
    direction[direction == 180] = 0
    
    result = np.zeros_like(magnitude)
    magnitude = np.pad(magnitude,1, mode='constant', constant_values=0)
    rows, cols = magnitude.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i - 1, j - 1]
            if angle == 0:
                p1 = magnitude[i, j-1]
                p2 = magnitude[i, j+1]
            elif angle == 45:
                p1 = magnitude[i-1, j+1]
                p2 = magnitude[i+1, j-1]
            elif angle == 90:
                p1 = magnitude[i-1, j]
                p2 = magnitude[i+1, j]
            elif angle == 135:
                p1 = magnitude[i-1, j-1]
                p2 = magnitude[i+1, j+1]

            if magnitude[i, j] >= p1 and magnitude[i, j] >= p2:
                result[i-1, j-1] = magnitude[i, j]

    gmax = result.max()
    if gmax > 0:
        result = np.round( result * (255.0 / gmax) )
    result = np.clip(result,0,255).astype(np.uint8)

    return result 

def canny(sigma, thr_high, thr_low, img: np.ndarray) -> np.ndarray:
    # Подавление немаксимумов
    suppressed = nonmax(sigma, img) / 255
    
    # Двойной порог
    strong_edges = (suppressed >= thr_high)
    weak_edges = ((suppressed >= thr_low) & (suppressed < thr_high))
    
    # Отслеживание по гистерезису
    result = np.zeros_like(suppressed, dtype=np.uint8)
    rows, cols = suppressed.shape
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if strong_edges[i, j]:
                result[i, j] = 255
            elif weak_edges[i, j]:
                # Проверяем наличие сильных границ среди соседей
                if (strong_edges[i-1:i+2, j-1:j+2].any()):  # Если есть сильная граница среди 8-соседей
                    result[i, j] = 255

    return result


def vessels(img) -> np.ndarray:

    return ...



if __name__ == '__main__':  
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

    # Загрузка первого изображения
    img1= skimage.io.imread(args.input_file)  # прочитать изображение
    # img1 = img1 / 255  # перевести во float и диапазон [0, 1]
    #if len(img1.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
    #    img = img1[:, :, 0]

    NeedToSave = False
    match args.command :
        # Task 1 functions
        case  'mirror':
            res = mirror(img1, args.parameters[0])
            NeedToSave = True

        case 'extract':
            left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
            res = extract(img1, left_x, top_y, width, height)
            NeedToSave = True

        case 'rotate':
            direction = args.parameters[0]
            angle = int(args.parameters[1])
            res = rotate(img1, direction, angle)
            NeedToSave = True

        case 'autocontrast':
            res = autocontrast(img1)
            NeedToSave = True

        case 'fixinterlace':
            res = fixinterlace(img1)
            NeedToSave = True


        # Task 2 functions
        case 'mse':

            if len(img1.shape) == 3:  
                img1 = img1[:, :, 0]
            # Загрузка второго изображения
            img2 = skimage.io.imread(args.output_file)
            if len(img2.shape) == 3:  
                img2 = img2[:, :, 0]

            res = compute_mse(input_file_1 = img1, input_file_2 = img2)
            #print('MSE = ', res)
            print(res)

        case 'psnr':

            if len(img1.shape) == 3:  
                img1 = img1[:, :, 0]
            # Загрузка второго изображения
            img2 = skimage.io.imread(args.output_file)
            if len(img2.shape) == 3:  
                img2 = img2[:, :, 0]
        
            #left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
            res = compute_psnr(input_file_1 = img1, input_file_2 = img2)
            #print('PSNR = ', res)
            print(res)

        case 'ssim':

            if len(img1.shape) == 3:  
                img1 = img1[:, :, 0]
            # Загрузка второго изображения
            img2 = skimage.io.imread(args.output_file)
            if len(img2.shape) == 3:  
                img2 = img2[:, :, 0]

            res = compute_ssim(input_file_1 = img1, input_file_2 = img2)
            #print('SSIM = ', res[0])
            print(res)

        case 'median':

            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]
            radius = [int(x) for x in args.parameters]
            radius = radius[0]

            res = median_filter(radius, input_file = img1)
            NeedToSave = True

        case 'gauss':

            img1 = img1 / 255
            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]
            sigma_d = [float(x) for x in args.parameters]
            sigma_d = sigma_d[0]

            res = gauss_filter(sigma_d, input_file = img1)
            res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
            res = np.round(res * 255).astype(np.uint8)  # конвертация в байты
            NeedToSave = True

        case 'bilateral':

            sigma_d, sigma_r = [float(x) for x in args.parameters]
            if len(img1.shape) == 2:
                img1 = np.dstack([img1]*3)

            res = bilateral_filter(sigma_d, sigma_r, input_file = img1)
            res = np.clip(res,0,255).astype(np.uint8)
            #print(compute_mse(res[...,0],skimage.io.imread(f"bilateral_{int(sigma_d)}_{int(sigma_r)}.png")))
            NeedToSave = True

        case 'compare':

            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]
            # Загрузка второго изображения
            img2 = skimage.io.imread(args.output_file)
            if len(img2.shape) == 3:  
                img2 = img2[:, :, 0]
                
            res = compare(input_file_1 = img1, input_file_2 = img2)
            print(res) # 1 - img1 и img2 - одно и тоже изображение, 0 - иначе 


        # Task 3 functions
        case 'grad':
        
            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]
            sigma = [float(x) for x in args.parameters]
            sigma = sigma[0]

            res = grad(sigma, img1)
            NeedToSave = True

        case 'nonmax':
                
            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]
            sigma = [float(x) for x in args.parameters]
            sigma = sigma[0]

            res = nonmax(sigma, img1)
            NeedToSave = True

        case 'canny':
                
            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]

            sigma, thr_heigh, thr_low = [float(x) for x in args.parameters]

            res = canny(sigma, thr_heigh, thr_low, img1)
            NeedToSave = True

        case 'vessels':
                
            if len(img1.shape) == 3:
                img1 = img1[:, :, 0]

            res = vessels(img1)
            NeedToSave = True

    if(NeedToSave):
        # сохранить результат
        skimage.io.imsave(args.output_file, res)

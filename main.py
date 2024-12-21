from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')

# Функция для обработки изображения
def process_image(image_path):
    try:
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Изображение '{image_path}' не найдено или не может быть загружено.")
        
        results = model(image)[0]
        object_count = 0

        # Получение результатов
        classes = results.boxes.cls.cpu().numpy()

        # Подсчет количества объектов
        object_count = len(classes)

        return object_count
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return 0

# Ввод названия изображения
try:
    name = input('Введите название изображения с расширением: ')
    object_count = process_image(f'{name}')
    print(f'Прыщей обнаружено: {object_count}')
except Exception as e:
    print(f"Ошибка при вводе: {e}")
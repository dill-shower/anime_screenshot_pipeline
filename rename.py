import os
import sys
import random
import string
from pathlib import Path

def generate_random_name(length=64):
    """Генерирует случайное имя из 64 символов (буквы разного регистра и цифры)"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def rename_files(directory):
    # Счетчик переименованных файлов
    renamed_count = 0

    # Обход всех файлов в директории и поддиректориях
    for root, _, files in os.walk(directory):
        for filename in files:
            old_path = os.path.join(root, filename)
            
            # Получаем расширение файла
            file_extension = os.path.splitext(filename)[1]
            
            # Генерируем новое имя с тем же расширением
            while True:
                new_name = generate_random_name() + file_extension
                new_path = os.path.join(root, new_name)
                # Проверяем, не существует ли уже файл с таким именем
                if not os.path.exists(new_path):
                    break
            
            try:
                os.rename(old_path, new_path)
                renamed_count += 1
                print(f"Переименован: {old_path} -> {new_path}")
            except Exception as e:
                print(f"Ошибка при переименовании {old_path}: {str(e)}")

    print(f"\nВсего переименовано файлов: {renamed_count}")

def main():
    if len(sys.argv) != 2:
        print("Использование: python3 rename.py /путь/к/директории")
        sys.exit(1)

    directory = sys.argv[1]
    
    if not os.path.isdir(directory):
        print("Указанный путь не является директорией")
        sys.exit(1)

    print(f"Начинаем переименование файлов в директории: {directory}")
    rename_files(directory)

if __name__ == "__main__":
    main()

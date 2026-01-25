import os
import subprocess
import sys
import argparse
from PIL import Image
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import uuid
import threading
import time

def get_file_size(file_path):
    """Возвращает размер файла в человекочитаемом формате"""
    size_bytes = os.path.getsize(file_path)
    for unit in ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def run_with_timeout(cmd, timeout_seconds):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(timeout_seconds, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout, stderr
    finally:
        timer.cancel()

def can_extract_frame(video_path, max_attempts=3):
    """Пробует извлечь несколько кадров из видео и проверяет их корректность"""
    for attempt in range(max_attempts):
        try:
            # Получаем длительность видео
            duration_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            try:
                duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
            except:
                return video_path, False, "Не удалось определить длительность видео"

            positions = [
                0,  # начало
                1,  # первая секунда
                duration * 0.25,  # 25%
                duration * 0.5,   # 50%
                duration * 0.75   # 75%
            ]

            successful_frames = 0
            frame_results = []

            for pos in positions:
                temp_frame = f"temp_frame_{uuid.uuid4()}.png"
                
                cmd = [
                    'ffmpeg',
                    '-v', 'error',
                    '-ss', str(pos),
                    '-i', video_path,
                    '-vframes', '1',
                    '-f', 'image2',
                    '-y',
                    temp_frame
                ]

                try:
                    timeout = 120
                    returncode, stdout, stderr = run_with_timeout(cmd, timeout)

                    if returncode == 0 and os.path.exists(temp_frame):
                        # Проверяем размер файла
                        frame_size = os.path.getsize(temp_frame)
                        if frame_size < 50 * 1024:  # Меньше 50КБ
                            frame_results.append(f"Кадр на {pos:.2f}с слишком маленький: {frame_size/1024:.2f}КБ")
                            continue

                        try:
                            with Image.open(temp_frame) as img:
                                img.verify()
                                img = Image.open(temp_frame)
                                img.load()
                                successful_frames += 1
                                frame_results.append(f"Кадр на {pos:.2f}с успешно извлечён ({frame_size/1024:.2f}КБ)")
                                
                                # Если хотя бы один кадр успешно извлечён, можно прервать проверку
                                if successful_frames > 0:
                                    return video_path, True, f"Файл в порядке: успешно извлечён кадр на {pos:.2f}с ({frame_size/1024:.2f}КБ)"
                                
                        except Exception as e:
                            frame_results.append(f"Кадр на {pos:.2f}с повреждён: {str(e)}")
                    else:
                        frame_results.append(f"Не удалось извлечь кадр на {pos:.2f}с")

                except Exception as e:
                    frame_results.append(f"Ошибка при обработке кадра на {pos:.2f}с: {str(e)}")
                finally:
                    if os.path.exists(temp_frame):
                        os.remove(temp_frame)

            # Если мы дошли до этой точки и не нашли ни одного успешного кадра
            if attempt < max_attempts - 1:
                continue
            return video_path, False, f"Не удалось извлечь ни одного кадра. Детали:\n" + "\n".join(frame_results)

        except Exception as e:
            if attempt == max_attempts - 1:
                return video_path, False, f"Критическая ошибка: {str(e)}"

        time.sleep(2)

    return video_path, False, "Превышено количество попыток"
def process_directory(directory, delete_files):
    """Рекурсивно обходит директорию и проверяет видеофайлы"""
    video_extensions = ('.mkv', '.ts', '.m2ts', '.mp4')
    video_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    total_files = len(video_files)
    if total_files == 0:
        print("Видеофайлы не найдены")
        return

    print(f"Найдено файлов для проверки: {total_files}")

    log_filename = f"video_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    problematic_files = 0
    results = []

    max_workers = min(8, os.cpu_count())  # Ограничиваем количество потоков

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(can_extract_frame, video_path): video_path
                         for video_path in video_files}

        with tqdm(total=total_files, desc="Проверка файлов") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                video_path, success, message = future.result()
                file_size = get_file_size(video_path)

                if not success:
                    problematic_files += 1
                    status = f"[ПРОБЛЕМА] {message}"
                    if delete_files:
                        try:
                            os.remove(video_path)
                            status += " (Файл удален)"
                        except Exception as e:
                            status += f" (Ошибка при удалении: {str(e)})"
                else:
                    status = f"[OK] {message}"

                results.append((video_path, file_size, status))
                pbar.update(1)

    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("Отчет о проверке видеофайлов\n")
        log_file.write(f"Дата проверки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Проверяемая директория: {directory}\n")
        log_file.write(f"Режим удаления: {'Включен' if delete_files else 'Выключен'}\n")
        log_file.write("-" * 80 + "\n\n")

        results.sort(key=lambda x: x[0])

        for video_path, file_size, status in results:
            log_entry = f"Файл: {video_path}\n"
            log_entry += f"Размер: {file_size}\n"
            log_entry += f"Статус: {status}\n"
            log_entry += "-" * 80 + "\n"

            log_file.write(log_entry)
            if "[ПРОБЛЕМА]" in status:
                print(f"\nПроблема обнаружена:\n{log_entry}")

        summary = f"\nИтоговая статистика:\n"
        summary += f"Всего проверено файлов: {total_files}\n"
        summary += f"Проблемных файлов: {problematic_files}\n"
        if total_files > 0:
            summary += f"Процент проблемных файлов: {(problematic_files/total_files*100):.2f}%\n"

        log_file.write(summary)
        print(summary)
        print(f"\nПодробный отчет сохранен в файл: {log_filename}")

def main():
    parser = argparse.ArgumentParser(description="Проверка видеофайлов на целостность")
    parser.add_argument("directory", help="Путь к директории для проверки")
    parser.add_argument("--delete", action="store_true", help="Удалять проблемные файлы")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print("Указанный путь не является директорией")
        sys.exit(1)

    print(f"Начало проверки директории: {args.directory}")
    print(f"Режим удаления файлов: {'Включен' if args.delete else 'Выключен'}")
    process_directory(args.directory, args.delete)
    print("Проверка завершена")

if __name__ == "__main__":
    main()
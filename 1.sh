#!/bin/bash

# Функция для обработки сигнала SIGUSR1
pause_handler() {
    if [ "$PAUSED" = false ]; then
        echo "Скрипт приостановлен. Нажмите Enter для продолжения..."
        PAUSED=true
    fi
}

# Функция для проверки состояния паузы
check_pause() {
    if [ "$PAUSED" = true ]; then
        read -r
        PAUSED=false
        echo "Возобновление работы скрипта..."
    fi
}

# Установка обработчика сигнала
trap pause_handler SIGUSR1

# Глобальная переменная для отслеживания состояния паузы
PAUSED=false

check_files_in_directory() {
    directory="$1"
    if [ -n "$(find "$directory" -type f)" ]; then
        return 0
    else
        return 1
    fi
}

delete_directory_contents() {
    directory="$1"
    rm -rf "$directory"/*
}

extract_frames() {
    src_dir="$1"
    dst_dir="/home/user/$(basename "$src_dir")"
    
    mkdir -p "$dst_dir"
    
    python3 automatic_pipeline.py --start_stage 1 --end_stage 1 --src_dir "$src_dir" --dst_dir "$dst_dir" --image_type screenshots --image_prefix dal5 --ep_init 0 --detect_duplicate_model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --similar_thresh 0.91 --detect_duplicate_batch_size 256
    
    check_pause  # Проверка паузы после выполнения Python-скрипта
    
    if check_files_in_directory "$dst_dir"; then
        echo "Извлечение прошло успешно. Удаление содержимого исходной папки."
        delete_directory_contents "$src_dir"
        return 0
    else
        echo "Извлечение не удалось. Исходная папка не будет удалена."
        return 1
    fi
}

run_ae_multigpu() {
    result_dir="$1"
    python3 /home/user/anime_screenshot_pipeline/anime2sd/ae.py "$result_dir"
    
    # Добавляем запуск скрипта для удаления дубликатов
    echo "Запуск удаления дубликатов для директории: $result_dir"
    python3 /home/user/anime_screenshot_pipeline/anime2sd/run.py --path "$result_dir" --batch-size 256 --threshold 0.9222
}

base_dir="/home/user/Anime"

# Автоматический поиск всех директорий в base_dir
while IFS= read -r -d '' src_path; do
    check_pause  # Проверка паузы перед обработкой каждого аниме
    
    anime="$(basename "$src_path")"
    echo "Обработка: $anime"
    
    if extract_frames "$src_path"; then
        result_dir="/home/user/$anime"
        run_ae_multigpu "$result_dir"
    fi
done < <(find "$base_dir" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

echo "Обработка завершена."
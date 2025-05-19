import cv2
import os
from tqdm import tqdm
import argparse
from datetime import timedelta
import numpy as np

def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))

def analyze_videos(folder_path, show_per_video=False):
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    total_videos = len(video_files)
    
    if total_videos == 0:
        print("Нет видеофайлов в указанной папке.")
        return

    total_frames = 0
    total_duration = 0
    fps_list = []
    resolution_list = []

    if show_per_video:
        print("\nИнформация по каждому видео:")

    for video_file in tqdm(video_files, desc="Обработка видео"):
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Не удалось открыть: {video_file}")
            continue

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps if fps > 0 else 0

        total_frames += frames
        total_duration += duration
        fps_list.append(fps)
        resolution_list.append((width, height))

        if show_per_video:
            print(f"- {video_file}: {frames} кадров, {format_duration(duration)} ({duration:.2f} сек), {fps:.2f} FPS, {width}x{height}")

        cap.release()

    avg_fps = np.mean(fps_list)
    avg_resolution = np.mean(resolution_list, axis=0)

    print("\n📊 Общая статистика по датасету:")
    print(f"Количество видео: {total_videos}")
    print(f"Общая длительность: {format_duration(total_duration)} ({total_duration:.2f} сек)")
    print(f"Общее количество кадров: {total_frames}")
    print(f"Среднее количество кадров на видео: {total_frames // total_videos}")
    print(f"Средний FPS: {avg_fps:.2f}")
    print(f"Среднее разрешение: {int(avg_resolution[0])}x{int(avg_resolution[1])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ видео в папке")
    parser.add_argument("folder", help="Путь к папке с видео")
    parser.add_argument("--details", action="store_true", help="Показать статистику по каждому видео")
    args = parser.parse_args()

    analyze_videos(args.folder, args.details)

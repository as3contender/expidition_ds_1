import rasterio
from pathlib import Path
from collections import defaultdict


def check_tif_crs():
    """Проверяет CRS всех TIF файлов в папках *_Li_карты"""
    train_dir = Path("data/train")

    if not train_dir.exists():
        print(f"Папка {train_dir} не найдена")
        return

    crs_stats = defaultdict(list)
    total_files = 0

    print(f"Ищем папки в: {train_dir}")

    # Ищем все папки с маской *_Li_карты на любом уровне вложенности
    li_karty_dirs = list(train_dir.glob("*/02_*Li_карты"))
    print(f"Найдено папок Li_карты: {len(li_karty_dirs)}")

    for region_dir in li_karty_dirs:
        print(f"\n=== {region_dir.name} ===")

        # Ищем все TIF файлы в папке
        tif_files = list(region_dir.glob("*.tif"))

        if not tif_files:
            print("  TIF файлы не найдены")
            continue

        for tif_file in tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    crs = src.crs
                    width = src.width
                    height = src.height

                    print(f"  {tif_file.name}")
                    print(f"    CRS: {crs}")
                    print(f"    Размер: {width}x{height}")

                    # Группируем по CRS для статистики
                    crs_str = str(crs) if crs else "None"
                    crs_stats[crs_str].append(str(tif_file))
                    total_files += 1

            except Exception as e:
                print(f"  {tif_file.name} - ОШИБКА: {e}")

    # Выводим статистику
    print(f"\n=== СТАТИСТИКА ===")
    print(f"Всего проверено файлов: {total_files}")
    print(f"Уникальных CRS: {len(crs_stats)}")

    for crs, files in crs_stats.items():
        print(f"\nCRS: {crs}")
        print(f"  Количество файлов: {len(files)}")
        if len(files) <= 5:  # Показываем файлы если их немного
            for f in files:
                print(f"    - {Path(f).name}")
        else:
            print(f"    - {Path(files[0]).name} (и еще {len(files)-1} файлов)")


if __name__ == "__main__":
    check_tif_crs()

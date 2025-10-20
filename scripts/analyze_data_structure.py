#!/usr/bin/env python3
"""
Скрипт для анализа структуры данных в папке data/train.
Собирает информацию о:
- Количестве территорий
- Количестве объектов каждого класса
- Количестве территорий с Li_карты и Ae_немецкая данными
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
import re


def analyze_data_structure(data_path):
    """
    Анализирует структуру данных в папке data/train

    Args:
        data_path (str): Путь к папке data/train

    Returns:
        dict: Словарь с результатами анализа
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise ValueError(f"Путь {data_path} не существует")

    results = {
        "total_territories": 0,
        "territories_with_li": 0,
        "territories_with_ae": 0,
        "class_counts": Counter(),
        "territory_details": [],
        "data_type_counts": Counter(),
    }

    # Паттерн для извлечения названия территории из имени папки
    territory_pattern = re.compile(r"^\d+_(.+)_FINAL$")

    for territory_dir in sorted(data_path.iterdir()):
        if not territory_dir.is_dir():
            continue

        # Проверяем, что это папка территории (содержит _FINAL)
        territory_match = territory_pattern.match(territory_dir.name)
        if not territory_match:
            continue

        territory_name = territory_match.group(1)
        results["total_territories"] += 1

        territory_info = {
            "name": territory_name,
            "full_name": territory_dir.name,
            "has_li": False,
            "has_ae": False,
            "classes": set(),
            "data_types": set(),
        }

        # Анализируем содержимое папки территории
        for subdir in territory_dir.iterdir():
            if not subdir.is_dir():
                continue

            subdir_name = subdir.name

            # Проверяем наличие Li карт
            if "Li_карты" in subdir_name or "LI_карты" in subdir_name:
                territory_info["has_li"] = True
                results["territories_with_li"] += 1
                territory_info["data_types"].add("Li")

            # Проверяем наличие Ae немецких карт
            elif "Ae_немецкая" in subdir_name or "AE_немецкая" in subdir_name:
                territory_info["has_ae"] = True
                results["territories_with_ae"] += 1
                territory_info["data_types"].add("Ae")

            # Анализируем папку разметки
            elif "разметка" in subdir_name:
                # Ищем подпапки Li и Ae в папке разметки
                for markup_subdir in subdir.iterdir():
                    if not markup_subdir.is_dir():
                        continue

                    markup_subdir_name = markup_subdir.name

                    if markup_subdir_name in ["Li", "LI"]:
                        territory_info["data_types"].add("Li")
                        # Анализируем файлы в папке Li
                        for geojson_file in markup_subdir.glob("*.geojson"):
                            class_name = extract_class_from_filename(geojson_file.name, territory_name, "Li")
                            if class_name:
                                territory_info["classes"].add(class_name)
                                results["class_counts"][class_name] += 1

                    elif markup_subdir_name in ["Ae", "AE"]:
                        territory_info["data_types"].add("Ae")
                        # Анализируем файлы в папке Ae
                        for geojson_file in markup_subdir.glob("*.geojson"):
                            class_name = extract_class_from_filename(geojson_file.name, territory_name, "Ae")
                            if class_name:
                                territory_info["classes"].add(class_name)
                                results["class_counts"][class_name] += 1

        # Подсчитываем типы данных для территории
        for data_type in territory_info["data_types"]:
            results["data_type_counts"][data_type] += 1

        results["territory_details"].append(territory_info)

    return results


def extract_class_from_filename(filename, territory_name, data_type):
    """
    Извлекает название класса из имени файла geojson

    Args:
        filename (str): Имя файла
        territory_name (str): Название территории
        data_type (str): Тип данных (Li или Ae)

    Returns:
        str: Название класса или None
    """
    # Паттерн для извлечения класса: {Территория}_{Тип}_{Класс}.geojson
    pattern = f"{territory_name}_{data_type}_(.+)\\.geojson"
    match = re.search(pattern, filename, re.IGNORECASE)

    if match:
        return match.group(1)
    return None


def print_analysis_results(results):
    """
    Выводит результаты анализа в удобном формате
    """
    print("=" * 80)
    print("АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
    print("=" * 80)

    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего территорий: {results['total_territories']}")
    print(f"   Территорий с Li картами: {results['territories_with_li']}")
    print(f"   Территорий с Ae немецкими картами: {results['territories_with_ae']}")

    print(f"\n🗺️ ТИПЫ ИСХОДНЫХ ДАННЫХ:")
    for data_type, count in results["data_type_counts"].most_common():
        print(f"   {data_type}: {count} территорий")

    print(f"\n🏛️ КЛАССЫ ОБЪЕКТОВ (всего {len(results['class_counts'])} уникальных классов):")
    for class_name, count in results["class_counts"].most_common():
        print(f"   {class_name}: {count} территорий")

    print(f"\n📍 ДЕТАЛЬНАЯ ИНФОРМАЦИЯ ПО ТЕРРИТОРИЯМ:")
    for territory in results["territory_details"]:
        data_types_str = ", ".join(sorted(territory["data_types"])) if territory["data_types"] else "Нет данных"
        classes_str = ", ".join(sorted(territory["classes"])) if territory["classes"] else "Нет классов"
        print(f"   {territory['name']}:")
        print(f"     - Типы данных: {data_types_str}")
        print(f"     - Классы объектов: {classes_str}")


def save_results_to_json(results, output_path):
    """
    Сохраняет результаты анализа в JSON файл
    """
    # Конвертируем Counter объекты в обычные словари для JSON
    json_results = {
        "total_territories": results["total_territories"],
        "territories_with_li": results["territories_with_li"],
        "territories_with_ae": results["territories_with_ae"],
        "class_counts": dict(results["class_counts"]),
        "data_type_counts": dict(results["data_type_counts"]),
        "territory_details": [],
    }

    # Конвертируем set объекты в списки для JSON
    for territory in results["territory_details"]:
        territory_json = {
            "name": territory["name"],
            "full_name": territory["full_name"],
            "has_li": territory["has_li"],
            "has_ae": territory["has_ae"],
            "classes": list(territory["classes"]),
            "data_types": list(territory["data_types"]),
        }
        json_results["territory_details"].append(territory_json)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Результаты сохранены в: {output_path}")


def main():
    """
    Основная функция скрипта
    """
    # Путь к папке с данными
    data_path = Path(__file__).parent.parent / "data" / "train"

    if not data_path.exists():
        print(f"❌ Ошибка: Папка {data_path} не найдена!")
        print("Убедитесь, что скрипт запускается из корневой папки проекта.")
        return

    try:
        print("🔍 Анализируем структуру данных...")
        results = analyze_data_structure(data_path)

        # Выводим результаты
        print_analysis_results(results)

        # Сохраняем результаты в JSON
        output_path = Path(__file__).parent / "data_analysis_results.json"
        save_results_to_json(results, output_path)

        print(f"\n✅ Анализ завершен успешно!")

    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        raise


if __name__ == "__main__":
    main()

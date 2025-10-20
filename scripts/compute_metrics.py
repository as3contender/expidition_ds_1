#!/usr/bin/env python3
"""
Минимальный скрипт расчета итоговой метрики F2 (final_score) для полигональных объектов.

Вход: два GeoJSON FeatureCollection с разметкой (predictions, ground truth)
Ожидается объединенный формат: у features в properties есть поля:
- region_name (обязательно)
- sub_region_name (опционально)
- class_name (обязательно)

Скрипт:
- разбивает объекты по источникам (region/sub_region) и классам
- считает IoU с гибридным допуском аннотации
- вычисляет F2 по классам, агрегирует по весам классов и усредняет по источникам

Вывод: одно число (final_score) в stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from decimal import Decimal, getcontext

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    from shapely import prepared
except ImportError:
    raise ImportError("Требуется библиотека shapely. Установите её: pip install shapely")


logger = logging.getLogger(__name__)


def round_to_precision(value: float, precision: int = 10) -> float:
    """
    Точное округление float значения до заданного количества знаков после запятой.
    Использует Decimal для избежания погрешностей округления с плавающей точкой.
    
    Args:
        value: Значение для округления
        precision: Количество знаков после запятой (по умолчанию 10)
        
    Returns:
        Округленное значение типа float
    """
    if not isinstance(value, float):
        return value
    
    # Устанавливаем точность для Decimal
    getcontext().prec = 28
    
    # Преобразуем в Decimal, округляем и возвращаем как float
    decimal_obj = Decimal(str(value))
    precision_format = Decimal('0.' + '0' * precision)
    rounded_decimal = decimal_obj.quantize(precision_format)
    return float(rounded_decimal)


# Веса классов (латинские ключи)
CLASS_WEIGHTS: Dict[str, float] = {
    "selishcha": 5.0,
    "kurgany": 4.0,
    "karavannye_puti": 3.0,
    "fortifikatsii": 2.0,
    "gorodishcha": 2.0,
    "arkhitektury": 2.0,
    "pashni": 1.0,
    "dorogi": 1.0,
    "yamy": 1.0,
    "inoe": 0.5,
    "mezha": 0.5,
    # "artefakty_lidara": 0.5 # этот класс не участвует в валидации
}


# Маппинг названий на стандартизированные
CLASS_NAME_MAPPING: Dict[str, str | None] = {
    # Селища
    "селище": "selishcha", "Селище": "selishcha", "селища": "selishcha", "Селища": "selishcha",
    # Пашни
    "пашня": "pashni", "Пашня": "pashni", "пашни": "pashni", "Пашни": "pashni",
    "пахота": "pashni", "pashnya": "pashni", "Pashnya": "pashni",
    "глубин": "pashni", "Глубин": "pashni",
    # Курганы
    "распаханные курганы": "kurgany", "курган": "kurgany", "Курган": "kurgany",
    "курганы": "kurgany", "Курганы": "kurgany", "kurgani": "kurgany", "Kurgani": "kurgany",
    # Караванные пути
    "караванные": "karavannye_puti", "Караванные": "karavannye_puti",
    "караванные пути": "karavannye_puti", "Караванные пути": "karavannye_puti",
    "пути": "karavannye_puti", "Пути": "karavannye_puti",
    # Фортификации
    "фортификация": "fortifikatsii", "Фортификация": "fortifikatsii",
    "фортификации": "fortifikatsii", "Фортификации": "fortifikatsii",
    # Городища
    "городище": "gorodishcha", "Городище": "gorodishcha",
    "городища": "gorodishcha", "Городища": "gorodishcha",
    "gorodishche": "gorodishcha", "Gorodishche": "gorodishcha",
    # Архитектуры
    "архитектура": "arkhitektury", "Архитектура": "arkhitektury",
    "архитектуры": "arkhitektury", "Архитектуры": "arkhitektury",
    # Дороги
    "дорога": "dorogi", "Дорога": "dorogi", "дороги": "dorogi", "Дороги": "dorogi",
    "dorogi": "dorogi", "Dorogi": "dorogi",
    # Ямы
    "яма": "yamy", "Яма": "yamy", "ямы": "yamy", "Ямы": "yamy",
    # Межа
    "межа": "mezha", "Межа": "mezha",
    # Артефакты лидара (игнор)
    "артефакты лидара": None, "Артефакты лидара": None,
    "артефакты_лидара": None, "Артефакты_лидара": None,
    "лидара": None, "артефакт": None, "Артефакт": None,
    # Иное
    "иное": "inoe", "Иное": "inoe", "inoe": "inoe", "Inoe": "inoe",
}


def split_merged_by_source_and_class(geojson_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    features = geojson_data.get("features", [])
    for feature in features:
        props = feature.get("properties", {}) or {}
        region_name = props.get("region_name", "default_source")
        sub_region_name = props.get("sub_region_name", "")
        if sub_region_name and sub_region_name.strip():
            source_name = f"{region_name}__{sub_region_name}"
        else:
            source_name = region_name
        class_name = props.get("class_name", "unknown")

        if source_name not in result:
            result[source_name] = {}
        if class_name not in result[source_name]:
            result[source_name][class_name] = {"type": "FeatureCollection", "features": []}

        result[source_name][class_name]["features"].append(feature)
    return result


class F2Calculator:
    def calculate_final_f2(
        self,
        predictions: Dict[str, Dict[str, Dict]],
        ground_truth: Dict[str, Dict[str, Dict]],
        tau: float = 0.5,
        pixel_resolution: float = 1.0,
        min_area_threshold_pixels: float = 3.0,
        annotation_tolerance_pixels: float = 2.0,
    ) -> float:
        self._validate_inputs(predictions, ground_truth, tau, pixel_resolution)

        pred_sources = set(predictions.keys())
        gt_sources = set(ground_truth.keys())

        if len(gt_sources) == 0:
            logger.warning("Ground truth не содержит источников — F2 метрика равна 0")
            return 0.0

        missing_in_predictions = gt_sources - pred_sources
        if missing_in_predictions:
            logger.warning(
                "Предсказания отсутствуют для %d GT-источников: %s",
                len(missing_in_predictions),
                sorted(missing_in_predictions)
            )

        weighted_f2_scores: List[float] = []
        processed_sources: List[str] = []

        for source_name in sorted(gt_sources):
            pred_source = predictions.get(source_name, {})

            if source_name not in predictions:
                logger.debug(
                    "Источник '%s' отсутствует в предсказаниях, используется пустой набор",
                    source_name
                )

            source_f2 = self._calculate_source_weighted_f2(
                pred_source,
                ground_truth[source_name],
                tau,
                pixel_resolution,
                min_area_threshold_pixels,
                annotation_tolerance_pixels,
            )
            weighted_f2_scores.append(source_f2)
            processed_sources.append(source_name)

        if weighted_f2_scores:
            final_score = float(np.mean(weighted_f2_scores))
            logger.info(
                "final_score рассчитан по %d источникам: %s",
                len(weighted_f2_scores),
                list(zip(processed_sources, [round(score, 4) for score in weighted_f2_scores]))
            )
            return final_score

        logger.warning("Не удалось рассчитать взвешенные F2 метрики по источникам")
        return 0.0

    def _calculate_source_weighted_f2(
        self,
        pred_source: Dict[str, Dict],
        gt_source: Dict[str, Dict],
        tau: float,
        pixel_resolution: float,
        min_area_threshold_pixels: float,
        annotation_tolerance_pixels: float,
    ) -> float:
        # собрать множество классов (с учетом имен в GeoJSON name при наличии)
        all_classes = set()
        for class_name in pred_source.keys():
            norm = CLASS_NAME_MAPPING.get(class_name, class_name)
            if norm is not None:
                all_classes.add(norm)
        for class_name in gt_source.keys():
            norm = CLASS_NAME_MAPPING.get(class_name, class_name)
            if norm is not None:
                all_classes.add(norm)

        total_weight = 0.0
        total_weighted_f2 = 0.0

        for class_name in all_classes:
            if class_name not in CLASS_WEIGHTS:
                if class_name is None or class_name == "artefakty_lidara":
                    continue
                raise ValueError(
                    f"Неизвестный класс '{class_name}'. Допустимые: {list(CLASS_WEIGHTS.keys())}"
                )
            pred_class = pred_source.get(class_name, {"features": []})
            gt_class = gt_source.get(class_name, {"features": []})

            f2 = self._calculate_class_f2(
                pred_class,
                gt_class,
                tau,
                pixel_resolution,
                min_area_threshold_pixels,
                annotation_tolerance_pixels,
            )

            weight = CLASS_WEIGHTS[class_name]
            total_weight += weight
            total_weighted_f2 += weight * f2

        return total_weighted_f2 / total_weight if total_weight > 0 else 0.0

    def _calculate_class_f2(
        self,
        pred_class: Dict,
        gt_class: Dict,
        tau: float,
        pixel_resolution: float,
        min_area_threshold_pixels: float,
        annotation_tolerance_pixels: float,
    ) -> float:
        pred_polygons = self._extract_polygons(pred_class)
        gt_polygons = self._extract_polygons(gt_class)

        pred_polygons = [
            p for p in pred_polygons if self._is_above_min_area(p, pixel_resolution, min_area_threshold_pixels)
        ]
        gt_polygons = [
            g for g in gt_polygons if self._is_above_min_area(g, pixel_resolution, min_area_threshold_pixels)
        ]

        if not pred_polygons and not gt_polygons:
            return 0.0

        iou_no_buffer = self._calculate_iou_matrix(pred_polygons, gt_polygons)
        if np.any(iou_no_buffer >= 0.999):
            iou_matrix = iou_no_buffer
        else:
            gt_buffered = [
                self._apply_annotation_tolerance(g, pixel_resolution, annotation_tolerance_pixels)
                for g in gt_polygons
            ]
            iou_matrix = self._calculate_iou_matrix(pred_polygons, gt_buffered)

        tp, fp, fn = self._calculate_tp_fp_fn(iou_matrix, tau)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return self._f2(precision, recall)

    def _f2(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 5 * (precision * recall) / (4 * precision + recall)

    def _extract_polygons(self, geojson_data: Dict) -> List[Polygon]:
        polygons: List[Polygon] = []
        for feature in geojson_data.get("features", []):
            geometry = feature.get("geometry")
            if not geometry:
                continue
            if geometry.get("type") != "Polygon":
                continue
            coordinates = geometry.get("coordinates", [])
            if not coordinates:
                continue
            try:
                poly = Polygon(coordinates[0], coordinates[1:] if len(coordinates) > 1 else None)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
            except Exception:
                continue
        return polygons

    def _is_above_min_area(self, polygon: Polygon, pixel_resolution: float, min_area_threshold_pixels: float) -> bool:
        min_area_sq_meters = min_area_threshold_pixels * (pixel_resolution ** 2)
        return polygon.area >= min_area_sq_meters

    def _apply_annotation_tolerance(
        self,
        polygon: Polygon,
        pixel_resolution: float,
        annotation_tolerance_pixels: float,
    ) -> Polygon:
        try:
            buffer_size = annotation_tolerance_pixels * pixel_resolution
            buffered = polygon.buffer(buffer_size)
            if not buffered.is_valid:
                buffered = make_valid(buffered)
            return buffered if buffered.is_valid else polygon
        except Exception:
            return polygon

    def _calculate_iou_matrix(self, pred_polygons: List[Polygon], gt_polygons: List[Polygon]) -> np.ndarray:
        if not pred_polygons or not gt_polygons:
            return np.zeros((len(pred_polygons), len(gt_polygons)))
        iou = np.zeros((len(pred_polygons), len(gt_polygons)))
        prepared_gt = [prepared.prep(g) for g in gt_polygons]
        for i, p in enumerate(pred_polygons):
            for j, (g, prep_g) in enumerate(zip(gt_polygons, prepared_gt)):
                if not prep_g.intersects(p):
                    continue
                iou[i, j] = self._polygon_iou(p, g)
        return iou

    def _polygon_iou(self, a: Polygon, b: Polygon) -> float:
        try:
            if not a.is_valid or not b.is_valid:
                return 0.0
            inter = a.intersection(b)
            union = a.union(b)
            if union.area == 0:
                return 0.0
            return inter.area / union.area
        except Exception:
            return 0.0

    def _calculate_tp_fp_fn(self, iou_matrix: np.ndarray, tau: float) -> Tuple[int, int, int]:
        if iou_matrix.size == 0:
            return 0, 0, 0
        num_pred, num_gt = iou_matrix.shape
        used_pred = set()
        used_gt = set()
        tp = 0
        while True:
            masked = iou_matrix.copy()
            for i in used_pred:
                masked[i, :] = 0
            for j in used_gt:
                masked[:, j] = 0
            max_iou = np.max(masked)
            if max_iou < tau:
                break
            i, j = np.unravel_index(np.argmax(masked), masked.shape)
            used_pred.add(i)
            used_gt.add(j)
            tp += 1
        fp = num_pred - tp
        fn = num_gt - tp
        return tp, fp, fn

    def _validate_inputs(self, predictions: Dict, ground_truth: Dict, tau: float, pixel_resolution: float) -> None:
        assert 0 < tau < 1, "Порог IoU должен быть в диапазоне (0, 1)"
        assert pixel_resolution > 0, "Разрешение пикселя должно быть положительным"
        assert isinstance(predictions, dict) and isinstance(ground_truth, dict), "Входные данные должны быть словарями"
        assert len(ground_truth) > 0, "Ground truth не должен быть пустым"
        if len(predictions) == 0:
            logger.warning("Предсказания не содержат источников — итоговый F2 будет равен 0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Расчет итоговой F2 метрики (final_score)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", "-p", required=True, help="GeoJSON с предсказаниями")
    parser.add_argument("--ground-truth", "-g", required=True, help="GeoJSON с разметкой (GT)")
    parser.add_argument("--tau", type=float, default=0.5, help="Порог IoU")
    parser.add_argument("--pixel-resolution", type=float, default=1.0, help="Метры на пиксель")
    parser.add_argument(
        "--min-area-threshold",
        type=float,
        default=3.0,
        help="Минимальная площадь объекта в пикселях",
    )
    parser.add_argument(
        "--annotation-tolerance",
        type=float,
        default=2.0,
        help="Допуск аннотации в пикселях",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pred_path = Path(args.predictions)
    gt_path = Path(args.ground_truth)
    if not pred_path.exists():
        raise FileNotFoundError(f"Файл предсказаний не найден: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Файл ground truth не найден: {gt_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions_geojson = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_geojson = json.load(f)

    predictions_struct = split_merged_by_source_and_class(predictions_geojson)
    gt_struct = split_merged_by_source_and_class(gt_geojson)

    calculator = F2Calculator()
    final_score = calculator.calculate_final_f2(
        predictions=predictions_struct,
        ground_truth=gt_struct,
        tau=args.tau,
        pixel_resolution=args.pixel_resolution,
        min_area_threshold_pixels=args.min_area_threshold,
        annotation_tolerance_pixels=args.annotation_tolerance,
    )

    # Единственный вывод
    print(f"{round_to_precision(final_score):.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



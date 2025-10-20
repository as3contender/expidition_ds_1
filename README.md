# Expedition DS 1

Мультиклассовая сегментация археологических объектов по лидара-растрам. Репозиторий включает подготовку сплитов, обучение U-Net, локальную валидацию и инференс в GeoJSON.

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd expidition_ds_1
```
2. Создайте и активируйте окружение, установите зависимости:
```bash
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Структура данных

Ожидается дерево территорий вида `XXX_ИМЯ_FINAL` в `data/train` (по умолчанию), внутри:
- `02_*_Li_карты/` — каналы hillshade (см. паттерны в `configs/default.yaml:data.hillshade_channels`)
- `06_*_разметка/Li/*.geojson` — векторная разметка классов

Пример ключей в конфиге `configs/default.yaml`:
- `data.regions_root` — корень с папками `*_FINAL`
- `data.hillshade_channels` — список паттернов каналов
- `data.labels_subpath` — паттерн до GeoJSON
- `data.classes` — словарь классов и весов
- `data.tile_size`, `data.tile_stride` — слайдинг окон
- `train.batch_size`, `train.epochs`, `train.lr`, `train.weight_decay`
- `train.limit_regions`, `train.limit_val_regions` — ограничение числа регионов для быстрого теста
- `log.out_dir` — директория для артефактов (по умолчанию `runs/default`)

## Логика пайплайна

1) Построение индекса территорий (`src/build_splits.py`):
- Ищутся территории по `data.regions_root/*_FINAL`
- По каждому региону собираются пути каналов согласно `data.hillshade_channels`
- Собираются пути GeoJSON согласно `data.labels_subpath`
- Формируются файлы:
  - `runs/default/index.csv`
  - `runs/default/train_regions.csv`
  - `runs/default/val_regions.csv`

2) Датасет (`src/dataset_sliding.py`):
- Для каждого региона выбирается референсный канал (первый существующий)
- Слайдинг окон `tile_size/stride`
- Для каждого окна по всем доступным каналам читаются тайлы в общие геограницы окна, нормализация per-channel (2–98 перцентили)
- Маска растеризуется из GeoJSON в CRS растра с учётом буферов `data.class_buffer_m`

3) Модель (`src/model_unet.py`):
- U-Net из `segmentation_models_pytorch`, входных каналов = длина `data.hillshade_channels`, выход — по числу классов

4) Обучение (`src/train.py`):
- Лоссы: Focal + Dice (multilabel)
- Валидация: метрика F2 (взвешенная) из `src/metrics_local.py`
- Логи/артефакты:
  - `runs/default/metrics.csv`
  - `runs/default/learning_curve.png`
  - `runs/default/overlays/*.png` — оверлеи предсказаний
  - `runs/default/best.pt` — лучший чекпойнт

5) Инференс (`src/infer_to_geojson.py`):
- Сшивка скользящих окон по региону, порогование по `thresholds` из конфига
- Полигонализация предсказаний в CRS исходного растра
- Выход: `runs/default/submission.geojson`

## Быстрый старт

1) Настройте конфиг `configs/default.yaml` (пути, классы, каналы). Для быстрого прогона можно ограничить регионы:
```yaml
train:
  limit_regions: 5         # первые 5 регионов в train
  limit_val_regions: 2     # первые 2 региона в val
  epochs: 5
  batch_size: 4
```

2) Построить сплиты:
```bash
python -m src.build_splits
```

3) Обучить модель:
```bash
python -m src.train
```

4) Инференс в GeoJSON (использует `runs/default/best.pt`):
```bash
python -m src.infer_to_geojson
```

## Выходные артефакты

- `runs/default/index.csv`, `train_regions.csv`, `val_regions.csv`
- Обучение: `metrics.csv`, `learning_curve.png`, `overlays/*.png`, `best.pt`
- Инференс: `submission.geojson`

## Частые вопросы / Ошибки

- Много шагов в эпохе (например, 7763) — это число батчей, не эпох. Уменьшите датасет (`limit_regions`) или увеличьте `batch_size`.
- Предупреждения AMP/CUDA/MPS на macOS: AMP автоматически отключится, обучение идёт на CPU/MPS. Можно выставить `train.amp: false`.
- Ошибки CRS при загрузке GeoJSON: код нормализует CRS слоёв к CRS растра, проблемные слои пропускаются.

## Лицензия

MIT License

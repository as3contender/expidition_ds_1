# Нормализация имён классов из GeoJSON к единому набору (ключи — возможные строки в ваших geojson)
# Взяты типовые варианты (кириллица/транслит). При необходимости дополни.
CLASS_NAME_MAPPING = {
    # Селища
    "селище": "selishcha",
    "селища": "selishcha",
    "Селище": "selishcha",
    "Селища": "selishcha",
    # Пашни
    "пашня": "pashni",
    "пашни": "pashni",
    "Пашня": "pashni",
    "Пашни": "pashni",
    "пахота": "pashni",
    # Курганы (если будут)
    "курган": "kurgany",
    "курганы": "kurgany",
    "Курган": "kurgany",
    "Курганы": "kurgany",
    # Караванные пути
    "караванные": "karavannye_puti",
    "Караванные": "karavannye_puti",
    "караванные пути": "karavannye_puti",
    "Караванные пути": "karavannye_puti",
    # Фортификации
    "фортификация": "fortifikatsii",
    "фортификации": "fortifikatsii",
    "Фортификация": "fortifikatsii",
    "Фортификации": "fortifikatsii",
    # Городища
    "городище": "gorodishcha",
    "городища": "gorodishcha",
    "Городище": "gorodishcha",
    "Городища": "gorodishcha",
    # Дороги
    "дорога": "dorogi",
    "дороги": "dorogi",
    "Дорога": "dorogi",
    "Дороги": "dorogi",
    # Ямы
    "яма": "yamy",
    "ямы": "yamy",
    "Яма": "yamy",
    "Ямы": "yamy",
    # Иное
    "иное": "inoe",
    "Иное": "inoe",
}


def normalize_class(name: str) -> str | None:
    if name is None:
        return None
    name = str(name).strip()
    return CLASS_NAME_MAPPING.get(name, name)

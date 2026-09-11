"""Built-in example prompts for quick testing."""

EXAMPLES = {
    "jazz_funk": {
        "style": "Jazz-funk, warm lead vocal, Rhodes piano, electric bass, tight drums",
        "lyrics": (
            "[Verse 1]\n"
            "Walking down the avenue\n"
            "Neon lights in shades of blue\n"
            "Coffee cup and morning rain\n"
            "Every day feels just the same\n"
            "\n"
            "[Chorus]\n"
            "But tonight we break the chain\n"
            "Dancing through the pouring rain\n"
            "Let the rhythm take control\n"
            "Feel the music in your soul\n"
        ),
        "seed": 42,
    },
    "cyber_metal": {
        "style": "Cyber metal, aggressive vocals, distorted guitars, industrial drums, dark atmosphere",
        "lyrics": (
            "[Verse 1]\n"
            "Steel and silicon collide\n"
            "Nothing left to hide inside\n"
            "Circuits burn and data flows\n"
            "Through the wires nobody knows\n"
            "\n"
            "[Chorus]\n"
            "We are the machine\n"
            "Running cold and clean\n"
            "No heart no dream\n"
            "Just code and scheme\n"
        ),
        "seed": 123,
    },
    "mandarin_pop": {
        "style": "Mandarin pop, gentle vocal, acoustic guitar, strings, emotional ballad",
        "lyrics": (
            "[Verse 1]\n"
            "窗外的雨还在下\n"
            "思念像风不停刮\n"
            "回忆在心底发芽\n"
            "你说过的话\n"
            "\n"
            "[Chorus]\n"
            "我在这里等你的回答\n"
            "不管世界怎么变化\n"
            "心里只有一个家\n"
            "就是你啊\n"
        ),
        "seed": 7,
    },
    "russian_rock": {
        "style": "Russian rock, powerful male vocal, electric guitars, driving drums, anthemic chorus",
        "lyrics": (
            "[Verse 1]\n"
            "Дождь стучит по старым крышам\n"
            "Город спит и никто не слышит\n"
            "Только ветер знает правду\n"
            "О дороге, что пройдена\n"
            "\n"
            "[Chorus]\n"
            "Мы идём вперёд сквозь тени\n"
            "Через ночь к новому дню\n"
            "Не страшны нам преграды\n"
            "Верь в себя — и я верю\n"
        ),
        "seed": 2026,
    },
}


def list_examples():
    """Return available example names."""
    return list(EXAMPLES.keys())


def get_example(name: str) -> dict:
    """Get an example prompt by name."""
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example: {name}. Available: {list_examples()}")
    return EXAMPLES[name]

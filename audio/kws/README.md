# Keyword spotting (каркас)

Прикладная часть проекта: модель класса DS-CNN на Google Speech Commands v2, затем
деплой на микроконтроллер как wake-word триггер для [Костяна](https://github.com/Vzzzz/skull-assistant/).

**Статус:** каркас написан, на реальных данных не проверялся.

## Файлы

- `features.py`: log-mel фронтенд на `torch.stft` без torchaudio. Параметры по
  умолчанию повторяют типичный TinyML-сетап: 16 кГц, окно 30 мс, шаг 20 мс, 40 мел-полос,
  49 кадров на секунду звука. Тот же фронтенд потом воспроизводится на CMSIS-DSP, поэтому
  никакой магии: STFT, мел-фильтры, логарифм.
- `model.py`: DS-CNN по Hello Edge (Zhang et al. 2017) из `QuantConv2d`, размер
  настраивается числом каналов и блоков.
- `data.py`: датасет Speech Commands v2 с официальным разбиением по
  `validation_list.txt` / `testing_list.txt`, 12 классов (10 слов, `_silence_`,
  `_unknown_`). Читает WAV стандартной библиотекой.

## Данные

Скачать и распаковать архив `speech_commands_v0.02.tar.gz` (≈2.3 ГБ) в `data/speech_commands_v2/`.
Источник: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

## Следующие шаги

1. Проверить загрузчик на данных, посчитать статистику классов.
2. FP32-бейзлайн DS-CNN (ориентир из Hello Edge: ~94–95% для DS-CNN S на 12 классах).
3. Скрипт `run.py` по образцу `sandbox_mnist/run.py`, свип лесенки.
4. Экспорт int8 в ONNX для X-CUBE-AI; упаковка весов <8 бит для своих ядер.

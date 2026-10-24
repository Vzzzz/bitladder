# bitladder

**Лесенка от 32 до 1 бита.** Проект об экстремальной оптимизации маленьких нейросетей для
always-on устройств: квантование (QAT, тернарное 1.58 бит, бинарное), прунинг, анализ
слоёв и честный подсчёт того, что остаётся в байтах, операциях и миллисекундах на
реальном железе. Два прикладных трека: **wake word на микроконтроллере** и **voice
enhancement на минимуме параметров** для смартфона и Jetson.

Прикладная цель: wake word для голосового ассистента
[Костяна](https://github.com/Vzzzz/skull-assistant/), который слушает всегда на
микроконтроллере и будит спящий Jetson по ключевому слову.

Вдохновение: [ViT-1.58b: Mobile Vision Transformers in the 1-bit Era](https://arxiv.org/abs/2406.18051).

## Результаты на сегодня (2026-10-24)

Keyword spotting на Speech Commands v2, DS-CNN 64×4 (23k параметров, 10.6 MMAC):

- **Бейзлайн**: 95.0 % на официальном тесте (12 классов); Hello Edge для DS-CNN S: 94.4 %.
- **Модульное расширение новыми словами** (воспроизведение
  [Яндекс, Interspeech 2026](https://arxiv.org/abs/2607.19918)): база заморожена, на новое
  слово обучается ветка ≤ 10k параметров. Порядок методов из статьи воспроизводится
  (ветка лучше отдельной модели и одной головы).
- **Новое**: квантование базы в 4 бита (16 КБ вместо 90 КБ) и ветки до тернарных весов
  не стоит ничего по FRR в лимите 10k параметров. **Новое слово стоит 1.5 КБ флеша**
  (ветка 6k параметров, тернарные веса, FRR 6.8 ± 0.8 % при FAR 1 %, три сида).
- Правило «сначала ядро» для невиданных слов не работает, и калибровка порогов на
  трудных негативах (Common Voice) этого не чинит: замороженная база уверенно
  реагирует на акустически близкие слова. Подробности и таблицы:
  [docs/results/kws_expansion_2026-10-23.md](docs/results/kws_expansion_2026-10-23.md).

## Структура проекта

```
bitladder/            библиотека
  quant/              STE-квантователи весов (uniform / LSQ, тернарный BitNet, бинарный XNOR),
                      PACT для активаций, слои QuantConv2d / QuantLinear, apply_quantization
  measure/            байты после упаковки битов, MACs и BOPs, анализатор слоёв
                      (статистика весов и активаций, чувствительность к битам и прунингу)
  models/, data/      маленькие модели и загрузчики для песочницы MNIST
  cli.py, train.py    общие аргументы квантования, цикл обучения
audio/kws/            keyword spotting: log-mel без torchaudio, кэш признаков, DS-CNN,
                      run.py (обучение), expand.py (модульное расширение), analyze.py,
                      cv_negatives.py (негативы из Common Voice)
colab/kws.ipynb       те же скрипты в Google Colab для длинных свипов
sandbox_mnist/        разогрев на MNIST
docs/                 план, decision-log (ADR), литобзор, железо, результаты
tests/                25 юнит-тестов (квантователи, упаковка, размеры, конвейер данных, анализатор)
results/              артефакты прогонов (в git не входят)
```

## Документы

- [docs/plan.md](docs/plan.md): цели, вопросы исследования, методы, метрики, фазы.
- [docs/decision-log.md](docs/decision-log.md): журнал решений в формате ADR.
- [docs/PUBLICATIONS.md](docs/PUBLICATIONS.md): литобзор, записи P-NNN с проверяемыми следствиями.
- [docs/results/](docs/results/): сводные таблицы экспериментов.
- [docs/hardware.md](docs/hardware.md): Jetson, микроконтроллер, аудио.
- [docs/references.md](docs/references.md): стартовый список статей и инструментов.

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # или cu128
pip install -e ".[dev]"
pytest -q

# Speech Commands v2 (CC BY 4.0): http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# и тестовый набор speech_commands_test_set_v0.02.tar.gz, распакованные в data/
python audio/kws/cache.py --root data/speech_commands_v2 --test-root data/speech_commands_test_v2
python audio/kws/run.py --epochs 30                                   # 12 классов, fp32
python audio/kws/run.py --keywords yes,no,up,down,left,right,on,off --exclude stop,go --unknown-ratio 1.0 --name base8
python audio/kws/expand.py --base results/base8.json --new stop,go --k 16 --separable --epochs 20
python audio/kws/expand.py --base results/base8.json --new stop,go --k 16 --separable --scheme ternary --abits 8 --no-first-last
python audio/kws/analyze.py --run results/base8.json                  # таблица и тепловая карта чувствительности
```

Каждый прогон пишет JSON и чекпоинт в `results/`: точность или FRR, параметры,
байты после упаковки битов, MACs и BOPs.

## Статус

Фаза 1, трек wake word. Следующие шаги в [CLAUDE.md](CLAUDE.md), раздел «Ближайшие шаги».

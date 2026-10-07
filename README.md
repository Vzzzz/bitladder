# bitladder

**Лесенка от 32 до 1 бита.** Микро-проект об экстремальной оптимизации нейросетей:
квантование (PTQ, QAT, тернарное 1.58 бит, бинарное), прунинг, дистилляция и честный
подсчёт того, что остаётся в байтах, операциях и миллисекундах на реальном железе.

Вдохновение: [ViT-1.58b: Mobile Vision Transformers in the 1-bit Era](https://arxiv.org/abs/2406.18051).

## Структура проекта

```
bitladder/            библиотека: квантователи, слои, измерения, модели, обучение
  quant/              STE-квантователи весов и активаций, обёртки QuantConv2d / QuantLinear
  measure/            размер в байтах (с упаковкой битов), MACs и BOPs
  models/             маленькие архитектуры для песочницы
  data/               загрузчики данных
sandbox_mnist/        песочница: один прогон и свип по лесенке битов
audio/kws/            каркас прикладной части: keyword spotting (DS-CNN, log-mel)
docs/                 план, decision-log, заметки по железу, ссылки
tests/                юнит-тесты квантователей и подсчёта размера
results/              артефакты прогонов (в git не входят)
```

## Документы

- [docs/plan.md](docs/plan.md): цели, вопросы исследования, методы, метрики, фазы.
- [docs/decision-log.md](docs/decision-log.md): журнал решений в формате ADR.
- [docs/hardware.md](docs/hardware.md): Jetson, микроконтроллер, аудио для Костяна.
- [docs/references.md](docs/references.md): статьи и инструменты.

## Быстрый старт

```bash
conda env create -f environment.yml
conda activate bitladder
pip install -e .
pytest -q
python sandbox_mnist/run.py --scheme fp32 --epochs 2
python sandbox_mnist/run.py --scheme ternary --abits 8 --epochs 2
python sandbox_mnist/sweep.py --epochs 3
```

Каждый прогон пишет JSON с точностью, числом параметров, упакованным размером в байтах,
MACs и BOPs в `results/`. Свип собирает лесенку в `results/ladder.csv`.

## Статус

Фаза 0: инфраструктура. См. [docs/plan.md](docs/plan.md).

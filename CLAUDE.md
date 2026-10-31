# bitladder: контекст для агентов

Этот файл читается Claude Code автоматически. Он даёт достаточный контекст, чтобы
начать работу без истории чатов. Подробности в `docs/`.

## Что это

Микро-проект для портфолио Валерия: исследование экстремальной оптимизации нейросетей
(квантование 32 → 1 бит, прунинг, дистилляция) с доведением до железа и короткой
публикацией. Вдохновлён статьёй ViT-1.58b (arXiv 2406.18051, тернарные веса, 8-битные
активации).

Проект связан с главным проектом портфолио, голосовым ассистентом
[**Костяном**](https://github.com/Vzzzz/skull-assistant/) (анатомический скелет на Jetson Orin Nano
Developer Kit, репозиторий `skull-assistant`). Прикладная цель здесь: keyword spotting на микроконтроллере,
который слушает всегда и будит спящий Jetson по ключевому слову.

## Текущее состояние (2026-10-23)

Приоритет по ADR-009: два прикладных трека, wake word на STM32 (первый) и voice
enhancement на минимуме параметров (смартфон, Jetson). Лесенка битов и SDK-швы остаются
инфраструктурой под треками. Литобзор ведётся в `docs/PUBLICATIONS.md` (ADR-010).

**Окружение.** Рабочая копия и обучение в WSL Ubuntu-22.04: `~/bitladder`, свой `.venv`
(Python 3.10, torch 2.14 cu130), GPU RTX 3080 Laptop 8 ГБ, toolkit CUDA 13.3.
Windows-копия репозитория служит зеркалом для документации. Источник правды: GitHub
(origin по SSH). Поток: коммит в любой копии → push → в другой `git pull --ff-only`.
В клоне WSL нельзя запускать `git clean` / `reset --hard` (в нём живут venv, данные,
чекпоинты). Перед GPU-работой в WSL остановить стенд skull-assistant:
`systemctl --user stop skull.target` (его ollama держит ~4 ГБ VRAM). Окружение
skull-assistant не трогать. Длинные свипы можно гонять в Google Colab или Kaggle: `colab/kws.ipynb` (ячейки `run_once` пропускают готовые прогоны). Результаты прогонов (JSON, MD, PNG, CSV)
уходят в приватный репозиторий `Vzzzz/bitladder-results` через `bitladder/results_sync.py`:
в Colab токен в секрете `RESULTS_TOKEN` (ячейка S), локально `RESULTS_TOKEN=... python -m
bitladder.results_sync --repo Vzzzz/bitladder-results --label local-rtx3080 results`. Чекпоинты `.pt` до 2 МБ
тоже идут (наши 0.1–0.5 МБ), чтобы новая сессия продолжала с места обрыва (`sync.restore`). Раскладка: `<label>/<name>.json`, label = `colab-<дата>` или `local-<машина>`.

**Данные** в `~/bitladder/data/` (gitignored): `speech_commands_v2/` (105 835 wav, 35 слов
плюс `_background_noise_`), `speech_commands_test_v2/` (официальный тест на 12 классов,
4 890 wav), `kws_cache/` (log-mel float16 для всех клипов, строится `audio/kws/cache.py`
за две минуты). Common Voice 27.0 ru распакован в `data/cv/`, кэши окон `kws_cache/cv_test.npy` (20k) и `cv_train.npy` (100k).

**Что работает (2026-10-25).** 39 тестов зелёные. `audio/kws/run.py`: DS-CNN 64×4
(23k параметров, 10.6 MMAC), 30 эпох по 4 с, 12 классов: 95.0 % на официальном тесте
(Hello Edge DS-CNN S: 94.4 %). `audio/kws/expand.py`: модульное расширение по P-001
(ветка на замороженной базе, режимы expand / separate / head, `--separable` для
depthwise-смешивания: k=16 → 7.1k параметров; три правила арбитража hard / soft / none;
FRR при FAR 1 %; `--neg cv` калибрует пороги на Common Voice). Итоги в
`docs/results/kws_expansion_2026-10-23.md`: порядок методов статьи воспроизведён; база
в 4 битах бесплатна, тернарная ветка стоит 1–4 пункта (бьёт по depthwise); в лимите 10k
новое слово = 1.5 КБ при 8.6 % FRR или 4.2 КБ при 6.6 % (две пары, три сида, T4; статья
4.37 % по пяти парам при ~10 КБ int8); Colab воспроизводит сам себя в пределах σ; правило «сначала ядро» для невиданных слов не
работает и калибровкой не чинится; калибровочные негативы обязаны быть невиданными
при обучении; числа Colab и локальные не смешивать (разброс между машинами больше,
чем между сидами). `audio/kws/analyze.py` + `bitladder/measure/analyze.py`: анализатор
слоёв (статистика весов и активаций, чувствительность одним прогоном к битам,
прунингу каналов и разреженности, markdown и тепловая карта).

**Код.** `bitladder/quant/` квантователи (uniform с опцией LSQ, тернарный BitNet с
тождественным STE, бинарный XNOR, PACT), слои `QuantConv2d` / `QuantConv1d` / `QuantLinear`
/ `QuantGRU` (веса на тензор, cuDNN-ядро), `apply_quantization` с переопределениями
битности по слоям (`QuantConfig.overrides`, маски fnmatch), тернарные режимы absmean / twn / ttq,
`fold.py` сворачивание BatchNorm; `bitladder/measure/mixed.py` жадный профиль смешанной точности
по отчёту анализатора (`--mixed-from`, `--mixed-budget`); `bitladder/train.py` дистилляция логитов
(`--teacher` в run.py); `bitladder/distill.py` дистилляция эмбеддингов без меток;
`bitladder/results_sync.py` отправка JSON/MD/PNG в репозиторий результатов после каждого прогона; `bitladder/measure/` байты после упаковки, MACs, BOPs, анализатор слоёв;
`bitladder/cli.py` общие аргументы квантования; `audio/kws/` features (log-mel без
torchaudio), cache, data, model (DS-CNN), checkpoint, run, expand, analyze, cv_negatives; `colab/kws.ipynb`; `sandbox_mnist/` разогрев на
MNIST (свип по MNIST не нужен); `tests/` 39 тестов, включая конвейер данных KWS на
синтетическом мини-датасете и анализатор слоёв.

## Ключевые решения

Все решения в `docs/decision-log.md` (формат ADR). Коротко:

1. Структура «песочница → перенос»: MNIST как разогрев, затем KWS на Speech Commands v2
   как основная песочница и цель деплоя.
2. Квантователи пишем сами на PyTorch (не Brevitas / torchao).
3. Размер считаем в байтах после упаковки битов с масштабами и смещениями fp16;
   параметры и «теоретические биты» только как справка.
4. Jetson будится импульсом на пин 12 хедера J14 (кнопка питания). Плата MCU:
   STM32F407G-DISC1, заказана 2026-10-25 (ADR-011; ADR-007 заменён).
5. Speech enhancement в бэклоге, нейросеть на FV-1 как отдельный спин-офф.
6. ADR-009: два прикладных трека (wake word на STM32 первый, voice enhancement второй),
   матрица устройств: RTX 3080 Laptop, Galaxy S25 Ultra, Jetson Orin Nano 8 ГБ, STM32.
7. ADR-010: литобзор в `docs/PUBLICATIONS.md` по формату skull-assistant (записи P-NNN).
8. ADR-012: брелок с анекдотами (ESP32-S3, LM + голос) живёт в отдельном приватном
   репозитории `joke-keychain` (GPL-3), bitladder там зависимость.

## Соглашения

- Документация и decision-log на русском. Код, идентификаторы и комментарии в коде на
  английском. Файлы с переносами LF (`.gitattributes` нормализует, но при записи
  скриптами на Windows задавать `newline="\n"`).
- Новое решение = новая запись ADR в конце `docs/decision-log.md` с датой. Старые записи
  не редактируются, а меняют статус.
- Разбор публикации = запись P-NNN в `docs/PUBLICATIONS.md` с проверяемым следствием.
- `README.md` (русский) и `README.en.md` (английский) держать синхронными по содержанию.
- CI (`.github/workflows/ci.yml`): `ruff check .` и `pytest -q` на CPU при каждом push; перед
  коммитом кода прогонять оба локально.
- Каждый эксперимент пишет JSON и чекпоинт `.pt` в `results/` (в git не входит). Сводные
  таблицы и графики для публикации складываются в `docs/results/`.
- Модели собираются из `QuantConv2d` / `QuantLinear` сразу, квантование включается
  через `apply_quantization(model, QuantConfig(...))`. Первый и последний слой по
  умолчанию остаются в 8 битах, вход первого слоя не квантуется.
- Совместимость с torch >= 2.0, без torchaudio.

## Как запускать

```bash
pip install -e ".[dev]"
pytest -q
python audio/kws/cache.py --root data/speech_commands_v2 --test-root data/speech_commands_test_v2
python audio/kws/run.py --epochs 30                                  # 12 классов, fp32
python audio/kws/run.py --keywords yes,no,up,down,left,right,on,off --exclude stop,go --unknown-ratio 1.0 --name base8
python audio/kws/expand.py --base results/base8.json --new stop,go --epochs 20 --k 16
python audio/kws/expand.py --base results/base8.json --new stop,go --scheme uniform --wbits 4 --abits 8 --no-first-last
python sandbox_mnist/run.py --scheme ternary --abits 8 --epochs 2    # разогрев на MNIST
```

## Ближайшие шаги

1. Расширение (P-001): главный рычаг найден, размер базы (results §9: DS-CNN 128×6 даёт
   2.1 % FRR против 6.9 %, один сид). Повторить по трём сидам и двум парам; ужать ветку
   на большой базе (`--taps 2..3`, k=8); досчитать из ячейки 3d: time-stride 1, сетка
   ветки; ячейка 5 (MNIST). GPU: локально при свободном стенде или Kaggle (адаптировать ноутбук).
2. Анализ слоёв DS-CNN (ячейка 3c), затем лесенка битов на 12 классах и прунинг по
   подсказкам анализатора.
3. Литобзор: партия 2 (P-002…P-009) сделана 2026-10-25; следующая партия: обзор
   квантования для MCU (arXiv 2508.15008), Larq / LCE, BitNet / ViT-1.58b.
4. Данные для своего слова «Костян»: синтетика TTS, аугментации, реальные записи для теста.
5. int8 на плате STM32F407G-DISC1 (ADR-011, едет) по `docs/stm32_deploy.md`;
   скрипты экспорта (`export_onnx.py`, `export_calib.py`, `export_testvector.py`) готовы и проверены; проверить на Jetson пробуждение из
   SC7 кнопкой (ADR-006).

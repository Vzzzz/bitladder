# bitladder: контекст для агентов

Этот файл читается Claude Code автоматически. Он даёт достаточный контекст, чтобы
начать работу без истории чатов. Подробности в `docs/`.

## Что это

Микро-проект для портфолио Валерия: исследование экстремальной оптимизации нейросетей
(квантование 32 → 1 бит, прунинг, дистилляция) с доведением до железа и короткой
публикацией. Вдохновлён статьёй ViT-1.58b (arXiv 2406.18051, тернарные веса, 8-битные
активации).

Проект связан с главным проектом портфолио, голосовым ассистентом **Костяном**
(анатомический скелет на Jetson Orin Nano Developer Kit, отдельный репозиторий
`skull-assistant`). Прикладная цель здесь: keyword spotting на микроконтроллере,
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
skull-assistant не трогать. Длинные свипы можно гонять в Google Colab: `colab/kws.ipynb`.

**Данные** в `~/bitladder/data/` (gitignored): `speech_commands_v2/` (105 835 wav, 35 слов
плюс `_background_noise_`), `speech_commands_test_v2/` (официальный тест на 12 классов,
4 890 wav), `kws_cache/` (log-mel float16 для всех клипов, строится `audio/kws/cache.py`
за две минуты). Архив Common Voice 27.0 ru скачан локально, не распакован.

**Что работает (2026-10-23).** 21 тест зелёный. `audio/kws/run.py`: DS-CNN 64×4
(23k параметров, 10.6 MMAC), 30 эпох по 4 с, 12 классов: 95.0 % на официальном тесте
(Hello Edge DS-CNN S: 94.4 %). `audio/kws/expand.py`: модульное расширение по P-001
(ветка ≤ 10k параметров на замороженной базе, режимы expand / separate / head, правило
«сначала ядро» по откалиброванным порогам, FRR при FAR 1 %). Первые результаты на
{stop, go}: без гейтинга ветка 11–15 % FRR, отдельная модель 35 %, только голова 92 %
(порядок как в статье, абсолютные числа хуже: у них 4.4 / 6.5 / 22). Ядро срабатывает
на 45–74 % клипов новых слов, что обнуляет гейтинг: открытый вопрос, нужны трудные
негативы для калибровки порогов (Common Voice). Сводки результатов в `docs/results/`.

**Код.** `bitladder/quant/` квантователи (uniform с опцией LSQ, тернарный BitNet с
тождественным STE, бинарный XNOR, PACT), слои `QuantConv2d` / `QuantLinear`,
`apply_quantization`; `bitladder/measure/` байты после упаковки, MACs, BOPs;
`bitladder/cli.py` общие аргументы квантования; `audio/kws/` features (log-mel без
torchaudio), cache, data, model (DS-CNN), run, expand; `sandbox_mnist/` разогрев на
MNIST (свип по MNIST не нужен); `tests/` 21 тест, включая конвейер данных KWS на
синтетическом мини-датасете.

## Ключевые решения

Все решения в `docs/decision-log.md` (формат ADR). Коротко:

1. Структура «песочница → перенос»: MNIST как разогрев, затем KWS на Speech Commands v2
   как основная песочница и цель деплоя.
2. Квантователи пишем сами на PyTorch (не Brevitas / torchao).
3. Размер считаем в байтах после упаковки битов с масштабами и смещениями fp16;
   параметры и «теоретические биты» только как справка.
4. Jetson будится импульсом на пин 12 хедера J14 (кнопка питания). Кандидат MCU:
   STM32 Discovery B-U585I-IOT02A (пока предложено, не куплено).
5. Speech enhancement в бэклоге, нейросеть на FV-1 как отдельный спин-офф.
6. ADR-009: два прикладных трека (wake word на STM32 первый, voice enhancement второй),
   матрица устройств: RTX 3080 Laptop, Galaxy S25 Ultra, Jetson Orin Nano 8 ГБ, STM32.
7. ADR-010: литобзор в `docs/PUBLICATIONS.md` по формату skull-assistant (записи P-NNN).

## Соглашения

- Документация и decision-log на русском. Код, идентификаторы и комментарии в коде на
  английском. Файлы с переносами LF (`.gitattributes` нормализует, но при записи
  скриптами на Windows задавать `newline="\n"`).
- Новое решение = новая запись ADR в конце `docs/decision-log.md` с датой. Старые записи
  не редактируются, а меняют статус.
- Разбор публикации = запись P-NNN в `docs/PUBLICATIONS.md` с проверяемым следствием.
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

1. Расширение (P-001): таблица «FRR и байты ветки против битности базы и ветки»;
   калибровка порогов ядра на Common Voice ru как трудных негативах.
2. Лесенка битов на DS-CNN 12 классов (uniform 8/4/2, тернарно, бинарно),
   чувствительность слоёв.
3. Литобзор: microWakeWord, openWakeWord, Hello Edge, Rusci 2020 / CMix-NN, Speech Commands.
4. Данные для своего слова «Костян»: синтетика TTS, аугментации, реальные записи для теста.
5. int8 на плате STM32 (ADR-007, плата не куплена); проверить на Jetson пробуждение из
   SC7 кнопкой (ADR-006).

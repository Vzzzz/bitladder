# bitladder: контекст для агентов

Единственный файл контекста для любого кодового ассистента: Codex, Cursor, Copilot и
другие читают `AGENTS.md` сами, Claude Code через `CLAUDE.md` (одна строка `@AGENTS.md`),
Gemini CLI через `context.fileName` в настройках. Он даёт достаточный контекст, чтобы
начать работу без истории чатов и без памяти конкретного ассистента: состояние здесь,
решения в `docs/decision-log.md`, числа в `docs/results/`, литература в `docs/PUBLICATIONS.md`,
план в `docs/plan.md`. Подробности в `docs/`.

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
skull-assistant не трогать. Длинные свипы можно гонять в Google Colab или Kaggle: `colab/kws.ipynb` (ячейки `run_once` пропускают готовые прогоны), лесенка на MNIST отдельно в `colab/mnist.ipynb` (один сплошной прогон, по ширине за `run_once`); всё недосчитанное по KWS одним прогоном в `colab/kws_full.ipynb` (блоки A–F: пара on/off на большой базе, пороги по CV, сетка ветки, методы по сидам 1–2, лесенка на 12 классах, сводка). Ячейка S восстанавливает файлы всех меток, новые раньше старых. Результаты прогонов (JSON, MD, PNG, CSV)
уходят в приватный репозиторий `Vzzzz/bitladder-results` через `bitladder/results_sync.py`:
в Colab токен в секрете `RESULTS_TOKEN` (ячейка S), локально `RESULTS_TOKEN=... python -m
bitladder.results_sync --repo Vzzzz/bitladder-results --label local-rtx3080 results`. Чекпоинты `.pt` до 2 МБ
тоже идут (наши 0.1–0.5 МБ), чтобы новая сессия продолжала с места обрыва (`sync.restore`). Раскладка: `<label>/<name>.json`, label = `colab-<дата>` или `local-<машина>`.

**Эта машина (для агента, проверено 2026-11-03).** Windows-копия в
`C:\Users\valer\git\bitladder`; системный Python Windows без torch и со сломанным ssl, им
ничего не запускать. Тесты и скрипты против Windows-копии гонять интерпретатором WSL:
`wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/c/Users/valer/git/bitladder && ~/bitladder/.venv/bin/python -m pytest -q -p no:cacheprovider"`
(так же `-m ruff check .`). GPU-работа только в `~/bitladder` в WSL. Репозиторий результатов
клонировать по SSH: `git clone git@github.com:Vzzzz/bitladder-results.git` (приватный, ключ
на машине есть), для сводок хватает чтения JSON; токен `RESULTS_TOKEN` живёт только в
секретах Colab/Kaggle и нигде не печатается. Временные файлы агента держать вне репозитория.
Стенд skull-assistant в WSL часто занят работой владельца: перед тяжёлыми запусками спросить.
Архив Common Voice ru лежит в `C:\Users\valer\Downloads\` (6.8 ГБ), распакован в `~/bitladder/data/cv/`.

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
`docs/results/kws_expansion_2026-10-23.md` (§1–13, 2026-11-03): порядок методов статьи воспроизведён; база
в 4 битах бесплатна (три сида); главный рычаг это размер базы: на DS-CNN 128×6 (69 КБ, 55 MMAC) ветка 6.3 КБ даёт
4.3 ± 2.1 % FRR, 2.2 КБ 5.3 %, 1.4 КБ 6.6 % (две пары, три сида, T4; статья 4.37 % по пяти парам при ~10 КБ int8),
на базе 64×4 4.2 КБ при 6.6 % или 1.5 КБ тернарных при 8.6 %; дистилляция от 128×6 даёт базе +0.3 и ветке −0.9 пункта;
TWN лучший тернарный режим, но по трём сидам стоит 2.5 пункта за половину байт; смешанная точность ничего не
даёт, пока 4 бита бесплатны; ветка недоучена (40 эпох при lr 3e-3: 4.3 % против 7.8 %, один сид); лесенка на 12
классах (офиц. тест): fp32 95.2 %, 4 бита 95.2 % в 16.2 КБ, тернарная 92.5 % в 10.6 КБ, 2 бита 88.2 %, бинарная 84.5 %;
Colab воспроизводит сам себя в пределах σ на 64×4, на 128×6 разброс между сессиями до 1.5 пункта; правило «сначала ядро» для невиданных слов не
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
MNIST (свип по MNIST идёт только в раздел sandbox_mnist, не в README); `tests/` 39 тестов, включая конвейер данных KWS на
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
   репозитории `micro-trakhtenberg` (GPL-3), bitladder там зависимость.

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

1. Расширение (P-001): три сида × две пары на большой базе сделаны (results §9). Дальше: ветка на большой базе
   за 40 эпох при lr 3e-3 (results §11), три сида и две пары; блок B (пороги по Common Voice, 18 прогонов)
   локально на 3080 с чекпоинтами из репозитория результатов или в Colab после загрузки `kws_cache_cv.tar`
   на Диск (`CV_FILE_ID` в ячейке 2a); лесенка на MNIST (`colab/mnist.ipynb`) ещё не прогнана.
2. Прунинг pointwise-слоёв базы по подсказкам анализатора, сравнить с ёмкостью по HOPE (P-011); ужать
   8-битные stem и fc на 12 классах (пол лесенки 3.5 КБ).
3. Литобзор: партия 2 (P-002…P-009) сделана 2026-10-25, P-010 и P-011 (HOPE, прунинг без данных) добавлены по одной; следующая партия: обзор
   квантования для MCU (arXiv 2508.15008), Larq / LCE, BitNet / ViT-1.58b.
4. Данные для своего слова «Костян»: синтетика TTS, аугментации, реальные записи для теста.
5. int8 на плате STM32F407G-DISC1 (ADR-011, едет) по `docs/stm32_deploy.md`;
   скрипты экспорта (`export_onnx.py`, `export_calib.py`, `export_testvector.py`) готовы и проверены; проверить на Jetson пробуждение из
   SC7 кнопкой (ADR-006).

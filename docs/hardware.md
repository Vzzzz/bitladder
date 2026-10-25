# Железо

## Jetson Orin Nano Developer Kit ([Костян](https://github.com/Vzzzz/skull-assistant/))

Стандартная плата-носитель NVIDIA. Проверено по документации и форуму 2026-10-07.

**Хедер J14 (12 пинов, «button header»):**

| Пин | Назначение |
|---|---|
| 5 и 6 | перемычка отключает автовключение при подаче питания |
| 8 | System Reset |
| 10 | Force Recovery |
| 11 | GND |
| 12 | Power button (к GND) |

**Глубокий сон SC7** (Linux suspend-to-RAM): `sudo systemctl suspend` или
`echo mem > /sys/power/state`. Запитаны остаются VDD_RTC, VDDIO_DDR, VDDIO_SYS и DRAM;
VDD_CORE и VDD_CPU выключены.

**Источники пробуждения** по документации: пин кнопки питания, будильник RTC
(`echo +10 > /sys/class/rtc/rtc0/wakealarm`), USB-C hotplug, клавиша USB-клавиатуры,
Wake-on-LAN, карта SD.

**Известная проблема:** на Jetson Linux 36.3 после выхода из SC7 плата бодрствовала
около 6 секунд и засыпала снова. Исправлено в 36.4.3, но потребовалась перепрошивка
загрузчика, обновления rootfs не хватило.

**Схема пробуждения от MCU:** GPIO микроконтроллера → транзистор с открытым
коллектором или оптопара → пин 12 на пин 11 (GND), импульс порядка 0.5 с.
Тот же импульс включает плату из выключенного состояния при перемычке 5-6.

**Что измерить на девките до покупки MCU:**
- версия JetPack / Jetson Linux;
- ручной тест `systemctl suspend` и пробуждения кнопкой;
- потребление в простое и в SC7 (ваттметр в розетку или USB-C PD метр).

Источники:
- https://forums.developer.nvidia.com/t/how-to-enter-sleep-on-jetson-orin-nano-devkit-and-how-to-wake-up/306292
- https://forums.developer.nvidia.com/t/jetson-orin-nano-wont-come-out-of-deep-sleep/324698
- https://forums.developer.nvidia.com/t/jetson-orin-nano-power-and-reset-buttons-setup/299742
- https://forums.developer.nvidia.com/t/power-button-with-auto-power-on/359920
- https://docs.nvidia.com/jetson/archives/r36.5.2/DeveloperGuide/AT/BoardAutomation.html
- https://docs.nvidia.com/jetson/archives/r36.5.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html

## Микроконтроллер (wake-word триггер)

**Заказана (ADR-011, 2026-10-25): STM32F407G-DISC1.** Cortex-M4F 168 МГц, 1 МБ флеша,
192 КБ ОЗУ, MEMS-микрофон MP45DT02 (PDM), ЦАП CS43L22, четыре светодиода, ST-LINK на борту.
Инструменты: STM32CubeIDE, X-CUBE-AI, CMSIS-DSP (PDM2PCM, log-mel), CMSIS-NN (int8).

Прежний кандидат (ADR-007, заменён): **STM32 Discovery B-U585I-IOT02A**. Cortex-M33 160 МГц, 2 МБ
флеша, 786 КБ ОЗУ, два цифровых MEMS-микрофона, Wi-Fi/BLE, профиль серии U5
ultra-low-power. Инструменты: STM32CubeIDE, X-CUBE-AI, CMSIS-DSP (log-mel), CMSIS-NN (int8).

Бейзлайн (опционально): **Seeed XIAO ESP32S3 Sense**, PDM-микрофон на борту, готовый
wake-word движок Espressif WakeNet.

Измерение тока: Nordic PPK2 или мультиметр через перемычку измерения тока на Discovery.

Ниже 8 бит стандартных ядер нет: упакованные веса распаковываются в int8 на лету
(выигрыш во флеше, проигрыш в латентности) либо бинарная сеть на XNOR + popcount.

## Аудио для Костяна на Jetson

Требование номер один: аппаратный AEC (эхоподавление), чтобы Костяна можно было
перебивать, пока он говорит. AEC работает, только если микрофоны и динамик на одном
устройстве.

Кандидат: **Seeed ReSpeaker Lite** (XMOS XU316, 2 микрофона, дальнее поле до 3 м,
AEC, подавление шума, AGC, подавление помех, USB или I2S). Проверить наличие выхода на
динамик у выбранной версии. Запасной вариант: USB-спикерфон.

В SC7 USB-устройства Jetson обесточены, поэтому у MCU свой микрофон.

Источник: https://wiki.seeedstudio.com/ja/reSpeaker_usb_v3/

## Ориентировочная корзина

| Компонент | Роль | Цена, ориентир |
|---|---|---|
| B-U585I-IOT02A | исследование и wake-word триггер | 70–80 $ |
| XIAO ESP32S3 Sense | бейзлайн WakeNet, опционально | 15 $ |
| ReSpeaker Lite | микрофон и AEC для ASR Костяна | 25–35 $ |
| PPK2 | измерение энергии, опционально | 100 $ |
| транзистор / оптопара, провода | связь MCU с пином 12 | мелочь |

## FV-1 (спин-офф)

Spin Semiconductor FV-1: программа ровно 128 инструкций на отсчёт, без циклов (только
пропуск вперёд), 32 регистра, коэффициенты 16-битные fixed-point в диапазоне ±2,
регистры 24-битные, память задержки 14-битная, насыщение на ±1. Регистры живут между
отсчётами, то есть рекуррентное состояние бесплатно. Открытый ассемблер: asfv1.
Перед прошивкой нужен побитово точный эмулятор на Python.

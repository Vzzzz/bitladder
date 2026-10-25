# Ссылки

## Квантование
- ViT-1.58b: Mobile Vision Transformers in the 1-bit Era. https://arxiv.org/abs/2406.18051
- The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58). https://arxiv.org/abs/2402.17764
- BitNet: Scaling 1-bit Transformers for Large Language Models. https://arxiv.org/abs/2310.11453
- Learned Step Size Quantization (LSQ). https://arxiv.org/abs/1902.08153
- PACT: Parameterized Clipping Activation. https://arxiv.org/abs/1805.06085
- XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. https://arxiv.org/abs/1603.05279
- Binarized Neural Networks (Hubara et al.). https://arxiv.org/abs/1602.02830
- DoReFa-Net. https://arxiv.org/abs/1606.06160
- Quantizing deep convolutional networks for efficient inference: A whitepaper (Krishnamoorthi). https://arxiv.org/abs/1806.08342
- HAWQ: Hessian AWare Quantization (смешанная точность). https://arxiv.org/abs/1905.03696

## Прунинг и сжатие
- Deep Compression: Pruning, Trained Quantization and Huffman Coding. https://arxiv.org/abs/1510.00149
- The Lottery Ticket Hypothesis. https://arxiv.org/abs/1803.03635
- Accelerating Sparse Deep Neural Networks (N:M sparsity, NVIDIA). https://arxiv.org/abs/2104.08378

## Дистилляция
- Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531

## Keyword spotting и TinyML
- Hello Edge: Keyword Spotting on Microcontrollers (DS-CNN). https://arxiv.org/abs/1711.07128
- Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition. https://arxiv.org/abs/1804.03209
- MLPerf Tiny Benchmark. https://arxiv.org/abs/2106.07597
- CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs. https://arxiv.org/abs/1801.06601

## Audio enhancement (бэклог)
- GTCRN: ультралёгкая модель speech enhancement (~24k параметров). Номер arXiv уточнить.
- RNNoise: A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement. https://arxiv.org/abs/1709.08243

## Инструменты
- Brevitas (перекрёстная проверка квантователей). https://github.com/Xilinx/brevitas
- torchao. https://github.com/pytorch/ao
- X-CUBE-AI. https://www.st.com/en/embedded-software/x-cube-ai.html
- asfv1, ассемблер FV-1. https://github.com/ndf-zz/asfv1

## Разобрано в PUBLICATIONS.md (партии 1–2), ссылки для самостоятельного чтения

### Keyword spotting на MCU и данные для wake word
- P-001 Яндекс, Scalable Keyword Spotting via Modular Network Expansion (Interspeech 2026). https://arxiv.org/abs/2607.19918 · блог: https://habr.com/ru/companies/yandex/articles/1061968/
- P-002 Hello Edge: Keyword Spotting on Microcontrollers. https://arxiv.org/abs/1711.07128 · код: https://github.com/ARM-software/ML-KWS-for-MCU
- P-003 End-to-End Efficiency in Keyword Spotting: A System-Level Approach for Embedded Microcontrollers (2025). https://arxiv.org/abs/2509.07051
- P-004 microWakeWord: https://github.com/kahrendt/microWakeWord · модели: https://github.com/esphome/micro-wake-word-models · блог Home Assistant: https://www.home-assistant.io/blog/2024/02/21/voice-chapter-6/
- P-005 openWakeWord: https://github.com/dscripka/openWakeWord · livekit-wakeword (голова conv + attention): https://github.com/livekit/livekit-wakeword
- Speech Commands (Warden 2018): https://arxiv.org/abs/1804.03209 · данные: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

### Квантование ниже 8 бит и ядра для Cortex-M
- P-006 Memory-Driven Mixed Low Precision Quantization for Enabling Deep Network Inference on Microcontrollers (MLSys 2020). https://arxiv.org/abs/1905.13082 · https://proceedings.mlsys.org/paper_files/paper/2020/hash/0c8abcf158ed12d0dd94480681186fda-Abstract.html · код: https://github.com/mrusci/training-mixed-precision-quantized-networks
- Leveraging Automated Mixed-Low-Precision Quantization for tiny edge microcontrollers (продолжение, RL-подбор). https://arxiv.org/abs/2008.05124
- P-007 CMix-NN: Mixed Low-Precision CNN Library for Memory-Constrained Edge Devices (IEEE TCAS-II 2020). https://github.com/EEESlab/CMix-NN · https://cris.unibo.it/handle/11585/762836
- MCU-MixQ: HW/SW co-optimized mixed-precision NN design framework for MCUs. https://arxiv.org/abs/2407.18267
- Neural Network Quantization for Microcontrollers: A Comprehensive Survey (2025). https://arxiv.org/abs/2508.15008
- CMSIS-NN kernels (Arm blog). https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/new-neural-network-kernels-boost-efficiency-in-microcontrollers-by-5x

### Сверхлёгкий speech enhancement
- P-008 GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources (ICASSP 2024, DOI 10.1109/ICASSP48485.2024.10448310). Код: https://github.com/Xiaobin-Rong/gtcrn (в статье 23.7k параметров / 39.6 MFLOPs, в репозитории 48.2k / 33 MMAC/с: считают по-разному, уточнить при чтении)
- H-GTCRN, двухканальный вариант. https://arxiv.org/abs/2505.19597
- LiSenNet: Lightweight Sub-band and Dual-Path Modeling for Real-Time Speech Enhancement (ICASSP 2025). https://arxiv.org/abs/2409.13285
- Beyond Model Size: Redesigning LiSenNet for embedded speech enhancement (GN, 2026, STM32N6). https://arxiv.org/pdf/2609.29866 (идентификатор проверить)
- CoFi-Lite (IEEE SPL 2026, 83k параметров, 12.9 MMAC/с). https://arxiv.org/abs/2607.10142
- A light-weight full-band speech enhancement model (48 кГц, 0.89M). https://arxiv.org/abs/2206.14524
- RNNoise (Valin 2018). https://arxiv.org/abs/1709.08243
- P-009 TinyLSTMs: Efficient Neural Speech Enhancement for Hearing Aids (Interspeech 2020). https://arxiv.org/abs/2005.11138
- Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with Mixed FP16-INT8 PTQ (TinyDenoiser, 2022). https://arxiv.org/abs/2210.07692
- UL-UNAS (2025, продолжение GTCRN): см. README репозитория gtcrn.

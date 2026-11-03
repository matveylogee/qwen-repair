# Что осталось подтвердить для M1

M0 проверена локально и в Colab. Код M1 реализован, локальный Docker и интеграция trainer проверены. После переподключения Drive локальная проверка нагрузки и файловой очереди прошла. Следующий шаг: один настоящий запрос из Colab через Drive без модели и GPU. Полный M1 затем требует запуска Qwen и обучения на T4. Настройка Drive: `docs/m1.md`; фактические результаты: `docs/m1_validation.md`.

1. Выбрать один опубликованный commit для Mac и Colab и новое одинаковое `SESSION_NAME`. Запустить один Mac `queue-worker` с папкой `My Drive/qwen-repair-exchange/sessions/<SESSION_NAME>`.
2. Открыть `notebooks/01_m1_feasibility.ipynb` в CPU runtime. Заполнить полный `COMMIT_SHA` и `SESSION_NAME`, оставить `RUN_INFERENCE=False` и `RUN_TRAINING=False`. Выполнить базовую установку, CPU doctor и подключение Drive.
3. Выполнить `executor-preflight` и `executor-probe` из Colab с созданным профилем сессии. Подтвердить настоящий запрос к Mac и измерить round-trip latency отдельно от Docker. Скачать отчёты, остановить worker и оценить нагрузку Drive. При ошибке или неприемлемой задержке остановиться на проверке связи.
4. После успешного транспорта провести отдельный запуск на T4 с `RUN_INFERENCE=True`. Сохранить GPU doctor: GPU, VRAM, Python, torch/CUDA и версии. Выполнить inference закреплённой `Qwen/Qwen3-0.6B` на авторских fixtures; проверить `environment.json`, точные token IDs и отдельную private evaluation.
5. Отдельно включить `RUN_TRAINING` и выполнить SFT/GRPO smoke: первый процесс сохраняет checkpoint после шага 1, второй продолжает до шага 2. Проверить итоговый `global_step`, adapter round-trip, masks/logprobs и отчёт о дисперсии rewards. Успешный inference не заменяет проверки обучения.
6. Скачать полный ZIP с отчётами и checkpoints до завершения runtime, затем остановить worker. Сопоставить VRAM, скорость генерации, время эпизодов и executor с результатами CPU-проверок. При необходимости сохранить завершённый архив в Drive.
7. При успешном коротком цикле отдельно оценить inference на уже проверенном KodCode-пилоте. Для этого перенести весь validated bundle и запустить worker с соответствующим registry. Массовая обработка и финальный аудит датасета пока не нужны.

До настоящего обучения отдельно проверить, что hidden feedback не влияет на prompt, stopping или выбор кандидата, а инфраструктурные сбои не получают reward=0. Проверки M0 останутся регрессионными тестами протокола, но не заменят preflight нового backend.

Версии ML-стека закреплены и проверены на маленькой CPU-модели. Docker preflight и небольшой KodCode-пилот выполнены локально. Эти проверки не заменяют T4 preflight, реальный rollout Qwen и измерение связи с Colab. Публикация данных, длительное обучение и платные сервисы не входят в текущую итерацию.

# Проверки M1

Дата: 29 октября 2026. Полный GPU feasibility ещё не завершён. Ниже разделены фактически выполненные проверки и работа, для которой нужен Colab T4.

Итоговый suite: **124 passed** за 38,95 с, включая настоящий Docker, закреплённый tokenizer и tiny CPU training/resume. TRL выдал ожидаемое предупреждение об экспериментальном `rollout_func`; API закреплён и проверен. `pip check` в обоих окружениях не обнаружил конфликтов. Wheel собран и установлен отдельно от checkout: все 38 исходных файлов и fixtures совпали побайтно, CLI help и M0 demo/summarize прошли.

## Выполнено локально

| Проверка | Результат и границы |
| --- | --- |
| Регрессия M0 | CLI demo/summarize совпали, synthetic final-success 2/4, infra исключён |
| CPU и ML-зависимости | Отдельные `.venv` и `.venv-ml`; pinned ML packages импортируются совместно |
| Настоящий tokenizer Qwen | Загружен только tokenizer закреплённой revision; проверены non-thinking template, точные IDs, сохранение префикса |
| Docker isolation | Docker Engine 29.9.2, pinned `python:3.13-slim`; все 10 probes прошли |
| Авторские функции | 16 reference/buggy public/private проверок прошли; tuple и поддельный stdout-verdict отвергнуты |
| Файловая очередь | Реальный CLI worker/probe на локальной файловой системе: 1,0083 с round trip, 0,6583 с executor; остановка worker отзывает readiness |
| Папка Google Drive | После исправления повторной публикации и чтения heartbeat два последовательных запроса прошли: round trip 1,5112 и 1,0092 с; полное время CLI 1,7623 и 1,2103 с. После паузы 143,62 с третий запрос тоже прошёл: round trip 1,0073 с, CLI 1,2398 с. Это ещё не измерение передачи между Colab и Mac |
| KodCode | Загружены 100 строк pinned revision; подготовлены 10, в Docker валидированы 7; 3 исключены без ошибки одновременно на public/private |
| SFT trainer | Маленькая случайная CPU Qwen3-архитектура, LoRA, реальные optimizer steps; новый процесс продолжил checkpoint с шага 1 до 2 |
| GRPO trainer | Стандартный TRL с маленькой CPU-моделью, две попытки и настоящие optimizer steps; отдельные тесты контролируют действия и verdict для проверки математики |
| Masks | Градиенты feedback/padding равны нулю; градиенты действий, включая вторую попытку после EOS, ненулевые |
| Adapter и resume | Сохранённый adapter после загрузки даёт те же probe logits; GRPO новый процесс пересобирает свежие группы после checkpoint |
| Notebook | Валидный JSON, компилируемые Python-ячейки, outputs отсутствуют; модель и trainer вызываются только через CLI |

Источники локальных результатов, исключённые из Git:

- `runs/m1-executor-validation-6a9fe5d5/report.json`;
- `runs/m1-cli-queue-62ec7497/report.json`;
- `runs/m1-drive-local-probe.json`;
- `runs/m1-drive-repaired-probe-2.json`, `runs/m1-drive-repaired-probe-3.json`, `runs/m1-drive-repaired-timings.json`;
- `runs/m1-drive-repaired-probe-4.json`, `runs/m1-drive-repaired-after-pause.json`;
- `data/m1-pilot-validated/manifest.json` и `validation.json`;
- `runs/m1-m0-regression/`;
- `runs/m1-local-gpu-preflight.json`: корректный отказ CPU-окружения без torch/CUDA.

KodCode validation выполнила 40 reference bundle runs, все успешны, и 34 mutation bundle runs. Суммарное измеренное время executor составило 113,9548 с; это не полный wall time команды и не оценка GPU throughput. Инфраструктурных сбоев не было. Данные не публиковались; кандидаты исполнялись только в Docker.

Первый длительный запуск worker в Drive обнаружил ошибку: сбой чтения существующего ответа приводил к попытке записать другой payload поверх неизменяемого файла. Исправленный worker сохраняет точный ответ для повторной публикации и не выполняет запрос заново при частичном или повреждённом ответе. Отдельные regression tests покрывают эти случаи, перезапуск и ограничение кэша. Старые heartbeat теперь отбрасываются по timestamp в имени до чтения содержимого; актуальность проверяется после I/O. Для этого формата обе стороны должны использовать новый commit.

Загрузчик Parquet дополнительно проверен повторным чтением 100 строк: процесс завершился с кодом 0, данные побайтно совпали с первоначальным срезом, SHA256 `5e05383ab17b94ae81e5db8523961230a1651eb9bb8dfdf9be85695ac863d39f`. Фоновое prefetch отключено, reader и файл закрываются явно.

Исходник установленного TRL 1.16.0 совпадает с [тегом v1.16.0](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_trainer.py): SHA256 `1c3dc2253fb3653269c11dfce4f946f74201a4cb63e0887a37e8e3bb4a3f5503`. `rollout_func` остаётся экспериментальным API, поэтому версия закреплена и проверяется перед обучением.

## Ещё не выполнено

- Загрузка Qwen3-0.6B и генерация на T4, реальные значения VRAM и tokens/sec.
- Round trip между Colab и Mac через синхронизируемую папку; локальный queue test этого не проверяет.
- SFT/GRPO updates именно Qwen3-0.6B на T4 с живым Docker worker, их rewards и дисперсия.
- GPU checkpoint/resume и восстановление после потери Colab runtime.
- Полный аудит групп/близких дубликатов KodCode и исследовательская оценка качества.

Следующая проверка выполняется через `notebooks/01_m1_feasibility.ipynb` после настройки общей папки и запуска локального worker. M1 закрывается по этим результатам, а не по наличию кода или маленьким CPU-тестам.

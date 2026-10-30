# Love in Every Line

Романтический сайт: на пустом листе линия за линией (реальные штрихи на Canvas) рисуется портрет.
`/` — подготовленный портрет, `/studio` — загрузка своего фото (FastAPI генерирует штрихи локально).

## Локальный запуск
```
npm install
python3 -m venv tools/.venv && tools/.venv/bin/pip install -r backend/requirements.txt pytest httpx
# модели (face landmarker, selfie segmenter, ONNX-эскиз) скачиваются автоматически в tools/models при первом запуске
cd backend && ../tools/.venv/bin/python -m uvicorn app:app --port 8000   # терминал 1
npm run dev                                                              # терминал 2: http://localhost:5173 (/api проксируется на :8000)
```
Портрет для главной: `python backend/pipeline.py assets/girlfriend.jpg public/strokes.json` (локально, ~15-20 с).
Тесты: `cd backend && ../tools/.venv/bin/python -m pytest tests -q`. Сборка фронтенда: `npm run build`.

## Docker
```
cp .env.example .env
docker compose up -d --build        # http://localhost
docker compose logs -f backend
```
Модели вшиты в образ backend (сборка падает, если файла нет; `LIE_OFFLINE=1` запрещает скачивание в рантайме).
Перед сборкой модели должны лежать в `tools/models/` (3 файла: `face_landmarker.task`, `selfie_segmenter.tflite`, `informative_anime_512.onnx`) — они появляются после локального запуска пайплайна.

## Переменные окружения (`.env`)
| Переменная | По умолчанию | Смысл |
|---|---|---|
| `MAX_UPLOAD_MB` | 8 | максимальный размер файла |
| `MAX_CONCURRENT_JOBS` | 1 | параллельные генерации (~0.7 ГБ RAM, ~15-20 с CPU каждая); лишние получают 503 |
| `JOB_TIMEOUT_SEC` | 90 | таймаут подпроцесса обработки (504) |
| `RATE_LIMIT_PER_MIN` | 4 | запросов /api/draw с IP в минуту (429); nginx ограничивает ещё и сам |
| `TRUST_PROXY` | 1 | за nginx брать IP из X-Forwarded-For |
| `ENABLE_DOCS` | 0 | 1 = открыть /api/docs |

## Linux-сервер (Ubuntu/Debian)
```
curl -fsSL https://get.docker.com | sh          # Docker + compose plugin
git clone <репозиторий> love && cd love          # или rsync проекта
# скопируйте tools/models/*.{task,tflite,onnx} и public/strokes.json, если их нет в репозитории
cp .env.example .env
docker compose up -d --build
curl -s localhost/api/health                     # {"status":"ok"}
```
Минимум: 2 vCPU, 2 ГБ RAM (backend ограничен 1.5 ГБ).

## Домен и HTTPS (вручную)
1. В DNS домена создайте A-запись на IP сервера (AAAA при IPv6), откройте порты 80 и 443.
2. Выпуск сертификата (пока сайт работает по HTTP): `sudo apt install certbot && sudo certbot certonly --webroot -w /var/www/certbot -d ВАШ.ДОМЕН`
   (или `sudo certbot certonly --standalone -d ВАШ.ДОМЕН`, остановив `docker compose stop web`).
3. Замените `nginx/default.conf` содержимым `nginx/https.conf.example` (подставьте домен), в `docker-compose.yml` у `web` добавьте
   `ports: ["80:80","443:443"]` и тома `/etc/letsencrypt:/etc/letsencrypt:ro`, `/var/www/certbot:/var/www/certbot`.
4. `docker compose up -d --build web`. Продление: `certbot renew` по cron/timer, затем `docker compose exec web nginx -s reload`.

## Безопасность
- Backend не публикуется в интернет (`expose`, только внутренняя сеть compose); наружу только nginx.
- Загрузка: лимит размера (413), проверка реального содержимого через Pillow (415/422), разрешение 400…6000 px и ≤24 МП, таймаут (504), один параллельный job (503), rate limit (429, backend + nginx).
- Фото обрабатывается локально, во внешние API не отправляется, лежит только во временной папке запроса (tmpfs, read-only ФС) и удаляется сразу после ответа.
- Контейнер без root, `cap_drop: ALL`, `no-new-privileges`.
- Ограничение: rate limit в памяти одного процесса (один worker). При масштабировании — Redis.

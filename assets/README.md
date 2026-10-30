# assets/ — приватные фотографии

Положи сюда фото: **assets/girlfriend.jpg** (также подойдут .jpeg / .png / .webp).
Содержимое папки (кроме этого файла) исключено из git и никогда не попадает в `dist/`.
Фото обрабатывается локально, никуда не загружается.

## Ручная маска регионов (необязательно, для экспериментальных профилей A/B/C)
`assets/girlfriend.regions.png` — картинка того же кадра, что и фото: красным закрась волосы (по умолчанию всё, что не кожа, считается тканью).
Запуск: `tools/.venv/bin/python tools/make_strokes_v4.py assets/girlfriend.jpg --profile B --out tools/out/variants/B.json`

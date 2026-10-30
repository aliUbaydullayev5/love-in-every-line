/** Контракт stroke data (формат strokes.json). Пишет tools/make_strokes.py, читает движок. */
export interface Stroke {
  id: number
  cat: 'contour' | 'feature' | 'hatch' | 'cross' | 'shadow'
  order: number // порядок рисования
  w: number // толщина, px в системе координат холста
  o: number // прозрачность 0..1
  color: string // CSS-цвет
  dur: number // вес длительности (относительные единицы)
  pts: number[] // [x0,y0,x1,y1,...]
}

export interface StrokeData {
  version: 1
  width: number
  height: number
  strokes: Stroke[]
}

export function validateStrokeData(raw: unknown): StrokeData {
  const d = raw as StrokeData
  if (!d || d.version !== 1 || !(d.width > 0) || !(d.height > 0) || !Array.isArray(d.strokes) || d.strokes.length === 0)
    throw new Error('Некорректный файл штрихов')
  for (const s of d.strokes) {
    if (!Array.isArray(s.pts) || s.pts.length < 4 || s.pts.length % 2 || !(s.w > 0) || !(s.dur > 0))
      throw new Error(`Некорректный штрих #${s?.id}`)
  }
  return d
}

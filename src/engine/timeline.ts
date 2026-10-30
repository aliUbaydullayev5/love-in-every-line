import type { Stroke } from './types'

export interface Scheduled {
  stroke: Stroke
  start: number // сек
  end: number
  cum: Float32Array // накопленная длина вершин (нормированная 0..1)
}

/**
 * Раскладывает штрихи по времени: старты равномерно по порядку (с весом длины),
 * несколько штрихов рисуются одновременно. Весь процесс ровно укладывается в totalSec.
 */
export function buildTimeline(strokes: Stroke[], totalSec: number, concurrency = 8): Scheduled[] {
  const sorted = [...strokes].sort((a, b) => a.order - b.order)
  const lens = sorted.map((s) => {
    let l = 0
    for (let i = 2; i < s.pts.length; i += 2) l += Math.hypot(s.pts[i]! - s.pts[i - 2]!, s.pts[i + 1]! - s.pts[i - 1]!)
    return Math.max(l, 1)
  })
  const sumLen = lens.reduce((a, b) => a + b, 0)
  const meanDur = (totalSec / sorted.length) * concurrency
  const durs = sorted.map((s, i) => Math.min(meanDur * 3, Math.max(0.05, ((lens[i]! / sumLen) * sorted.length * meanDur * s.dur))))
  const maxDur = Math.max(...durs)
  const span = Math.max(totalSec - maxDur, totalSec * 0.5)
  let acc = 0
  const out = sorted.map((stroke, i): Scheduled => {
    const start = (acc / sumLen) * span
    acc += lens[i]!
    const cum = new Float32Array(stroke.pts.length / 2)
    let c = 0
    for (let k = 1; k < cum.length; k++) {
      c += Math.hypot(stroke.pts[2 * k]! - stroke.pts[2 * k - 2]!, stroke.pts[2 * k + 1]! - stroke.pts[2 * k - 1]!)
      cum[k] = c
    }
    const last = cum[cum.length - 1]! || 1
    for (let k = 0; k < cum.length; k++) cum[k] = cum[k]! / last
    return { stroke, start, end: start + durs[i]!, cum }
  })
  // нормировка: последний штрих заканчивается ровно в totalSec
  const end = Math.max(...out.map((s) => s.end))
  const f = totalSec / end
  for (const s of out) {
    s.start *= f
    s.end *= f
  }
  return out
}

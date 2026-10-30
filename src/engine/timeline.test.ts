import { describe, expect, it } from 'vitest'
import { buildTimeline } from './timeline'
import { validateStrokeData, type Stroke } from './types'

const mk = (id: number, order: number): Stroke => ({ id, cat: 'hatch', order, w: 1, o: 0.5, color: '#222', dur: 1, pts: [0, 0, 10 + id, 5, 20, 20] })

describe('timeline', () => {
  const strokes = Array.from({ length: 2000 }, (_, i) => mk(i, 2000 - i))
  const tl = buildTimeline(strokes, 24)
  it('укладывается в длительность', () => {
    expect(Math.max(...tl.map((s) => s.end))).toBeCloseTo(24, 5)
    expect(Math.min(...tl.map((s) => s.start))).toBe(0)
  })
  it('сортирует по order и старты не убывают', () => {
    expect(tl[0]!.stroke.order).toBe(1)
    for (let i = 1; i < tl.length; i++) expect(tl[i]!.start).toBeGreaterThanOrEqual(tl[i - 1]!.start)
  })
  it('cum нормирован 0..1', () => {
    expect(tl[0]!.cum[0]).toBe(0)
    expect(tl[0]!.cum[tl[0]!.cum.length - 1]).toBeCloseTo(1)
  })
  it('валидатор отвергает мусор', () => {
    expect(() => validateStrokeData({})).toThrow()
    expect(() => validateStrokeData({ version: 1, width: 1, height: 1, strokes: [{ id: 1, pts: [1], w: 1, dur: 1 }] })).toThrow()
  })
})

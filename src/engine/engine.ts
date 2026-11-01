import { buildTimeline, type Scheduled } from './timeline'
import type { StrokeData } from './types'

const PAPER = '#fbf6ea' // тот же цвет бумаги, что --color-paper в index.css

export type EngineState = 'idle' | 'playing' | 'paused' | 'done'

/** Canvas-рендерер: каждый штрих физически растёт вдоль своей траектории, нарисованное остаётся. */
export class StrokeEngine {
  private ctx: CanvasRenderingContext2D
  private tl: Scheduled[]
  private drawn: Float32Array // доля, уже нарисованная для каждого штриха
  private t = 0
  private last = 0
  private raf = 0
  private nextIdx = 0 // первый штрих, который ещё не завершён
  state: EngineState = 'idle'
  speed = 1
  onState?: (s: EngineState) => void
  onProgress?: (p: number) => void

  constructor(private canvas: HTMLCanvasElement, private data: StrokeData, public duration: number) {
    this.ctx = canvas.getContext('2d')!
    this.tl = buildTimeline(data.strokes, duration)
    this.drawn = new Float32Array(this.tl.length)
    this.resize()
  }

  /** Подгоняет размер буфера под CSS-размер и dpr; перерисовывает нарисованное. */
  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const r = this.canvas.getBoundingClientRect()
    const w = Math.max(1, Math.round(r.width * dpr))
    if (this.canvas.width === w && this.canvas.height) return
    this.canvas.width = w
    this.canvas.height = Math.round((w * this.data.height) / this.data.width)
    this.redraw()
  }

  private scale() {
    return this.canvas.width / this.data.width
  }

  private segment(i: number, from: number, to: number, ctx: CanvasRenderingContext2D = this.ctx, k: number = this.scale()) {
    const s = this.tl[i]!
    const { pts, w, o, color } = s.stroke
    const n = s.cum.length
    ctx.strokeStyle = color
    ctx.globalAlpha = o
    ctx.lineWidth = w * k
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    const point = (f: number): [number, number] => {
      let j = 1
      while (j < n - 1 && s.cum[j]! < f) j++
      const a = s.cum[j - 1]!
      const u = s.cum[j]! > a ? (f - a) / (s.cum[j]! - a) : 1
      return [pts[2 * j - 2]! + (pts[2 * j]! - pts[2 * j - 2]!) * u, pts[2 * j - 1]! + (pts[2 * j + 1]! - pts[2 * j - 1]!) * u]
    }
    const [x0, y0] = point(from)
    ctx.moveTo(x0 * k, y0 * k)
    for (let j = 0; j < n; j++) if (s.cum[j]! > from && s.cum[j]! < to) ctx.lineTo(pts[2 * j]! * k, pts[2 * j + 1]! * k)
    const [x1, y1] = point(to)
    ctx.lineTo(x1 * k, y1 * k)
    ctx.stroke()
  }

  private advance(t: number) {
    for (let i = this.nextIdx; i < this.tl.length; i++) {
      const s = this.tl[i]!
      if (s.start > t) break // отсортированы по start
      const f = Math.min(1, (t - s.start) / (s.end - s.start))
      if (f > this.drawn[i]!) {
        this.segment(i, this.drawn[i]!, f)
        this.drawn[i] = f
      }
      if (i === this.nextIdx && this.drawn[i]! >= 1) this.nextIdx++
    }
  }

  /** Полная перерисовка до текущего момента (resize). */
  private redraw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.drawn.fill(0)
    this.nextIdx = 0
    this.advance(this.t)
  }

  /**
   * Готовый портрет в PNG: те же штрихи тем же кодом, но целиком и в высоком разрешении, на бумаге (не прозрачный фон).
   * Как на экране: штрихи на прозрачном слое, затем multiply на цвет бумаги.
   */
  exportPng(width = 2160): Promise<Blob> {
    const k = width / this.data.width
    const h = Math.round(this.data.height * k)
    const ink = document.createElement('canvas')
    ink.width = width
    ink.height = h
    const ictx = ink.getContext('2d')!
    for (let i = 0; i < this.tl.length; i++) this.segment(i, 0, 1, ictx, k)
    const out = document.createElement('canvas')
    out.width = width
    out.height = h
    const c = out.getContext('2d')!
    c.fillStyle = PAPER
    c.fillRect(0, 0, width, h)
    c.globalCompositeOperation = 'multiply'
    c.drawImage(ink, 0, 0)
    return new Promise((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error('PNG export failed'))), 'image/png'))
  }

  private setState(s: EngineState) {
    this.state = s
    this.onState?.(s)
  }

  private frame = (now: number) => {
    const dt = Math.min(0.1, (now - this.last) / 1000)
    this.last = now
    this.t = Math.min(this.duration, this.t + dt * this.speed)
    this.advance(this.t)
    this.onProgress?.(this.t / this.duration)
    if (this.t >= this.duration) {
      this.advance(this.duration + 1)
      this.setState('done')
      return
    }
    this.raf = requestAnimationFrame(this.frame)
  }

  play() {
    if (this.state === 'playing') return
    if (this.state === 'done') this.reset()
    this.last = performance.now()
    this.setState('playing')
    this.raf = requestAnimationFrame(this.frame)
  }
  pause() {
    if (this.state !== 'playing') return
    cancelAnimationFrame(this.raf)
    this.setState('paused')
  }
  reset() {
    cancelAnimationFrame(this.raf)
    this.t = 0
    this.drawn.fill(0)
    this.nextIdx = 0
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.onProgress?.(0)
    this.setState('idle')
  }
  replay() {
    this.reset()
    this.play()
  }
  destroy() {
    cancelAnimationFrame(this.raf)
  }
}

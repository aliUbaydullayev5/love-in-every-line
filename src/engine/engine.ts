import { buildTimeline, type Scheduled } from './timeline'
import type { StrokeData } from './types'

const PAPER = '#fbf6ea' // тот же цвет бумаги, что --color-paper в index.css

/** Лучший доступный формат записи: MP4 (H.264), иначе WebM. null — браузер не умеет записывать canvas. */
export function pickVideoMime(): { mime: string; ext: 'mp4' | 'webm' } | null {
  if (typeof MediaRecorder === 'undefined' || !('captureStream' in HTMLCanvasElement.prototype)) return null
  const list: [string, 'mp4' | 'webm'][] = [
    ['video/mp4;codecs=avc1.640028', 'mp4'],
    ['video/mp4;codecs=avc1.42E01E', 'mp4'],
    ['video/mp4;codecs=avc1', 'mp4'],
    ['video/mp4', 'mp4'],
    ['video/webm;codecs=vp9', 'webm'],
    ['video/webm;codecs=vp8', 'webm'],
    ['video/webm', 'webm'],
  ]
  for (const [mime, ext] of list) if (MediaRecorder.isTypeSupported(mime)) return { mime, ext }
  return null
}

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

  /**
   * Запись самого процесса рисования: штрихи растут тем же кодом (segment) на отдельном холсте с бумагой,
   * кадры пишет MediaRecorder в реальном времени (длительность видео = длительность анимации).
   * Запускается только по запросу; на экранную анимацию не влияет.
   */
  recordVideo(opts: { width?: number; fps?: number; onProgress?: (p: number) => void; signal?: AbortSignal } = {}): Promise<{ blob: Blob; ext: 'mp4' | 'webm' }> {
    const fmt = pickVideoMime()
    if (!fmt) return Promise.reject(new Error('Запись видео не поддерживается этим браузером'))
    const { width = 1080, fps = 30, onProgress, signal } = opts
    const k = width / this.data.width
    const w = Math.round(width / 2) * 2 // H.264 требует чётных размеров
    const h = Math.round((this.data.height * k) / 2) * 2
    const ink = document.createElement('canvas')
    ink.width = w
    ink.height = h
    const ictx = ink.getContext('2d')!
    const out = document.createElement('canvas')
    out.width = w
    out.height = h
    const octx = out.getContext('2d')!
    const compose = () => {
      octx.globalCompositeOperation = 'source-over'
      octx.globalAlpha = 1
      octx.fillStyle = PAPER
      octx.fillRect(0, 0, w, h)
      octx.globalCompositeOperation = 'multiply'
      octx.drawImage(ink, 0, 0)
    }
    const drawn = new Float32Array(this.tl.length)
    let next = 0
    const advance = (t: number) => {
      for (let i = next; i < this.tl.length; i++) {
        const s = this.tl[i]!
        if (s.start > t) break
        const f = Math.min(1, (t - s.start) / (s.end - s.start))
        if (f > drawn[i]!) {
          this.segment(i, drawn[i]!, f, ictx, k)
          drawn[i] = f
        }
        if (i === next && drawn[i]! >= 1) next++
      }
    }
    compose()
    const stream = out.captureStream(fps)
    const rec = new MediaRecorder(stream, { mimeType: fmt.mime, videoBitsPerSecond: 8_000_000 })
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    return new Promise((resolve, reject) => {
      let aborted = false
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        if (aborted) reject(new DOMException('Запись отменена', 'AbortError'))
        else resolve({ blob: new Blob(chunks, { type: fmt.mime.split(';')[0] }), ext: fmt.ext })
      }
      rec.onerror = () => reject(new Error('Ошибка записи видео'))
      rec.start(1000)
      const t0 = performance.now()
      const tick = (now: number) => {
        if (signal?.aborted) {
          aborted = true
          if (rec.state !== 'inactive') rec.stop()
          return
        }
        const t = (now - t0) / 1000
        advance(Math.min(t, this.duration))
        compose()
        onProgress?.(Math.min(1, t / this.duration))
        if (t >= this.duration + 1.5) { // пауза на готовом рисунке, затем конец
          advance(this.duration + 1)
          compose()
          rec.stop()
          return
        }
        requestAnimationFrame(tick)
      }
      requestAnimationFrame(tick)
    })
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

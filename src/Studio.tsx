import { useEffect, useRef, useState } from 'react'
import { config } from './config'
import { StrokeEngine, type EngineState } from './engine/engine'
import { validateStrokeData } from './engine/types'

const MAX_BYTES = 8 * 1024 * 1024
const TYPES = ['image/jpeg', 'image/png', 'image/webp']
const STATUS_MSG: Record<number, string> = {
  413: 'Файл слишком большой (максимум 8 МБ).',
  415: 'Формат не поддерживается. Нужен JPG, PNG или WebP.',
  422: 'Не удалось обработать это фото. Попробуйте другое: портрет, хорошее освещение, от 400 px.',
  429: 'Слишком часто. Подождите минуту и попробуйте снова.',
  503: 'Сервер занят, попробуйте через минуту.',
  504: 'Обработка заняла слишком много времени. Попробуйте фото поменьше.',
}

type Step = 'pick' | 'busy' | 'draw'

// ponytail: проверка размера/типа/декодирования только на клиенте для удобства; настоящая защита — на сервере
function checkImage(file: File): Promise<string | null> {
  if (!TYPES.includes(file.type)) return Promise.resolve('Нужен файл JPG, PNG или WebP.')
  if (file.size > MAX_BYTES) return Promise.resolve('Файл больше 8 МБ.')
  return new Promise((res) => {
    const url = URL.createObjectURL(file)
    const im = new Image()
    im.onload = () => {
      URL.revokeObjectURL(url)
      res(Math.min(im.naturalWidth, im.naturalHeight) < 400 ? 'Фото слишком маленькое: нужно минимум 400 px по меньшей стороне.' : null)
    }
    im.onerror = () => {
      URL.revokeObjectURL(url)
      res('Не удалось прочитать изображение.')
    }
    im.src = url
  })
}

export default function Studio() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const engine = useRef<StrokeEngine | null>(null)
  const abort = useRef<AbortController | null>(null)
  const [step, setStep] = useState<Step>('pick')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [err, setErr] = useState('')
  const [aspect, setAspect] = useState(0.8)
  const [state, setState] = useState<EngineState>('idle')
  const [data, setData] = useState<ReturnType<typeof validateStrokeData> | null>(null)

  useEffect(() => () => { engine.current?.destroy(); abort.current?.abort() }, [])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  // движок создаётся, когда canvas появился в DOM
  useEffect(() => {
    if (step !== 'draw' || !data || !canvas.current) return
    const eng = new StrokeEngine(canvas.current, data, config.durationSec)
    eng.onState = setState
    engine.current = eng
    eng.play()
    const onResize = () => eng.resize()
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      eng.destroy()
      engine.current = null
    }
  }, [step, data])

  async function onPick(f: File | undefined) {
    setErr('')
    if (!f) return
    const bad = await checkImage(f)
    if (bad) { setErr(bad); return }
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  async function start() {
    if (!file) return
    const ctl = new AbortController()
    abort.current = ctl
    const timer = setTimeout(() => ctl.abort(), 120_000)
    setErr('')
    setStep('busy')
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await fetch('/api/draw', { method: 'POST', body: fd, signal: ctl.signal })
      if (!r.ok) {
        const j = await r.json().catch(() => null)
        throw new Error(typeof j?.detail === 'string' ? j.detail : STATUS_MSG[r.status] ?? `Ошибка сервера (${r.status}).`)
      }
      const d = validateStrokeData(await r.json())
      setAspect(d.width / d.height)
      setData(d)
      setFile(null) // оригинал больше не нужен
      setPreview('')
      setStep('draw')
    } catch (e) {
      setErr(ctl.signal.aborted ? 'Время ожидания истекло или обработка отменена.' : (e as Error).message || 'Не удалось связаться с сервером.')
      setStep('pick')
    } finally {
      clearTimeout(timer)
    }
  }

  const reset = () => { setData(null); setFile(null); setPreview(''); setErr(''); setStep('pick') }
  const toggle = () => (state === 'playing' ? engine.current?.pause() : engine.current?.play())

  return (
    <main className="flex min-h-full w-full flex-col items-center px-4 pb-24 pt-10 text-center">
      <a className="ctl mb-6" href="/">← На главную</a>
      <h1 className="font-serif text-4xl italic sm:text-5xl">Нарисовать своё</h1>

      {step !== 'draw' && (
        <p className="mt-3 max-w-md font-sans text-sm font-light text-graphite/70">
          Загрузите портрет, и он нарисуется линия за линией. Фото обрабатывается на сервере и не сохраняется.
        </p>
      )}

      {step === 'pick' && (
        <section className="mt-8 flex w-full max-w-sm flex-col items-center gap-5">
          <label className="ctl cursor-pointer border border-rose/40" aria-label="Выбрать фото">
            {file ? 'Выбрать другое' : 'Выбрать фото'}
            <input className="sr-only" type="file" accept={TYPES.join(',')} onChange={(e) => onPick(e.target.files?.[0])} />
          </label>
          {preview && (
            <img src={preview} alt="Предпросмотр выбранного фото" className="max-h-[50dvh] max-w-full rounded-[3px] shadow-md" />
          )}
          {file && <button className="ctl border border-rose/40 !text-rose" onClick={start}>Начать рисование</button>}
          {err && <p role="alert" className="font-sans text-sm text-rose">{err}</p>}
        </section>
      )}

      {step === 'busy' && (
        <section className="mt-12 flex flex-col items-center gap-5" aria-live="polite">
          <p className="font-serif text-2xl italic">Рисуем линии…</p>
          <p className="font-sans text-xs font-light uppercase tracking-[0.25em] text-rose">обычно 15–30 секунд</p>
          <button className="ctl" onClick={() => abort.current?.abort()}>Отмена</button>
        </section>
      )}

      {step === 'draw' && (
        <>
          <div className="paper relative mt-6 max-w-full rounded-[3px]" style={{ aspectRatio: aspect, height: `min(70dvh, calc(88vw / ${aspect}))` }}>
            <canvas ref={canvas} className="block h-full w-full mix-blend-multiply" aria-label="Рисунок, нарисованный линиями" />
          </div>
          {state === 'done' && <p className="mt-5 font-serif text-xl italic sm:text-3xl">{config.finalText}</p>}
          <nav className="mt-6 flex flex-wrap justify-center gap-1">
            {state !== 'done' && <button className="ctl" onClick={toggle}>{state === 'playing' ? 'Пауза' : 'Продолжить'}</button>}
            <button className="ctl" onClick={() => engine.current?.replay()}>Заново</button>
            <button className="ctl" onClick={reset}>Другое фото</button>
          </nav>
        </>
      )}
    </main>
  )
}

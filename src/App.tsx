import { useCallback, useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import { config } from './config'
import { StrokeEngine, type EngineState } from './engine/engine'
import { validateStrokeData } from './engine/types'
import { usePortraitDownload, useVideoDownload } from './usePortraitDownload'

type Phase = 'intro' | 'drawing' | 'final'
const SPEEDS = [1, 1.5, 2]

export default function App() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const engine = useRef<StrokeEngine | null>(null)
  const bar = useRef<HTMLDivElement>(null)
  const root = useRef<HTMLDivElement>(null)
  const [phase, setPhase] = useState<Phase>('intro')
  const [state, setState] = useState<EngineState>('idle')
  const [speed, setSpeed] = useState(1)
  const [err, setErr] = useState('')
  const [aspect, setAspect] = useState(0.8)

  // загрузка + движок
  useEffect(() => {
    let dead = false
    let eng: StrokeEngine | null = null
    fetch(config.strokesUrl)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`strokes.json: HTTP ${r.status}`))))
      .then((raw) => {
        if (dead || !canvas.current) return
        const data = validateStrokeData(raw)
        setAspect(data.width / data.height)
        requestAnimationFrame(() => {
          if (dead || !canvas.current) return
          eng = new StrokeEngine(canvas.current, data, config.durationSec)
          eng.onState = (s) => {
            setState(s)
            if (s === 'done') setPhase('final')
          }
          eng.onProgress = (p) => bar.current && (bar.current.style.transform = `scaleX(${p})`)
          engine.current = eng
          startIntro()
        })
      })
      .catch((e: Error) => setErr(e.message))
    const onResize = () => eng?.resize()
    window.addEventListener('resize', onResize)
    return () => {
      dead = true
      eng?.destroy()
      window.removeEventListener('resize', onResize)
    }
  }, [])

  function startIntro() {
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
    const tl = gsap.timeline({ onComplete: () => { setPhase('drawing'); engine.current?.play() } })
    tl.fromTo('.intro-a', { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: reduced ? 0.01 : 1.4, ease: 'power2.out' })
      .fromTo('.intro-b', { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: reduced ? 0.01 : 1.2, ease: 'power2.out' }, '-=0.5')
      .to('.intro', { opacity: 0, duration: reduced ? 0.01 : 0.9, delay: reduced ? 0 : 1.1, ease: 'power1.inOut' })
      .fromTo('.sheet', { opacity: 0, y: 30, scale: 0.97 }, { opacity: 1, y: 0, scale: 1, duration: reduced ? 0.01 : 1.4, ease: 'power3.out' }, '-=0.3')
  }

  // финал
  useEffect(() => {
    if (phase !== 'final') return
    const tl = gsap.timeline()
    tl.fromTo('.final-text', { opacity: 0, y: 16, letterSpacing: '0.08em' }, { opacity: 1, y: 0, letterSpacing: '0.01em', duration: 2.2, ease: 'power2.out' })
    tl.fromTo('.final-name', { opacity: 0 }, { opacity: 1, duration: 1.6 }, '-=1.2')
    root.current?.querySelectorAll('.petal').forEach((p, i) => {
      // лепестки идут по боковым полосам (3–25vw и 75–97vw), чтобы не пролетать над лицом в центре
      const side = i % 2 === 0 ? 3 + Math.random() * 22 : 75 + Math.random() * 22
      const spin = (Math.random() < 0.5 ? -1 : 1) * (300 + Math.random() * 320)
      gsap.fromTo(p, { x: `${side}vw`, y: '-5vh', opacity: 0, rotate: 0, scale: 0.8 + Math.random() * 0.45 },
        { y: '105vh', x: `+=${(Math.random() - 0.5) * 90}`, rotate: spin, opacity: 0.55, duration: 9 + Math.random() * 5, delay: i * 0.5, ease: 'none', repeat: -1, repeatDelay: Math.random() * 1.5 })
    })
    return () => { tl.kill(); gsap.killTweensOf('.petal') }
  }, [phase])

  const getEngine = useCallback(() => engine.current, [])
  const dl = usePortraitDownload(getEngine, phase === 'final')
  const vid = useVideoDownload(getEngine, phase === 'final')
  const toggle = () => (state === 'playing' ? engine.current?.pause() : engine.current?.play())
  const cycleSpeed = () => {
    const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]!
    setSpeed(next)
    if (engine.current) engine.current.speed = next
  }
  const replay = () => {
    gsap.set('.final-text, .final-name', { opacity: 0 })
    setPhase('drawing')
    engine.current?.replay()
  }

  return (
    <div ref={root} className="relative flex h-full w-full flex-col items-center justify-center overflow-hidden px-4">
      {err && <p className="font-serif text-xl text-rose">Не удалось загрузить рисунок: {err}</p>}

      <div className="intro pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center text-center">
        <p className="intro-a font-serif text-4xl italic text-graphite opacity-0 sm:text-6xl">{config.herName}</p>
        <p className="intro-b mt-4 font-sans text-xs font-light uppercase tracking-[0.4em] text-rose opacity-0">{config.introLine}</p>
      </div>

      <div className="sheet paper relative max-h-[78dvh] max-w-full rounded-[3px] opacity-0" style={{ aspectRatio: aspect, height: 'min(78dvh, calc(88vw / ' + aspect + '))' }}>
        <canvas ref={canvas} className="block h-full w-full mix-blend-multiply" aria-label="Портрет, рисуемый линиями" />
      </div>

      <div className={`mt-6 min-h-[84px] text-center transition-opacity duration-700 ${phase === 'intro' ? 'opacity-0' : 'opacity-100'}`}>
        <p className="final-text font-serif text-2xl italic text-graphite opacity-0 sm:text-4xl">{config.finalText}</p>
        <p className="final-name mt-2 font-sans text-[11px] font-light uppercase tracking-[0.35em] text-rose opacity-0">{config.herName}</p>
        {(vid.state === 'recording' || vid.error) && <p className="mt-2 font-sans text-[10px] font-light uppercase tracking-[0.2em] text-graphite/60">{vid.error || 'Идёт запись видео — не сворачивайте вкладку'}</p>}
        {phase === 'drawing' && (
          <div className="mx-auto mt-6 h-px w-40 bg-graphite/10">
            <div ref={bar} className="h-full origin-left bg-rose/70" style={{ transform: 'scaleX(0)' }} />
          </div>
        )}
      </div>

      <nav className={`fixed bottom-[max(1rem,env(safe-area-inset-bottom))] flex max-w-full flex-wrap justify-center gap-1 px-2 transition-opacity duration-700 ${phase === 'intro' ? 'opacity-0' : 'opacity-100'}`}>
        {phase === 'drawing' && <button className="ctl" onClick={toggle}>{state === 'playing' ? 'Пауза' : 'Продолжить'}</button>}
        {phase === 'drawing' && <button className="ctl" onClick={cycleSpeed}>{speed}×</button>}
        {phase === 'final' && <button className="ctl ctl-primary" disabled={!dl.ready} onClick={dl.save}>{dl.ready ? 'Скачать портрет' : 'Готовим…'}</button>}
        {phase === 'final' && vid.supported && <button className="ctl ctl-primary" disabled={vid.state === 'recording'} onClick={vid.start}>{vid.state === 'recording' ? `Запись видео… ${Math.round(vid.progress * 100)}%` : vid.state === 'ready' && vid.ios ? 'Сохранить видео' : 'Скачать видео'}</button>}
        <button className="ctl" onClick={replay}>Заново</button>
        {phase !== 'intro' && <a className="ctl" href="/studio">Нарисовать своё ♡</a>}
      </nav>

      {phase === 'final' && Array.from({ length: 9 }, (_, i) => <span key={i} className="petal" />)}
    </div>
  )
}

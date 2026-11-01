import { useCallback, useEffect, useRef, useState } from 'react'
import { pickVideoMime, type StrokeEngine } from './engine/engine'

const NAME = 'love-in-every-line.png'
export const isIOS = () => /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)

/** iPhone/iPad: системное меню «Сохранить»; остальные: обычное скачивание. */
export async function saveFile(file: File) {
  if (isIOS() && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'Love in Every Line' })
      return
    } catch (e) {
      if ((e as DOMException).name === 'AbortError') return // пользователь закрыл меню
    }
  }
  const url = URL.createObjectURL(file)
  const a = document.createElement('a')
  a.href = url
  a.download = file.name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

/**
 * Готовит PNG заранее (когда рисование закончено), чтобы нажатие «Скачать» срабатывало мгновенно:
 * iOS требует вызвать share/скачивание сразу в обработчике нажатия.
 */
export function usePortraitDownload(getEngine: () => StrokeEngine | null, active: boolean) {
  const [file, setFile] = useState<File | null>(null)

  useEffect(() => {
    if (!active) {
      setFile(null)
      return
    }
    let dead = false
    const t = setTimeout(async () => {
      try {
        const blob = await getEngine()?.exportPng()
        if (blob && !dead) setFile(new File([blob], NAME, { type: 'image/png' }))
      } catch {
        /* кнопка просто останется неактивной */
      }
    }, 300)
    return () => {
      dead = true
      clearTimeout(t)
    }
  }, [active, getEngine])

  const save = useCallback(async () => {
    if (file) await saveFile(file)
  }, [file])

  return { ready: file !== null, save }
}

/**
 * Видео процесса рисования: пишется только по нажатию (в реальном времени, ~длительность анимации).
 * Не-iOS: файл скачивается сразу по окончании. iOS: после записи нужно нажать «Сохранить видео»
 * (меню «Поделиться» требует свежего нажатия пальцем).
 */
export function useVideoDownload(getEngine: () => StrokeEngine | null, active: boolean) {
  const supported = typeof window !== 'undefined' && pickVideoMime() !== null
  const [state, setState] = useState<'idle' | 'recording' | 'ready'>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const fileRef = useRef<File | null>(null)
  const ctl = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!active) {
      ctl.current?.abort()
      fileRef.current = null
      setState('idle')
      setProgress(0)
      setError('')
    }
  }, [active])
  useEffect(() => () => ctl.current?.abort(), [])

  const start = useCallback(async () => {
    setError('')
    if (fileRef.current) {
      await saveFile(fileRef.current)
      return
    }
    const eng = getEngine()
    if (!eng) return
    const c = new AbortController()
    ctl.current = c
    setState('recording')
    setProgress(0)
    try {
      const { blob, ext } = await eng.recordVideo({ onProgress: setProgress, signal: c.signal })
      const file = new File([blob], `love-in-every-line.${ext}`, { type: blob.type })
      fileRef.current = file
      setState('ready')
      if (!isIOS()) await saveFile(file)
    } catch (e) {
      setState('idle')
      if ((e as DOMException).name !== 'AbortError') setError('Не удалось записать видео')
    }
  }, [getEngine])

  return { supported, state, progress, error, start, ios: isIOS() }
}

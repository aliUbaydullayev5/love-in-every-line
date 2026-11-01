import { useCallback, useEffect, useState } from 'react'
import type { StrokeEngine } from './engine/engine'

const NAME = 'love-in-every-line.png'
const isIOS = () => /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)

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
    if (!file) return
    // iPhone/iPad: системное меню «Сохранить изображение» (в Фото/Файлы)
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
    a.download = NAME
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 10_000)
  }, [file])

  return { ready: file !== null, save }
}

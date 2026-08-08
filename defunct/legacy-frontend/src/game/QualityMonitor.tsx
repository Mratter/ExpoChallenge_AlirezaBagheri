import { useFrame } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import { medianFrameRate } from './renderQuality'

const QUALITY_WINDOW_FRAMES = 90

/** Samples one fixed-size frame window and reports only its median FPS. */
export function QualityMonitor({ onWindow }: { onWindow: (medianFps: number) => void }) {
  const samples = useRef<number[]>(Array(QUALITY_WINDOW_FRAMES).fill(0))
  const sampleCount = useRef(0)
  const onWindowRef = useRef(onWindow)

  useEffect(() => {
    onWindowRef.current = onWindow
  }, [onWindow])

  useFrame((_, delta) => {
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
      sampleCount.current = 0
      return
    }
    samples.current[sampleCount.current] = delta
    sampleCount.current += 1
    if (sampleCount.current < QUALITY_WINDOW_FRAMES) return
    const median = medianFrameRate(samples.current)
    sampleCount.current = 0
    if (median !== null) onWindowRef.current(median)
  })

  return null
}

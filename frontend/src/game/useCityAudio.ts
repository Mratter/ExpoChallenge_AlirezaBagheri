import { useCallback, useEffect, useRef, useState } from 'react'
import { ProceduralCityAudio, type CityAudioSnapshot } from './audio'

export function useCityAudio(snapshot: CityAudioSnapshot | null) {
  const engine = useRef<ProceduralCityAudio | null>(null)
  const [muted, setMuted] = useState(false)
  const getEngine = useCallback(() => {
    if (!engine.current) engine.current = new ProceduralCityAudio()
    return engine.current
  }, [])

  useEffect(() => {
    getEngine().update(snapshot)
  }, [getEngine, snapshot])

  useEffect(() => {
    const unlock = () => {
      window.removeEventListener('pointerdown', unlock, true)
      window.removeEventListener('keydown', unlock, true)
      getEngine().unlock()
    }
    window.addEventListener('pointerdown', unlock, { capture: true, once: true })
    window.addEventListener('keydown', unlock, { capture: true, once: true })
    return () => {
      window.removeEventListener('pointerdown', unlock, true)
      window.removeEventListener('keydown', unlock, true)
    }
  }, [getEngine])

  useEffect(() => {
    getEngine().setMuted(muted)
  }, [getEngine, muted])

  useEffect(() => () => {
    engine.current?.dispose()
    engine.current = null
  }, [])

  const toggleMuted = useCallback(() => setMuted((current) => !current), [])
  return { muted, toggleMuted }
}

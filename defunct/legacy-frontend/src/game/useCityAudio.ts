import { useCallback, useEffect, useRef, useState } from 'react'
import { ProceduralCityAudio, type CityAudioSnapshot } from './audio'

export const CITY_AUDIO_MUTED_STORAGE_KEY = 'relay-city:audio-muted'

export function readStoredCityAudioMuted(storage?: Pick<Storage, 'getItem'>): boolean {
  if (!storage) {
    if (typeof window === 'undefined') return false
    try {
      storage = window.localStorage
    } catch {
      return false
    }
  }
  try {
    return storage.getItem(CITY_AUDIO_MUTED_STORAGE_KEY) === 'true'
  } catch {
    return false
  }
}

export function persistCityAudioMuted(
  muted: boolean,
  storage?: Pick<Storage, 'setItem'>,
): void {
  if (!storage) {
    if (typeof window === 'undefined') return
    try {
      storage = window.localStorage
    } catch {
      return
    }
  }
  try {
    storage.setItem(CITY_AUDIO_MUTED_STORAGE_KEY, String(muted))
  } catch {
    // Storage may be unavailable in a locked-down browser; mute still works in memory.
  }
}

export function useCityAudio(snapshot: CityAudioSnapshot | null) {
  const engine = useRef<ProceduralCityAudio | null>(null)
  const [muted, setMuted] = useState(readStoredCityAudioMuted)
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
    persistCityAudioMuted(muted)
  }, [getEngine, muted])

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => getEngine().setReducedSensory(query.matches)
    update()
    query.addEventListener?.('change', update)
    return () => query.removeEventListener?.('change', update)
  }, [getEngine])

  useEffect(() => {
    const syncStoredPreference = (event: StorageEvent) => {
      if (event.key !== CITY_AUDIO_MUTED_STORAGE_KEY || event.newValue === null) return
      setMuted(event.newValue === 'true')
    }
    window.addEventListener('storage', syncStoredPreference)
    return () => window.removeEventListener('storage', syncStoredPreference)
  }, [])

  useEffect(() => () => {
    engine.current?.dispose()
    engine.current = null
  }, [])

  const toggleMuted = useCallback(() => setMuted((current) => !current), [])
  return { muted, toggleMuted }
}

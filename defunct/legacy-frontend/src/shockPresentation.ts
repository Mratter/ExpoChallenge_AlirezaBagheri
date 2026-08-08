import type { ShockType } from './types'

/**
 * Presentation names are intentionally separate from the frozen engine keys. In
 * particular, the authored `aftershock` key is shown to players as Earthquake while
 * requests, persisted scenarios, and deterministic result identities keep the raw key.
 */
export const SHOCK_DISPLAY_NAMES: Readonly<Record<ShockType, string>> = {
  aftershock: 'Earthquake',
  supply: 'Supply',
  epidemic: 'Epidemic',
  utility: 'Utility',
  weather: 'Weather',
}

export function shockDisplayName(type: string | null | undefined): string {
  if (!type) return 'No shock'
  if (type in SHOCK_DISPLAY_NAMES) return SHOCK_DISPLAY_NAMES[type as ShockType]
  return type.charAt(0).toUpperCase() + type.slice(1).replaceAll('_', ' ')
}

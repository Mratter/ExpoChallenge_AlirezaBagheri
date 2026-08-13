export type AppRoute = 'landing' | 'toolbox' | 'game'

export function routeFromHashValue(hash: string): AppRoute {
  const normalized = hash.trim().toLowerCase()
  if (normalized === '#/toolbox') return 'toolbox'
  if (normalized === '#/game') return 'game'
  return 'landing'
}

export function hashForRoute(route: AppRoute): string {
  return route === 'landing' ? '#/' : `#/${route}`
}

export function titleForRoute(route: AppRoute): string {
  if (route === 'toolbox') return 'RELAY | Analyst Toolbox'
  if (route === 'game') return 'RELAY | Interactive 3D city'
  return 'RELAY | Evidence-led municipal recovery planning'
}

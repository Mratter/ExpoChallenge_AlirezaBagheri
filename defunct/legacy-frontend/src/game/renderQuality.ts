export const RENDER_QUALITY_TIERS = ['reduced', 'balanced', 'high'] as const
export type RenderQualityTier = (typeof RENDER_QUALITY_TIERS)[number]
export type RenderDetailTier = 'low' | 'medium' | 'full'

export type ViewportMetrics = {
  width: number
  height: number
  devicePixelRatio: number
}

export type ViewportQualityPolicy = {
  initialTier: RenderQualityTier
  ceiling: RenderQualityTier
}

export type AdaptiveQualityState = {
  tier: RenderQualityTier
  ceiling: RenderQualityTier
  pressureWindows: number
  recoveryWindows: number
}

export type RenderQualityProfile = {
  tier: RenderQualityTier
  dpr: number
  shadowsEnabled: boolean
  shadowMapSize: 512 | 1024 | 2048
  smallObjectShadows: boolean
  detailTier: RenderDetailTier
  treeStride: 1 | 2 | 3
  vehicleParts: 3 | 4 | 5
  essentialDepots: true
  essentialIncidents: true
}

const QUALITY_RANK: Readonly<Record<RenderQualityTier, number>> = {
  reduced: 0,
  balanced: 1,
  high: 2,
}

function tierAtRank(rank: number): RenderQualityTier {
  return RENDER_QUALITY_TIERS[Math.max(0, Math.min(RENDER_QUALITY_TIERS.length - 1, rank))]
}

function clampTier(tier: RenderQualityTier, ceiling: RenderQualityTier): RenderQualityTier {
  return tierAtRank(Math.min(QUALITY_RANK[tier], QUALITY_RANK[ceiling]))
}

/**
 * Portrait/mobile viewports begin in a restrained balanced tier and cannot silently
 * promote to the desktop-only high tier. The selector may still degrade them when
 * measured frame pressure persists.
 */
export function viewportQualityPolicy({
  width,
  height,
  devicePixelRatio,
}: ViewportMetrics): ViewportQualityPolicy {
  const safeWidth = Math.max(1, width)
  const safeHeight = Math.max(1, height)
  const portrait = safeHeight > safeWidth * 1.18
  const compact = safeWidth <= 720 || safeWidth * safeHeight < 650_000
  const densePixels = devicePixelRatio > 2.25 && safeWidth <= 1024
  if (portrait || compact || densePixels) {
    return { initialTier: 'balanced', ceiling: 'balanced' }
  }
  return { initialTier: 'high', ceiling: 'high' }
}

export function createAdaptiveQualityState(
  policy: ViewportQualityPolicy,
): AdaptiveQualityState {
  return {
    tier: clampTier(policy.initialTier, policy.ceiling),
    ceiling: policy.ceiling,
    pressureWindows: 0,
    recoveryWindows: 0,
  }
}

/** Preserves the current tier across resize while enforcing the new device ceiling. */
export function applyQualityCeiling(
  state: AdaptiveQualityState,
  ceiling: RenderQualityTier,
): AdaptiveQualityState {
  return {
    tier: clampTier(state.tier, ceiling),
    ceiling,
    pressureWindows: 0,
    recoveryWindows: 0,
  }
}

/**
 * Deterministic hysteresis over completed FPS windows. Two sustained low windows step
 * down one tier; four sustained healthy windows step up. The separated thresholds
 * prevent quality oscillation near 60fps.
 */
export function advanceAdaptiveQuality(
  state: AdaptiveQualityState,
  medianFps: number,
): AdaptiveQualityState {
  if (!Number.isFinite(medianFps) || medianFps <= 0) return state
  const pressureThreshold = state.tier === 'high' ? 54 : 44
  const recoveryThreshold = state.tier === 'reduced' ? 52 : 59

  if (medianFps < pressureThreshold && state.tier !== 'reduced') {
    const pressureWindows = state.pressureWindows + 1
    if (pressureWindows >= 2) {
      return {
        ...state,
        tier: tierAtRank(QUALITY_RANK[state.tier] - 1),
        pressureWindows: 0,
        recoveryWindows: 0,
      }
    }
    return { ...state, pressureWindows, recoveryWindows: 0 }
  }

  if (
    medianFps > recoveryThreshold
    && QUALITY_RANK[state.tier] < QUALITY_RANK[state.ceiling]
  ) {
    const recoveryWindows = state.recoveryWindows + 1
    if (recoveryWindows >= 4) {
      return {
        ...state,
        tier: clampTier(tierAtRank(QUALITY_RANK[state.tier] + 1), state.ceiling),
        pressureWindows: 0,
        recoveryWindows: 0,
      }
    }
    return { ...state, recoveryWindows, pressureWindows: 0 }
  }

  if (state.pressureWindows === 0 && state.recoveryWindows === 0) return state
  return { ...state, pressureWindows: 0, recoveryWindows: 0 }
}

/** Median FPS retains genuine frame pressure down to 4fps. Hidden tabs are reset by the monitor. */
export function medianFrameRate(frameDurations: readonly number[]): number | null {
  const valid = frameDurations
    .filter((duration) => Number.isFinite(duration) && duration >= 1 / 240 && duration <= 0.25)
    .map((duration) => 1 / duration)
    .sort((left, right) => left - right)
  if (valid.length === 0) return null
  const middle = Math.floor(valid.length / 2)
  return valid.length % 2 === 0
    ? (valid[middle - 1] + valid[middle]) / 2
    : valid[middle]
}

export function renderQualityProfile(
  tier: RenderQualityTier,
  viewport: ViewportMetrics,
): RenderQualityProfile {
  const ratio = Math.max(0.75, viewport.devicePixelRatio || 1)
  const portrait = viewport.height > viewport.width
  if (tier === 'high') {
    return {
      tier,
      dpr: Math.min(ratio, portrait ? 1.25 : 1.5),
      shadowsEnabled: true,
      shadowMapSize: 2048,
      smallObjectShadows: true,
      detailTier: 'full',
      treeStride: 1,
      vehicleParts: 5,
      essentialDepots: true,
      essentialIncidents: true,
    }
  }
  if (tier === 'balanced') {
    return {
      tier,
      dpr: Math.min(ratio, portrait ? 1.1 : 1.25),
      shadowsEnabled: true,
      shadowMapSize: 1024,
      smallObjectShadows: false,
      detailTier: 'medium',
      treeStride: 2,
      vehicleParts: 4,
      essentialDepots: true,
      essentialIncidents: true,
    }
  }
  return {
    tier,
    dpr: Math.max(0.85, Math.min(ratio, 1)),
    shadowsEnabled: false,
    shadowMapSize: 512,
    smallObjectShadows: false,
    detailTier: 'low',
    treeStride: 3,
    vehicleParts: 3,
    essentialDepots: true,
    essentialIncidents: true,
  }
}

/** Rich landmarks step from three to two to one per district; dense infill remains. */
export function usesDetailedBuilding(
  buildingIndex: number,
  detailTier: RenderDetailTier,
): boolean {
  if (buildingIndex === 0) return true
  if (buildingIndex === 17) return detailTier !== 'low'
  if (buildingIndex === 35) return detailTier === 'full'
  return false
}

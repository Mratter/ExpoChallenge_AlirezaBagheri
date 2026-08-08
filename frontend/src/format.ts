import type { EvidenceMetric, ModelTrack } from './types'

export function formatInteger(value: number): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)
}

export function compactHash(value: string): string {
  return value.length > 22 ? `${value.slice(0, 12)}…${value.slice(-8)}` : value
}

export function humanizeToken(value: string): string {
  return value.replaceAll('_', ' ')
}

export function metricById(track: ModelTrack, id: string): EvidenceMetric | undefined {
  return track.evaluation.metrics.find((metric) => metric.id === id)
}

export function isUntrained(track: ModelTrack): boolean {
  return !track.training.started || track.training.transitions === 0
}

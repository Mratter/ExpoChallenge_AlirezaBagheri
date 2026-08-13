export type PointerDayScale = {
  boundsLeft: number
  boundsWidth: number
  clientX: number
  dayEnd: number
  dayStart: number
  plotLeft: number
  plotWidth: number
  viewWidth: number
}

export function dayFromChartPointer({
  boundsLeft,
  boundsWidth,
  clientX,
  dayEnd,
  dayStart,
  plotLeft,
  plotWidth,
  viewWidth,
}: PointerDayScale): number {
  if (!Number.isFinite(boundsWidth) || boundsWidth <= 0 || plotWidth <= 0) return dayStart
  const viewX = ((clientX - boundsLeft) / boundsWidth) * viewWidth
  const plotX = Math.max(0, Math.min(plotWidth, viewX - plotLeft))
  return Math.round(dayStart + (plotX / plotWidth) * (dayEnd - dayStart))
}

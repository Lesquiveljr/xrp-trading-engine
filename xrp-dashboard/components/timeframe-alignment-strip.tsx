type TimeframeSignal = {
  timeframe: string
  signal: string
  score?: number
}

type TimeframeAlignmentStripProps = {
  items: TimeframeSignal[]
}

function getSignalStyles(signal: string, score?: number) {
  if (score !== undefined && score !== null) {
    if (score >= 2) {
      return "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
    }

    if (score >= 1) {
      return "bg-green-500/15 text-green-300 border-green-500/30"
    }

    if (score === 0) {
      return "bg-slate-500/15 text-slate-300 border-slate-500/30"
    }

    if (score <= -2) {
      return "bg-rose-500/15 text-rose-300 border-rose-500/30"
    }

    if (score < 0) {
      return "bg-orange-500/15 text-orange-300 border-orange-500/30"
    }
  }

  const normalized = signal.toLowerCase()

  if (normalized.includes("bullish")) {
    return "bg-green-500/15 text-green-300 border-green-500/30"
  }

  if (normalized.includes("bearish")) {
    return "bg-red-500/15 text-red-300 border-red-500/30"
  }

  if (normalized.includes("mixed")) {
    return "bg-yellow-500/15 text-yellow-300 border-yellow-500/30"
  }

  return "bg-slate-500/15 text-slate-300 border-slate-500/30"
}

export default function TimeframeAlignmentStrip({
  items,
}: TimeframeAlignmentStripProps) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-4">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-white">Trend Alignment</h3>
        <p className="text-xs text-slate-400">
          Quick view of signal direction across timeframes
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {items.map((item) => (
          <div
            key={item.timeframe}
            className={`rounded-xl border px-3 py-2 text-sm ${getSignalStyles(
              item.signal,
              item.score
            )}`}
          >
            <div className="font-semibold">{item.timeframe}</div>
            <div className="text-xs opacity-90">{item.signal}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
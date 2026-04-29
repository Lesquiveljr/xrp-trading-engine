"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  BarChart3,
  Clock3,
  Layers3,
  RefreshCw,
  Zap,
} from "lucide-react";

const API_BASE = "";
const DEFAULT_MODE = "balanced";
const REFRESH_MS = 15000;
const VIEW_OPTIONS = ["live", "review", "deep"] as const;
const TIMEFRAME_ORDER = ["1m", "5m", "15m", "1h", "4h", "1d"] as const;

const PANEL =
  "rounded-3xl border border-slate-800/90 bg-slate-900/85 shadow-[0_0_0_1px_rgba(15,23,42,0.35),0_18px_42px_rgba(2,6,23,0.5)] backdrop-blur";
const SUBPANEL =
  "rounded-2xl border border-slate-800/90 bg-slate-900/75 shadow-[0_8px_24px_rgba(2,6,23,0.28)]";
const METRIC_PANEL =
  "rounded-2xl border border-slate-800/90 bg-slate-950/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]";
const PILL_BASE =
  "border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] shadow-sm";

const badgeMap = {
  strongBull: "border-emerald-400/35 bg-emerald-500/15 text-emerald-300",
  bull: "border-green-400/35 bg-green-500/15 text-green-300",
  strongBear: "border-rose-400/35 bg-rose-500/15 text-rose-300",
  bear: "border-red-400/35 bg-red-500/15 text-red-300",
  neutral: "border-slate-600/70 bg-slate-800/70 text-slate-300",
  watch: "border-sky-400/35 bg-sky-500/15 text-sky-300",
  pullback: "border-amber-400/35 bg-amber-500/15 text-amber-300",
  warning: "border-orange-400/40 bg-orange-500/15 text-orange-300",
};

type ViewMode = (typeof VIEW_OPTIONS)[number];

type PriceZone = {
  low?: number;
  high?: number;
  mid?: number;
  label?: string;
  description?: string;
  reference_timeframes?: string[];
};

type TakeProfitZone = {
  tp1?: number;
  tp2?: number;
  tp3?: number;
  description?: string;
  reference_timeframes?: string[];
};

type TriggerContext = {
  setup_timeframe?: string;
  trigger_timeframe?: string;
  setup_signal?: string;
  setup?: string | null;
  trigger_signal?: string;
  trigger_structure?: string | null;
  state?: string;
  confirmation_level?: number;
  ladder_stage?: string;
  activation_ready?: boolean;
  setup_support_directional?: boolean;
  setup_support_continuation?: boolean;
  trigger_indicator_confirmed?: boolean;
  trigger_band_break_required?: boolean;
  trigger_band_break_confirmed?: boolean | null;
  trigger_band_break_method?: string;
  blocking_flags?: string[];
  activation_blockers?: string[];
  primary_blocker?: string | null;
  reasons?: string[];
};

type ChecklistItem = {
  condition?: string;
  passed?: boolean;
};

type EntryValidation = {
  allowed?: boolean;
  status?: string;
  status_detail?: string;
  checklist?: ChecklistItem[];
  blocking_reasons?: string[];
};

type ConfidenceComponents = {
  trend_quality?: number;
  entry_quality?: number;
  risk_quality?: number;
  notes?: {
    trend?: string[];
    entry?: string[];
    risk?: string[];
  };
};

type EntryScore = {
  score?: number;
  tier?: string;
  notes?: string[];
};

type ExecutionLensLane = {
  state?: string;
  label?: string;
  score?: number;
  window?: string;
  instruction?: string;
  ready?: boolean;
  blockers?: string[];
};

type ExecutionLens = {
  primary_lane?: string;
  headline?: string;
  summary?: string;
  direction?: string;
  scalp?: ExecutionLensLane;
  directional?: ExecutionLensLane & {
    higher_timeframe_aligned?: boolean;
    overall_aligned?: boolean;
  };
  notes?: string[];
};

type ExecutionReview = {
  market_entry_price?: number | null;
  planned_entry_price?: number | null;
  ideal_entry_price?: number | null;
  entry_zone_low?: number | null;
  entry_zone_high?: number | null;
  entry_location?: string | null;
  zone_width_pct?: number | null;
  market_vs_planned_pct?: number | null;
  market_vs_ideal_pct?: number | null;
  reward_to_risk_estimate?: number | null;
  execution_efficiency_score?: number | null;
  execution_efficiency_label?: string | null;
  quality?: string | null;
  planned_entry_touched_after_signal?: boolean | null;
  ideal_entry_touched_after_signal?: boolean | null;
  entry_zone_retested_after_signal?: boolean | null;
  market_entry_latest_pnl_pct?: number | null;
  planned_entry_latest_pnl_pct?: number | null;
  ideal_entry_latest_pnl_pct?: number | null;
  planned_edge_vs_market_latest_pct?: number | null;
  ideal_edge_vs_market_latest_pct?: number | null;
  notes?: string[];
};

type LimitOrderPlan = {
  enabled?: boolean;
  planner_state?: string;
  preference?: string;
  side?: string | null;
  order_type?: string;
  limit_price?: number | null;
  backup_limit_price?: number | null;
  entry_zone?: PriceZone | null;
  invalidation_price?: number | null;
  take_profit_reference?: number | null;
  market_entry_allowed?: boolean;
  expiry_snapshots?: number;
  expiry_minutes_estimate?: number;
  cancel_if?: string[];
  promotion_rules?: string[];
  reason?: string;
  notes?: string[];
  historical_hint?: string | null;
  source?: string;
};


type ScaleOutStep = {
  target?: string;
  size_pct?: number;
  price?: number | null;
  purpose?: string;
};

type ExecutionManagement = {
  stop_policy?: string;
  stop_distance_pct?: number | null;
  break_even_after?: string;
  trailing_stop_after?: string;
  trailing_stop_reference?: string;
  runner_policy?: string;
  scale_out_plan?: ScaleOutStep[];
  management_notes?: string[];
};

type ExecutionPlan = {
  execution_ready?: boolean;
  direction?: string;
  entry_model?: string;
  timing_state?: string;
  primary_timeframes?: {
    setup?: string;
    trigger?: string;
    risk?: string;
  };
  entry_zone?: PriceZone | null;
  invalidation_zone?: PriceZone | null;
  take_profit_zone?: TakeProfitZone | null;
  reward_to_risk_estimate?: number | null;
  notes?: string[];
  entry_style?: string;
  position_sizing?: string;
  aggressiveness?: string;
  execution_instruction?: string;
  playbook?: string[];
  execution_management?: ExecutionManagement;
  limit_order_plan?: LimitOrderPlan;
  entry_score?: EntryScore;
  execution_review?: ExecutionReview;
  execution_lens?: ExecutionLens;
};

type Strategy = {
  strategy_bias?: string;
  market_phase?: string;
  state?: string;
  action?: string;
  entry_timing?: string;
  trade_quality?: string;
  setup_timeframes?: string[];
  trigger_timeframes?: string[];
  risk_timeframes?: string[];
  risk_state?: string;
  summary?: string;
  confidence?: number;
  confidence_components?: ConfidenceComponents;
  entry_permission?: boolean;
  entry_validation?: EntryValidation;
  trigger_context?: TriggerContext;
  entry_score?: EntryScore;
  execution_lens?: ExecutionLens;
  execution_review?: ExecutionReview;
  limit_order_plan?: LimitOrderPlan;
  execution_plan?: ExecutionPlan;
};

type SignalDetails = {
  trend_bias?: string;
  momentum_state?: string;
  rsi_state?: string;
  bollinger_state?: string;
  position_vs_ema9?: string;
  position_vs_sma20?: string;
};

type EngineTimeframeItem = {
  signal?: string;
  signal_score?: number;
  price?: number;
  setup?: string | null;
  setup_confidence?: number;
  structure?: string | null;
  trap_risk?: string | null;
  summary?: string;
  history?: number[];
  indicators?: {
    rsi_14?: number;
    atr_14?: number;
  };
  signal_details?: SignalDetails;
  fibonacci?: {
    levels?: Record<string, number>;
  };
  candle?: {
    time?: string;
  };
};

type ModeMeta = {
  active?: string;
  label?: string;
  available?: string[];
};

type EngineResponse = {
  symbol?: string;
  status?: string;
  engine_version?: string;
  generated_at?: string;
  mode?: ModeMeta;
  consensus?: {
    mode?: ModeMeta;
    short_term?: { bias?: string; average_score?: number };
    higher_timeframes?: { bias?: string; average_score?: number };
    overall?: { bias?: string; average_score?: number };
    dashboard_summary?: string;
    strategy?: Strategy;
  };
  timeframes?: Record<string, EngineTimeframeItem>;
};

type TradeTrackerSummary = {
  total_trades?: number;
  open_trades?: number;
  closed_trades?: number;
  wins?: number;
  losses?: number;
  flats?: number;
  mixed?: number;
  win_rate?: number | null;
  avg_snapshots_held?: number | null;
  avg_market_entry_resolved_pct?: number | null;
  avg_planned_edge_vs_market_resolved_pct?: number | null;
  avg_execution_efficiency_score?: number | null;
};

type TradeRecord = {
  trade_id?: string;
  direction?: string;
  status?: string;
  outcome_status?: string;
  outcome_label?: string;
  opened_at?: string;
  closed_at?: string | null;
  opening_action?: string;
  opening_market_phase?: string;
  confidence_best?: number;
  entry_score_best?: number;
  entry_model_family?: string | null;
  reward_to_risk_estimate?: number | null;
  entry_market_price?: number | null;
  planned_entry_price?: number | null;
  execution_efficiency_score?: number | null;
  execution_efficiency_label?: string | null;
  entry_location?: string | null;
  tp1_hit?: boolean;
  invalidation_hit?: boolean;
  snapshots_held?: number | null;
  market_entry_resolved_pct?: number | null;
  planned_edge_vs_market_resolved_pct?: number | null;
  notes?: string[];
};

type TradeTrackerSummaryResponse = {
  summary?: TradeTrackerSummary;
};

type TradeTrackerResponse = {
  summary?: TradeTrackerSummary;
  trades?: TradeRecord[];
};

function prettyLabel(value: unknown) {
  if (value === null || value === undefined || value === "") return "—";
  return String(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatNumber(value: unknown, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  });
}

function formatPercent(value: unknown, digits = 2, signed = true) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const num = Number(value);
  const prefix = signed && num > 0 ? "+" : "";
  return `${prefix}${num.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  })}%`;
}

function formatTime(value: unknown) {
  if (!value) return "—";
  const d = new Date(String(value));
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString();
}

function boolLabel(value: boolean | null | undefined) {
  if (value === null || value === undefined) return "Pending";
  return value ? "Yes" : "No";
}

function actionTone(action?: string | null) {
  const value = (action || "").toLowerCase();
  if (value === "enter_long") return badgeMap.strongBull;
  if (value === "enter_short") return badgeMap.strongBear;
  if (value.includes("watch")) return badgeMap.watch;
  if (value.includes("pullback")) return badgeMap.pullback;
  if (value.includes("wait")) return badgeMap.neutral;
  if (value.includes("avoid") || value.includes("trap")) return badgeMap.warning;
  return badgeMap.neutral;
}

function biasTone(bias?: string | null) {
  switch ((bias || "").toLowerCase()) {
    case "strong_bullish":
      return badgeMap.strongBull;
    case "bullish":
      return badgeMap.bull;
    case "strong_bearish":
      return badgeMap.strongBear;
    case "bearish":
      return badgeMap.bear;
    default:
      return badgeMap.neutral;
  }
}

function qualityTone(quality?: string | null) {
  switch ((quality || "").toUpperCase()) {
    case "A":
      return badgeMap.strongBull;
    case "B":
      return badgeMap.bull;
    case "C":
      return badgeMap.pullback;
    case "D":
      return badgeMap.strongBear;
    default:
      return badgeMap.neutral;
  }
}

function riskTone(riskState?: string | null) {
  switch ((riskState || "").toLowerCase()) {
    case "favorable":
      return badgeMap.strongBull;
    case "high":
      return badgeMap.strongBear;
    case "mixed":
      return badgeMap.pullback;
    default:
      return badgeMap.neutral;
  }
}

function efficiencyTone(label?: string | null) {
  switch ((label || "").toLowerCase()) {
    case "excellent":
      return badgeMap.strongBull;
    case "good":
      return badgeMap.bull;
    case "acceptable":
      return badgeMap.watch;
    case "stretched":
      return badgeMap.pullback;
    case "poor":
      return badgeMap.strongBear;
    default:
      return badgeMap.neutral;
  }
}

function entryLocationTone(location?: string | null) {
  switch ((location || "").toLowerCase()) {
    case "inside_zone":
      return badgeMap.strongBull;
    case "above_zone":
    case "below_zone":
      return badgeMap.pullback;
    default:
      return badgeMap.neutral;
  }
}

function plannerTone(plan?: LimitOrderPlan) {
  const preference = (plan?.preference || "").toLowerCase();
  const state = (plan?.planner_state || "").toLowerCase();

  if (!plan?.enabled) {
    if (state.includes("monitor")) return badgeMap.watch;
    return badgeMap.neutral;
  }
  if (preference === "limit_only") return badgeMap.strongBull;
  if (preference === "limit_preferred") return badgeMap.bull;
  if (preference === "monitor_then_limit") return badgeMap.watch;
  if (preference === "market_or_limit") return badgeMap.pullback;
  return badgeMap.neutral;
}

function limitSideTone(side?: string | null) {
  if ((side || "").toLowerCase() === "buy") return badgeMap.strongBull;
  if ((side || "").toLowerCase() === "sell") return badgeMap.strongBear;
  return badgeMap.neutral;
}

function pageBackgroundForState(bias?: string | null, plan?: LimitOrderPlan) {
  const normalizedBias = (bias || "").toLowerCase();
  const preference = (plan?.preference || "").toLowerCase();
  const plannerGlow = preference === "limit_only" ? "rgba(168,85,247,0.10)" : "rgba(34,211,238,0.07)";

  if (normalizedBias.includes("bull")) {
    return `radial-gradient(circle at top left, rgba(16,185,129,0.16), transparent 24%), radial-gradient(circle at top right, ${plannerGlow}, transparent 22%), linear-gradient(180deg,#020617 0%,#020617 100%)`;
  }
  if (normalizedBias.includes("bear")) {
    return `radial-gradient(circle at top left, rgba(244,63,94,0.13), transparent 24%), radial-gradient(circle at top right, ${plannerGlow}, transparent 22%), linear-gradient(180deg,#020617 0%,#020617 100%)`;
  }
  return `radial-gradient(circle at top, rgba(34,211,238,0.10), transparent 22%), radial-gradient(circle at top right, ${plannerGlow}, transparent 18%), linear-gradient(180deg,#020617 0%,#020617 100%)`;
}

function visualAmbientTone(strategy?: Strategy, plan?: LimitOrderPlan) {
  const direction = (strategy?.strategy_bias || "").toLowerCase();
  const preference = (plan?.preference || "").toLowerCase();

  if (preference === "limit_only") return "from-violet-500/20 via-cyan-500/5 to-transparent";
  if (direction === "long") return "from-emerald-500/20 via-cyan-500/5 to-transparent";
  if (direction === "short") return "from-rose-500/20 via-orange-500/5 to-transparent";
  return "from-cyan-500/10 via-slate-500/5 to-transparent";
}

function asNumber(value: unknown): number | undefined {
  if (value === null || value === undefined || value === "") return undefined;
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function directionalDistancePercent(direction: string | undefined, fromPrice?: number, toPrice?: number) {
  if (!fromPrice || !toPrice) return undefined;
  const raw = ((toPrice - fromPrice) / fromPrice) * 100;
  return direction === "short" ? -raw : raw;
}

function tradeOutcomeTone(label?: string | null) {
  switch ((label || "").toLowerCase()) {
    case "tp1_win":
    case "win":
      return badgeMap.strongBull;
    case "stopped_out":
    case "loss":
      return badgeMap.strongBear;
    case "signal_flip_exit":
    case "mixed":
      return badgeMap.pullback;
    case "timed_out":
    case "flat":
      return badgeMap.neutral;
    case "still_open":
      return badgeMap.watch;
    default:
      return badgeMap.neutral;
  }
}

function directionTone(direction?: string | null) {
  if ((direction || "").toLowerCase() === "long") return badgeMap.strongBull;
  if ((direction || "").toLowerCase() === "short") return badgeMap.strongBear;
  return badgeMap.neutral;
}

function laneTone(lane?: string | null) {
  switch ((lane || "").toLowerCase()) {
    case "scalp":
      return badgeMap.strongBull;
    case "directional_watch":
      return badgeMap.watch;
    case "stand_aside":
      return badgeMap.neutral;
    default:
      return badgeMap.pullback;
  }
}

function lensLaneTone(lane?: ExecutionLensLane) {
  switch ((lane?.state || "").toLowerCase()) {
    case "ready":
    case "limit_wait":
    case "active_bias":
      return badgeMap.strongBull;
    case "watch":
    case "wait_confirmation":
      return badgeMap.watch;
    case "blocked":
      return badgeMap.strongBear;
    default:
      return badgeMap.neutral;
  }
}

function compactDirection(strategy?: Strategy) {
  if ((strategy?.strategy_bias || "").toLowerCase() === "long") return "Long";
  if ((strategy?.strategy_bias || "").toLowerCase() === "short") return "Short";
  return "Neutral";
}

function compactTiming(strategy?: Strategy, execution?: ExecutionPlan) {
  if (execution?.execution_ready || strategy?.entry_permission) return "Enter Now";
  const action = (strategy?.action || "").toLowerCase();
  if (action.includes("wait_pullback")) return "Wait";
  if (action.includes("watch")) return "Wait";
  if (action === "wait") return "Stand Aside";
  return "Wait";
}

function compactEntryQuality(review?: ExecutionReview, strategy?: Strategy) {
  const quality = (review?.quality || "").toLowerCase();
  const location = (review?.entry_location || "").toLowerCase();
  if (quality === "chasing" || location === "above_zone" || location === "below_zone") return "Chasing";
  if (quality === "ideal") return "Perfect";
  if (quality === "in_band" || location === "inside_zone") return "Good Entry";
  if ((strategy?.action || "").toLowerCase() === "wait") return "No Trade";
  return "Needs Confirmation";
}

function buildLiveSentence(strategy?: Strategy, execution?: ExecutionPlan, review?: ExecutionReview) {
  const lens = strategy?.execution_lens || execution?.execution_lens;
  if (lens?.headline) return lens.headline;
  const direction = compactDirection(strategy).toUpperCase();
  const timing = compactTiming(strategy, execution).toUpperCase();
  const quality = compactEntryQuality(review, strategy).toUpperCase();
  return `${direction} • ${timing} • ${quality}`;
}

function activeLimitPlan(strategy?: Strategy, execution?: ExecutionPlan) {
  return strategy?.limit_order_plan || execution?.limit_order_plan;
}

function marketPermissionTone(plan?: LimitOrderPlan) {
  if (plan?.market_entry_allowed) return badgeMap.strongBull;
  if (plan?.enabled) return badgeMap.strongBear;
  return badgeMap.neutral;
}

function marketPermissionLabel(plan?: LimitOrderPlan) {
  if (plan?.market_entry_allowed) return "Market Allowed";
  if (plan?.enabled) return "Market Disabled";
  return "No Market Plan";
}

function executionMethodLabel(plan?: LimitOrderPlan, lens?: ExecutionLens) {
  const preference = (plan?.preference || "").toLowerCase();
  if (preference === "limit_only") return "Limit Only";
  if (preference === "limit_preferred") return "Limit Preferred";
  if (preference === "market_or_limit") return "Market Or Limit";
  if (preference === "monitor_then_limit") return "Monitor Then Limit";
  if (lens?.primary_lane === "directional_watch") return "Monitor";
  return "Wait";
}

function buildNextActionText(plan?: LimitOrderPlan, lens?: ExecutionLens) {
  const side = prettyLabel(plan?.side || lens?.direction || "");
  const limit = formatNumber(plan?.limit_price, 4);
  if (lens?.primary_lane === "scalp" && plan?.market_entry_allowed) {
    return plan?.limit_price
      ? `Staged market entry is allowed; passive ${side.toLowerCase()} limit near ${limit} is still cleaner.`
      : "Staged market entry is allowed; avoid chasing outside the planned band.";
  }
  if (lens?.primary_lane === "scalp" && plan?.enabled) {
    return `Wait for the ${side.toLowerCase()} retest near ${limit}; do not chase market price.`;
  }
  if (lens?.primary_lane === "directional_watch") {
    return "Track the directional read, but wait for the scalp lane to confirm before execution.";
  }
  return "Stand aside until the engine promotes a cleaner setup.";
}

function buildDecisionCommand(strategy?: Strategy, execution?: ExecutionPlan) {
  const lens = strategy?.execution_lens || execution?.execution_lens;
  const plan = activeLimitPlan(strategy, execution);
  const lane = lens?.primary_lane || "legacy";
  const marketAllowed = Boolean(plan?.market_entry_allowed);
  const preference = (plan?.preference || "").toLowerCase();
  const direction = prettyLabel(lens?.direction || strategy?.strategy_bias || execution?.direction || "neutral");
  const signal = lens?.scalp?.label || lens?.directional?.label || compactTiming(strategy, execution);
  const method = executionMethodLabel(plan, lens);
  const permission = marketPermissionLabel(plan);
  const reason = plan?.reason || lens?.scalp?.blockers?.[0] || lens?.summary || execution?.execution_instruction || strategy?.summary || "Waiting for a clear trade instruction.";

  let headline = "STAND ASIDE";
  let tone = badgeMap.neutral;
  if (lane === "scalp" && marketAllowed && preference !== "limit_only") {
    headline = "ACTIONABLE NOW";
    tone = badgeMap.strongBull;
  } else if (lane === "scalp" && plan?.enabled) {
    headline = "WAIT FOR RETEST";
    tone = badgeMap.pullback;
  } else if (lane === "directional_watch") {
    headline = "DIRECTIONAL ONLY";
    tone = badgeMap.watch;
  } else if (lane === "scalp") {
    headline = "SCALP FORMING";
    tone = badgeMap.watch;
  }

  return {
    headline,
    tone,
    direction,
    signal,
    method,
    permission,
    reason,
    nextAction: buildNextActionText(plan, lens),
    lane,
    plan,
    lens,
  };
}

function firstNonEmpty<T>(...values: Array<T | null | undefined>): T | undefined {
  return values.find((value) => value !== null && value !== undefined);
}

function formatZone(zone?: PriceZone | null) {
  if (!zone) return "—";
  return `${formatNumber(zone.low, 4)} – ${formatNumber(zone.high, 4)}`;
}

function metricLabelTone(value: number | null | undefined, goodThreshold: number, watchThreshold: number) {
  if (value === null || value === undefined) return badgeMap.neutral;
  if (value >= goodThreshold) return badgeMap.strongBull;
  if (value >= watchThreshold) return badgeMap.watch;
  return badgeMap.pullback;
}

function Metric({
  label,
  value,
  className = "",
}: {
  label: string;
  value: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`${METRIC_PANEL} p-3 ${className}`}>
      <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">{label}</div>
      <div className="mt-1.5 text-lg font-semibold text-slate-50">{value}</div>
    </div>
  );
}

function SectionHeading({ icon: Icon, title, subtitle }: { icon: React.ComponentType<{ className?: string }>; title: string; subtitle: string }) {
  return (
    <div className="mb-4 flex items-start gap-3">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-2">
        <Icon className="h-4 w-4 text-cyan-300" />
      </div>
      <div>
        <h2 className="text-xl font-semibold text-white">{title}</h2>
        <p className="mt-1 text-sm text-slate-400">{subtitle}</p>
      </div>
    </div>
  );
}


type VisualLegendRow = {
  key: string;
  label: string;
  value: React.ReactNode;
  tone: string;
  helper?: string;
};

function ExecutionVisualCard({
  data,
  strategy,
  execution,
  review,
  plan,
  currentPrice,
}: {
  data?: EngineResponse | null;
  strategy?: Strategy;
  execution?: ExecutionPlan;
  review?: ExecutionReview;
  plan?: LimitOrderPlan;
  currentPrice?: number;
}) {
  const [hoveredLevel, setHoveredLevel] = useState<string | null>(null);

  const historySource = data?.timeframes?.["5m"]?.history || data?.timeframes?.["15m"]?.history || [];
  const priceHistory = useMemo(() => {
    const clean = (historySource || [])
      .map((value) => asNumber(value))
      .filter((value): value is number => value !== undefined);
    const latest = asNumber(currentPrice);
    const limited = clean.slice(-30);

    if (latest === undefined) return limited;
    if (!limited.length) return [latest];

    const last = limited[limited.length - 1];
    if (Math.abs(last - latest) > 0.0000001) {
      return [...limited.slice(-29), latest];
    }
    return limited;
  }, [historySource, currentPrice]);

  const direction = (execution?.direction || strategy?.strategy_bias || "neutral").toLowerCase();
  const isShort = direction === "short";
  const latestPrice = asNumber(currentPrice) ?? priceHistory[priceHistory.length - 1];
  const previousPrice = priceHistory.length >= 2 ? priceHistory[priceHistory.length - 2] : undefined;
  const priceChange = latestPrice !== undefined && previousPrice !== undefined ? latestPrice - previousPrice : 0;
  const pathStroke = priceChange > 0 ? "#34d399" : priceChange < 0 ? "#fb7185" : "#22d3ee";
  const mutedCurrent = plan?.market_entry_allowed === false || (plan?.preference || "").toLowerCase() === "limit_only";

  const entryZone = plan?.entry_zone || execution?.entry_zone || null;
  const invalidationZone = execution?.invalidation_zone || null;
  const takeProfit = execution?.take_profit_zone || null;
  const limitPrice = asNumber(plan?.limit_price);
  const backupLimitPrice = asNumber(plan?.backup_limit_price);
  const invalidationPrice = asNumber(plan?.invalidation_price);
  const tp1 = asNumber(takeProfit?.tp1);
  const tp2 = asNumber(takeProfit?.tp2);
  const tp3 = asNumber(takeProfit?.tp3);

  const levelValues = [
    ...priceHistory,
    latestPrice,
    limitPrice,
    backupLimitPrice,
    asNumber(entryZone?.low),
    asNumber(entryZone?.high),
    asNumber(entryZone?.mid),
    invalidationPrice,
    asNumber(invalidationZone?.low),
    asNumber(invalidationZone?.high),
    tp1,
    tp2,
    tp3,
  ].filter((value): value is number => value !== undefined);

  const chartMinRaw = levelValues.length ? Math.min(...levelValues) : 0;
  const chartMaxRaw = levelValues.length ? Math.max(...levelValues) : 1;
  const chartRangeRaw = Math.max(chartMaxRaw - chartMinRaw, latestPrice ? latestPrice * 0.0015 : 0.001);
  const chartMin = chartMinRaw - chartRangeRaw * 0.18;
  const chartMax = chartMaxRaw + chartRangeRaw * 0.18;
  const chartRange = Math.max(chartMax - chartMin, 0.000001);

  const width = 940;
  const height = 360;
  const padLeft = 42;
  const padRight = 24;
  const padTop = 34;
  const padBottom = 34;
  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const xForIndex = (index: number) => {
    if (priceHistory.length <= 1) return padLeft + chartWidth;
    return padLeft + (index / (priceHistory.length - 1)) * chartWidth;
  };

  const yForPrice = (price?: number) => {
    if (price === undefined) return padTop + chartHeight / 2;
    return padTop + ((chartMax - price) / chartRange) * chartHeight;
  };

  const pricePath = priceHistory
    .map((value, index) => `${index === 0 ? "M" : "L"}${xForIndex(index).toFixed(2)} ${yForPrice(value).toFixed(2)}`)
    .join(" ");

  const areaPath = priceHistory.length
    ? `${pricePath} L ${xForIndex(priceHistory.length - 1).toFixed(2)} ${height - padBottom} L ${xForIndex(0).toFixed(2)} ${height - padBottom} Z`
    : "";

  const currentX = priceHistory.length ? xForIndex(priceHistory.length - 1) : padLeft + chartWidth;
  const currentY = yForPrice(latestPrice);

  type LevelLine = {
    key: string;
    label: string;
    value?: number;
    stroke: string;
    dash?: string;
    width?: number;
  };

  const lines: LevelLine[] = [
    { key: "tp3", label: "TP3", value: tp3, stroke: "#10b981", dash: "8 8" },
    { key: "tp2", label: "TP2", value: tp2, stroke: "#34d399", dash: "8 8" },
    { key: "tp1", label: "TP1", value: tp1, stroke: "#6ee7b7", dash: "8 8" },
    { key: "limit", label: "Limit", value: limitPrice, stroke: "#a78bfa", dash: "7 7", width: 2.5 },
    { key: "backup", label: "Backup", value: backupLimitPrice, stroke: "#fbbf24", dash: "6 6" },
    { key: "current", label: "Current", value: latestPrice, stroke: pathStroke, width: 2.2 },
    { key: "invalidation", label: "Invalidation", value: invalidationPrice, stroke: "#fb7185", dash: "6 5", width: 2.2 },
  ].filter((line) => line.value !== undefined);

  const highlighted = (key: string) => !hoveredLevel || hoveredLevel === key;
  const yGridLevels = [0, 0.25, 0.5, 0.75, 1].map((ratio) => chartMin + chartRange * ratio);
  const limitBandLow = Math.min(limitPrice ?? Infinity, backupLimitPrice ?? Infinity);
  const limitBandHigh = Math.max(limitPrice ?? -Infinity, backupLimitPrice ?? -Infinity);
  const hasLimitBand = Number.isFinite(limitBandLow) && Number.isFinite(limitBandHigh) && limitBandHigh > limitBandLow;

  const legendRows: VisualLegendRow[] = [
    {
      key: "current",
      label: "Current",
      value: formatNumber(latestPrice, 4),
      tone: "bg-cyan-400",
      helper: mutedCurrent ? "Dimmed: planner prefers patience" : "Live reference",
    },
    {
      key: "limit",
      label: "Limit",
      value: formatNumber(limitPrice, 4),
      tone: "bg-violet-400",
      helper: prettyLabel(plan?.preference),
    },
    {
      key: "backup",
      label: "Backup Limit",
      value: formatNumber(backupLimitPrice, 4),
      tone: "bg-amber-300",
      helper: "Secondary passive level",
    },
    {
      key: "entry_zone",
      label: "Entry Band",
      value: formatZone(entryZone),
      tone: "bg-sky-400",
      helper: entryZone?.label ? prettyLabel(entryZone.label) : "Acceptable execution zone",
    },
    {
      key: "invalidation_zone",
      label: "Invalidation",
      value: invalidationPrice !== undefined ? formatNumber(invalidationPrice, 4) : formatZone(invalidationZone),
      tone: "bg-rose-400",
      helper: "Cancel / wrong-side reference",
    },
    {
      key: "tp1",
      label: "TP1",
      value: formatNumber(tp1, 4),
      tone: "bg-emerald-300",
      helper: latestPrice !== undefined && tp1 !== undefined ? `${formatPercent(directionalDistancePercent(direction, latestPrice, tp1), 3)} away` : "First scale-out",
    },
    {
      key: "tp2",
      label: "TP2",
      value: formatNumber(tp2, 4),
      tone: "bg-emerald-400",
      helper: latestPrice !== undefined && tp2 !== undefined ? `${formatPercent(directionalDistancePercent(direction, latestPrice, tp2), 3)} away` : "Second scale-out",
    },
    {
      key: "tp3",
      label: "TP3",
      value: formatNumber(tp3, 4),
      tone: "bg-emerald-500",
      helper: "Runner target",
    },
  ];

  const hoverRow = legendRows.find((row) => row.key === hoveredLevel);
  const plannerState = prettyLabel(plan?.planner_state || "Monitor");
  const plannerPreference = prettyLabel(plan?.preference || "Monitor");
  const tp1Distance = directionalDistancePercent(direction, latestPrice, tp1);
  const stopReference = invalidationPrice ?? (isShort ? asNumber(invalidationZone?.low) : asNumber(invalidationZone?.high));
  const stopDistance = directionalDistancePercent(direction, latestPrice, stopReference);
  const riskDistance = stopDistance !== undefined ? Math.abs(stopDistance) : undefined;
  const rewardDistance = tp1Distance !== undefined ? Math.max(tp1Distance, 0) : undefined;
  const currentRewardRisk = riskDistance && rewardDistance !== undefined ? rewardDistance / riskDistance : undefined;
  const marketEntryText = plan ? (plan.market_entry_allowed ? "Allowed" : "Disabled") : "—";

  function renderZone(zone: PriceZone | null | undefined, key: string, fill: string, stroke: string, opacity = 0.18) {
    const low = asNumber(zone?.low);
    const high = asNumber(zone?.high);
    if (low === undefined || high === undefined) return null;
    const yTop = yForPrice(Math.max(low, high));
    const yBottom = yForPrice(Math.min(low, high));
    return (
      <g
        key={key}
        className="cursor-pointer"
        opacity={highlighted(key) ? 1 : 0.22}
        onMouseEnter={() => setHoveredLevel(key)}
        onMouseLeave={() => setHoveredLevel(null)}
      >
        <rect
          x={padLeft}
          y={yTop}
          width={chartWidth}
          height={Math.max(yBottom - yTop, 2)}
          rx="8"
          fill={fill}
          fillOpacity={opacity}
          stroke={stroke}
          strokeOpacity="0.35"
        />
      </g>
    );
  }

  function renderLine(line: LevelLine) {
    if (line.value === undefined) return null;
    const active = highlighted(line.key);
    const isCurrent = line.key === "current";
    const y = yForPrice(line.value);
    return (
      <g
        key={line.key}
        className="cursor-pointer"
        opacity={active ? (isCurrent && mutedCurrent ? 0.58 : 1) : 0.22}
        onMouseEnter={() => setHoveredLevel(line.key)}
        onMouseLeave={() => setHoveredLevel(null)}
      >
        <line
          x1={padLeft}
          x2={padLeft + chartWidth}
          y1={y}
          y2={y}
          stroke={line.stroke}
          strokeWidth={active ? (line.width || 1.6) + 0.6 : line.width || 1.4}
          strokeDasharray={line.dash}
          strokeOpacity={isCurrent && mutedCurrent ? 0.45 : 0.88}
          filter={active && !isCurrent ? "url(#executionSoftGlow)" : undefined}
        />
      </g>
    );
  }

  function LegendRow({ row }: { row: VisualLegendRow }) {
    const active = highlighted(row.key);
    return (
      <button
        type="button"
        onMouseEnter={() => setHoveredLevel(row.key)}
        onMouseLeave={() => setHoveredLevel(null)}
        className={`w-full rounded-2xl border px-3 py-2.5 text-left transition ${
          active ? "border-slate-600 bg-slate-950/70" : "border-slate-800 bg-slate-950/30 opacity-45"
        }`}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${row.tone}`} />
            <span className="truncate text-xs font-semibold uppercase tracking-[0.12em] text-slate-300">{row.label}</span>
          </div>
          <span className="shrink-0 text-sm font-semibold text-slate-50">{row.value}</span>
        </div>
        {row.helper ? <div className="mt-1 truncate text-xs text-slate-500">{row.helper}</div> : null}
      </button>
    );
  }

  return (
    <Card className={`${PANEL} relative overflow-hidden`}>
      <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${visualAmbientTone(strategy, plan)} opacity-90`} />
      <CardHeader className="relative">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <CardTitle className="text-2xl text-white">Execution Visual</CardTitle>
            <p className="mt-1 text-sm text-slate-400">
              Price path, entry band, limit plan, invalidation, and TP ladder in one execution map.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 xl:justify-end">
            <Badge className={`${PILL_BASE} ${directionTone(strategy?.strategy_bias)}`}>{prettyLabel(strategy?.strategy_bias || "Neutral")} Bias</Badge>
            <Badge className={`${PILL_BASE} ${plannerTone(plan)}`}>{plannerPreference}</Badge>
            <Badge className={`${PILL_BASE} ${plan?.market_entry_allowed ? badgeMap.pullback : badgeMap.strongBear}`}>Market {marketEntryText}</Badge>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-4">
          <Metric label="Decision" value={buildLiveSentence(strategy, execution, review)} />
          <Metric label="Entry Score" value={`${formatNumber(strategy?.entry_score?.score, 0)} · ${prettyLabel(strategy?.entry_score?.tier)}`} />
          <Metric label="Planner State" value={plannerState} className={plannerTone(plan)} />
        </div>
      </CardHeader>

      <CardContent className="relative space-y-4">
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
          <div className={`${SUBPANEL} relative overflow-hidden p-3`}>
            <div className={`pointer-events-none absolute inset-0 transition-opacity duration-500 ${mutedCurrent ? "bg-violet-500/10" : "bg-cyan-500/5"}`} />
            {hoverRow ? (
              <div className="absolute left-5 top-5 z-10 rounded-2xl border border-slate-700/80 bg-slate-950/85 px-3 py-2 shadow-2xl backdrop-blur">
                <div className="text-[10px] uppercase tracking-[0.14em] text-slate-400">Hover</div>
                <div className="mt-1 text-sm font-semibold text-white">{hoverRow.label}: {hoverRow.value}</div>
                {hoverRow.helper ? <div className="mt-1 max-w-[280px] text-xs text-slate-400">{hoverRow.helper}</div> : null}
              </div>
            ) : null}

            <svg viewBox={`0 0 ${width} ${height}`} className="relative h-[360px] w-full overflow-visible">
              <defs>
                <linearGradient id="executionAreaGradient" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor={pathStroke} stopOpacity="0.22" />
                  <stop offset="100%" stopColor={pathStroke} stopOpacity="0.01" />
                </linearGradient>
                <filter id="executionSoftGlow" x="-25%" y="-25%" width="150%" height="150%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              <rect x="0" y="0" width={width} height={height} rx="24" fill="rgba(2,6,23,0.45)" />
              {yGridLevels.map((value) => {
                const y = yForPrice(value);
                return (
                  <g key={`grid-${value}`}>
                    <line x1={padLeft} x2={padLeft + chartWidth} y1={y} y2={y} stroke="rgba(148,163,184,0.12)" strokeWidth="1" />
                    <text x={padLeft + chartWidth + 8} y={y + 4} fill="rgba(203,213,225,0.58)" fontSize="11">
                      {formatNumber(value, 4)}
                    </text>
                  </g>
                );
              })}

              {renderZone(entryZone, "entry_zone", "#38bdf8", "#38bdf8", 0.16)}
              {hasLimitBand ? (
                <g
                  className="cursor-pointer"
                  opacity={highlighted("limit") || highlighted("backup") ? 1 : 0.22}
                  onMouseEnter={() => setHoveredLevel("limit")}
                  onMouseLeave={() => setHoveredLevel(null)}
                >
                  <rect
                    x={padLeft}
                    y={yForPrice(limitBandHigh)}
                    width={chartWidth}
                    height={Math.max(yForPrice(limitBandLow) - yForPrice(limitBandHigh), 2)}
                    rx="8"
                    fill="#a78bfa"
                    fillOpacity="0.11"
                    stroke="#a78bfa"
                    strokeOpacity="0.25"
                  />
                </g>
              ) : null}
              {renderZone(invalidationZone, "invalidation_zone", "#fb7185", "#fb7185", 0.12)}

              {areaPath ? <path d={areaPath} fill="url(#executionAreaGradient)" opacity={mutedCurrent ? 0.42 : 0.64} /> : null}
              {lines.map(renderLine)}
              {pricePath ? (
                <path
                  key={data?.generated_at || pricePath}
                  d={pricePath}
                  fill="none"
                  stroke={pathStroke}
                  strokeWidth="3.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  pathLength={1}
                  className="execution-price-path"
                  opacity={mutedCurrent ? 0.74 : 1}
                  filter="url(#executionSoftGlow)"
                />
              ) : null}

              {latestPrice !== undefined ? (
                <g opacity={mutedCurrent ? 0.66 : 1}>
                  <circle cx={currentX} cy={currentY} r="10" fill="none" stroke={pathStroke} strokeWidth="4" className="execution-price-pulse" />
                  <circle cx={currentX} cy={currentY} r="5" fill={pathStroke} stroke="#e2e8f0" strokeWidth="1.5" />
                </g>
              ) : null}
            </svg>
          </div>

          <div className={`${SUBPANEL} p-4`}>
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-white">Level Legend</div>
                <div className="mt-1 text-xs text-slate-500">Hover a row to highlight the map.</div>
              </div>
              <Badge className={`${PILL_BASE} ${plannerTone(plan)}`}>{prettyLabel(plan?.side || "Watch")}</Badge>
            </div>
            <div className="space-y-2">
              {legendRows.map((row) => <LegendRow key={row.key} row={row} />)}
            </div>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
          <Metric label="Chart Low" value={formatNumber(chartMinRaw, 4)} />
          <Metric label="Chart High" value={formatNumber(chartMaxRaw, 4)} />
          <Metric label="Path Points" value={priceHistory.length} />
          <Metric label="TP1 Distance" value={formatPercent(tp1Distance, 3)} />
          <Metric label="Stop Distance" value={stopDistance !== undefined ? formatPercent(stopDistance, 3) : "—"} />
          <Metric label="Current R:R" value={currentRewardRisk !== undefined ? formatNumber(currentRewardRisk, 2) : "—"} />
        </div>

        {plan?.historical_hint ? (
          <div className={`${SUBPANEL} border-cyan-500/20 p-4 text-sm leading-7 text-slate-300`}>
            <span className="font-semibold text-cyan-200">Retest hint:</span> {plan.historical_hint}
          </div>
        ) : null}
      </CardContent>

      <style>{`
        @keyframes executionPriceDraw {
          from { stroke-dashoffset: 1; }
          to { stroke-dashoffset: 0; }
        }
        @keyframes executionPulseSvg {
          0%, 100% { opacity: 0.20; stroke-width: 4; }
          50% { opacity: 0.78; stroke-width: 9; }
        }
        .execution-price-path {
          stroke-dasharray: 1;
          stroke-dashoffset: 0;
          animation: executionPriceDraw 850ms cubic-bezier(0.22, 1, 0.36, 1) both;
          transition: opacity 400ms ease, filter 400ms ease;
        }
        .execution-price-pulse {
          animation: executionPulseSvg 1.8s ease-in-out infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .execution-price-path,
          .execution-price-pulse {
            animation: none;
          }
        }
      `}</style>
    </Card>
  );
}

function DecisionBar({
  strategy,
  execution,
  review,
}: {
  strategy?: Strategy;
  execution?: ExecutionPlan;
  review?: ExecutionReview;
}) {
  const lens = strategy?.execution_lens || execution?.execution_lens;
  const scalp = lens?.scalp;
  const directional = lens?.directional;
  const command = buildDecisionCommand(strategy, execution);

  return (
    <Card className={PANEL}>
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className={`mb-3 inline-flex rounded-full ${PILL_BASE} ${command.tone}`}>{command.headline}</div>
            <CardTitle className="text-3xl text-white">{command.nextAction}</CardTitle>
            <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300">{command.reason}</p>
          </div>
          <div className="flex flex-wrap gap-2 lg:justify-end">
            <Badge className={`${PILL_BASE} ${directionTone(command.direction)}`}>{command.direction}</Badge>
            <Badge className={`${PILL_BASE} ${lensLaneTone(scalp)}`}>{command.signal}</Badge>
            <Badge className={`${PILL_BASE} ${plannerTone(command.plan)}`}>{command.method}</Badge>
            <Badge className={`${PILL_BASE} ${marketPermissionTone(command.plan)}`}>{command.permission}</Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-4">
          <div className={`${SUBPANEL} p-4`}>
            <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Action Lane</div>
            <Badge className={`mt-3 ${PILL_BASE} ${laneTone(lens?.primary_lane)}`}>{prettyLabel(lens?.primary_lane || "legacy")}</Badge>
          </div>
          <div className={`${SUBPANEL} p-4`}>
            <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Direction</div>
            <Badge className={`mt-3 ${PILL_BASE} ${directionTone(command.direction)}`}>{command.direction}</Badge>
          </div>
          <div className={`${SUBPANEL} p-4`}>
            <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Execution Method</div>
            <Badge className={`mt-3 ${PILL_BASE} ${plannerTone(command.plan)}`}>{command.method}</Badge>
          </div>
          <div className={`${SUBPANEL} p-4`}>
            <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Market Entry</div>
            <Badge className={`mt-3 ${PILL_BASE} ${marketPermissionTone(command.plan)}`}>{command.permission}</Badge>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Entry Score" value={`${formatNumber(strategy?.entry_score?.score, 0)} · ${prettyLabel(strategy?.entry_score?.tier)}`} />
          <Metric label="Scalp Score" value={formatNumber(scalp?.score, 0)} className={lensLaneTone(scalp)} />
          <Metric label="Directional Score" value={formatNumber(directional?.score, 0)} className={lensLaneTone(directional)} />
          <Metric label="Entry Quality" value={compactEntryQuality(review, strategy)} className={efficiencyTone(review?.execution_efficiency_label)} />
          <Metric label="Risk State" value={prettyLabel(strategy?.risk_state)} className={riskTone(strategy?.risk_state)} />
        </div>
        {lens?.notes?.length ? (
          <div className={`${SUBPANEL} p-4 text-sm leading-6 text-slate-300`}>
            {lens.notes.slice(0, 2).join(" ")}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function PricePlanCard({
  currentPrice,
  strategy,
  execution,
  review,
}: {
  currentPrice?: number;
  strategy?: Strategy;
  execution?: ExecutionPlan;
  review?: ExecutionReview;
}) {
  const plan = activeLimitPlan(strategy, execution);
  const command = buildDecisionCommand(strategy, execution);
  return (
    <Card className={PANEL}>
      <CardHeader>
        <CardTitle className="text-2xl text-white">Price Plan</CardTitle>
        <p className="text-sm text-slate-400">This is the only map you should need during live execution.</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
          <Metric label="Current Price" value={formatNumber(currentPrice, 4)} />
          <Metric label="Entry Band" value={formatZone(execution?.entry_zone)} className={entryLocationTone(review?.entry_location)} />
          <Metric label="Invalidation" value={formatZone(execution?.invalidation_zone)} />
          <Metric label="TP Ladder" value={`${formatNumber(execution?.take_profit_zone?.tp1, 4)} / ${formatNumber(execution?.take_profit_zone?.tp2, 4)} / ${formatNumber(execution?.take_profit_zone?.tp3, 4)}`} />
          <Metric label="Market Entry" value={command.permission} className={marketPermissionTone(plan)} />
          <Metric label="Setup / Trigger" value={`${(strategy?.setup_timeframes || []).join(", ") || "—"} → ${(strategy?.trigger_timeframes || []).join(", ") || "—"}`} />
        </div>

        <div className={`${SUBPANEL} p-4`}>
          <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Next Action</div>
          <div className="space-y-2 text-sm text-slate-200">
            <p>{command.nextAction}</p>
            <p className="text-slate-400">{command.reason}</p>
            {review?.notes?.[0] ? <p className="text-slate-500">{review.notes[0]}</p> : null}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function EntryQualityCard({ review }: { review?: ExecutionReview }) {
  return (
    <Card className={PANEL}>
      <CardHeader>
        <CardTitle className="text-2xl text-white">Entry Quality</CardTitle>
        <p className="text-sm text-slate-400">One quick read on whether the current price is helping or hurting you.</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Badge className={`${PILL_BASE} ${efficiencyTone(review?.execution_efficiency_label)}`}>
            {prettyLabel(review?.execution_efficiency_label)} · {formatNumber(review?.execution_efficiency_score, 0)}
          </Badge>
          <Badge className={`${PILL_BASE} ${entryLocationTone(review?.entry_location)}`}>
            {prettyLabel(review?.entry_location)}
          </Badge>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Market" value={formatNumber(review?.market_entry_price, 4)} />
          <Metric label="Plan" value={formatNumber(review?.planned_entry_price, 4)} />
          <Metric label="Ideal" value={formatNumber(review?.ideal_entry_price, 4)} />
          <Metric label="Zone Width" value={formatPercent(review?.zone_width_pct, 3)} />
        </div>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Market vs Plan" value={formatPercent(review?.market_vs_planned_pct, 3)} />
          <Metric label="Market vs Ideal" value={formatPercent(review?.market_vs_ideal_pct, 3)} />
          <Metric label="Plan Touched" value={boolLabel(review?.planned_entry_touched_after_signal)} />
          <Metric label="Zone Retested" value={boolLabel(review?.entry_zone_retested_after_signal)} />
        </div>
        {review?.notes?.length ? (
          <div className={`${SUBPANEL} p-4 text-sm text-slate-300`}>
            {review.notes.map((note, idx) => (
              <div key={`review-note-${idx}`} className="mb-2 last:mb-0">{note}</div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function ReviewTradeCard({ trade }: { trade: TradeRecord }) {
  return (
    <div className={`${SUBPANEL} p-4`}>
      <div className="flex flex-wrap items-center gap-2">
        <Badge className={`${PILL_BASE} ${directionTone(trade.direction)}`}>{prettyLabel(trade.direction)}</Badge>
        <Badge className={`${PILL_BASE} ${tradeOutcomeTone(trade.outcome_status || trade.outcome_label)}`}>
          {prettyLabel(trade.outcome_label || trade.outcome_status)}
        </Badge>
        <Badge className={`${PILL_BASE} ${actionTone(trade.opening_action)}`}>{prettyLabel(trade.opening_action)}</Badge>
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Opened" value={formatTime(trade.opened_at)} />
        <Metric label="Resolved P/L" value={formatPercent(trade.market_entry_resolved_pct, 3)} />
        <Metric label="Plan Edge" value={formatPercent(trade.planned_edge_vs_market_resolved_pct, 3)} />
        <Metric label="Efficiency" value={`${formatNumber(trade.execution_efficiency_score, 0)} · ${prettyLabel(trade.execution_efficiency_label)}`} className={efficiencyTone(trade.execution_efficiency_label)} />
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4 text-sm text-slate-300">
        <div><span className="text-slate-400">Entry:</span> {formatNumber(trade.entry_market_price, 4)}</div>
        <div><span className="text-slate-400">Planned:</span> {formatNumber(trade.planned_entry_price, 4)}</div>
        <div><span className="text-slate-400">Held:</span> {trade.snapshots_held ?? "—"} snapshots</div>
        <div><span className="text-slate-400">TP1 / Invalidation:</span> {trade.tp1_hit ? "Yes" : "No"} / {trade.invalidation_hit ? "Yes" : "No"}</div>
      </div>
      {trade.notes?.[0] ? <p className="mt-3 text-sm text-slate-400">{trade.notes[0]}</p> : null}
    </div>
  );
}

function ReviewMode({ summary, trades }: { summary?: TradeTrackerSummary; trades: TradeRecord[] }) {
  const openTrades = trades.filter((trade) => (trade.status || "").toLowerCase() === "open");
  const closedTrades = trades.filter((trade) => (trade.status || "").toLowerCase() !== "open");

  const trackerReadout = useMemo(() => {
    const winRate = Number(summary?.win_rate ?? 0);
    const avgPnl = Number(summary?.avg_market_entry_resolved_pct ?? 0);
    if (winRate >= 85 && avgPnl < 0.05) {
      return "High hit-rate, small-move system. The edge is real, but entry discipline matters more than signal hunting.";
    }
    if (winRate >= 60) {
      return "The engine is finding workable ideas, but the spread between plan and market entry is still a meaningful source of leakage.";
    }
    return "Use this review screen to find which trade families deserve trust and which ones need tighter filtering.";
  }, [summary]);

  return (
    <div className="space-y-6">
      <Card className={PANEL}>
        <CardHeader>
          <CardTitle className="text-2xl text-white">Review</CardTitle>
          <p className="text-sm text-slate-400">This view tells you how the engine has actually been doing over time.</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
            <Metric label="Total Trades" value={summary?.total_trades ?? 0} />
            <Metric label="Open" value={summary?.open_trades ?? 0} />
            <Metric label="Closed" value={summary?.closed_trades ?? 0} />
            <Metric label="Win Rate" value={formatPercent(summary?.win_rate, 1, false)} className={metricLabelTone(summary?.win_rate, 75, 60)} />
            <Metric label="Avg Market P/L" value={formatPercent(summary?.avg_market_entry_resolved_pct, 3)} />
            <Metric label="Avg Efficiency" value={formatNumber(summary?.avg_execution_efficiency_score, 0)} />
          </div>
          <div className={`${SUBPANEL} p-4 text-sm text-slate-300`}>{trackerReadout}</div>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card className={PANEL}>
          <CardHeader>
            <CardTitle className="text-xl text-white">Open Trades</CardTitle>
            <p className="text-sm text-slate-400">Active trade families that have not resolved yet.</p>
          </CardHeader>
          <CardContent className="space-y-4">
            {openTrades.length ? openTrades.map((trade) => <ReviewTradeCard key={trade.trade_id} trade={trade} />) : <div className="text-sm text-slate-400">No open trade families right now.</div>}
          </CardContent>
        </Card>

        <Card className={PANEL}>
          <CardHeader>
            <CardTitle className="text-xl text-white">Recent Closed Trades</CardTitle>
            <p className="text-sm text-slate-400">Recently resolved engine trade families.</p>
          </CardHeader>
          <CardContent className="space-y-4">
            {closedTrades.length ? closedTrades.slice(0, 6).map((trade) => <ReviewTradeCard key={trade.trade_id} trade={trade} />) : <div className="text-sm text-slate-400">No closed trades yet.</div>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}


function LimitOrderPlannerCard({ plan }: { plan?: LimitOrderPlan }) {
  const enabled = Boolean(plan?.enabled);
  const cancelRules = plan?.cancel_if || [];
  const promotionRules = plan?.promotion_rules || [];

  return (
    <Card className={PANEL}>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="text-2xl text-white">Limit Order Planner</CardTitle>
            <p className="mt-1 text-sm text-slate-400">
              Passive-entry guidance from the v36 planner. This is a plan to observe and test, not automatic execution.
            </p>
          </div>
          <Badge className={`${PILL_BASE} ${plannerTone(plan)}`}>
            {enabled ? prettyLabel(plan?.preference) : prettyLabel(plan?.planner_state || "Inactive")}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
          <Metric label="Planner State" value={prettyLabel(plan?.planner_state)} />
          <Metric label="Side" value={prettyLabel(plan?.side)} className={limitSideTone(plan?.side)} />
          <Metric label="Limit" value={formatNumber(plan?.limit_price, 4)} />
          <Metric label="Backup" value={formatNumber(plan?.backup_limit_price, 4)} />
          <Metric label="Invalidation" value={formatNumber(plan?.invalidation_price, 4)} />
          <Metric label="TP Ref" value={formatNumber(plan?.take_profit_reference, 4)} />
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Market Entry" value={plan?.market_entry_allowed ? "Allowed" : "Disabled"} className={plan?.market_entry_allowed ? badgeMap.pullback : badgeMap.strongBear} />
          <Metric label="Expiry" value={plan?.expiry_snapshots ? `${formatNumber(plan.expiry_snapshots, 0)} snapshots` : "—"} />
          <Metric label="Approx Time" value={plan?.expiry_minutes_estimate ? `${formatNumber(plan.expiry_minutes_estimate, 0)} min` : "—"} />
          <Metric label="Order Type" value={prettyLabel(plan?.order_type || "limit")} />
        </div>

        <div className={`${SUBPANEL} p-4`}>
          <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Planner read</div>
          <p className="text-sm leading-7 text-slate-200">
            {plan?.reason || "No limit-order plan is active yet."}
          </p>
          {plan?.historical_hint ? <p className="mt-2 text-sm leading-6 text-slate-400">{plan.historical_hint}</p> : null}
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <div className={`${SUBPANEL} p-4`}>
            <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Cancel if</div>
            {cancelRules.length ? (
              <ul className="space-y-2 text-sm text-slate-300">
                {cancelRules.map((rule, idx) => (
                  <li key={`cancel-${idx}`}>{rule}</li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-slate-400">No cancel rules while planner is inactive.</div>
            )}
          </div>
          <div className={`${SUBPANEL} p-4`}>
            <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Promote if</div>
            {promotionRules.length ? (
              <ul className="space-y-2 text-sm text-slate-300">
                {promotionRules.map((rule, idx) => (
                  <li key={`promote-${idx}`}>{rule}</li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-slate-400">No promotion rules while planner is inactive.</div>
            )}
          </div>
        </div>

        {plan?.notes?.length ? (
          <div className={`${SUBPANEL} p-4 text-sm text-slate-300`}>
            {plan.notes.map((note, idx) => (
              <div key={`limit-note-${idx}`} className="mb-2 last:mb-0">{note}</div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function ManagementCard({ management }: { management?: ExecutionManagement }) {
  const scaleOut = management?.scale_out_plan || [];
  return (
    <Card className={PANEL}>
      <CardHeader>
        <CardTitle className="text-2xl text-white">Trade Management</CardTitle>
        <p className="text-sm text-slate-400">How to manage the position after entry: stop, scale-outs, and runner rules.</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Stop Policy" value={prettyLabel(management?.stop_policy)} />
          <Metric label="Stop Distance" value={formatPercent(management?.stop_distance_pct, 3, false)} />
          <Metric label="Break Even After" value={prettyLabel(management?.break_even_after)} />
          <Metric label="Trail After" value={prettyLabel(management?.trailing_stop_after)} />
        </div>

        <div className={`${SUBPANEL} p-4`}>
          <div className="mb-3 text-[11px] uppercase tracking-[0.14em] text-slate-400">Scale-out plan</div>
          {scaleOut.length ? (
            <div className="grid gap-3 md:grid-cols-3">
              {scaleOut.map((step, idx) => (
                <div key={`scale-out-${idx}`} className={METRIC_PANEL + " p-3"}>
                  <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">{prettyLabel(step.target)}</div>
                  <div className="mt-1.5 text-lg font-semibold text-slate-50">
                    {formatNumber(step.size_pct, 0)}% @ {formatNumber(step.price, 4)}
                  </div>
                  <p className="mt-2 text-xs leading-5 text-slate-400">{step.purpose || "—"}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-slate-400">No scale-out plan is active yet.</div>
          )}
        </div>

        {management?.management_notes?.length ? (
          <div className={`${SUBPANEL} p-4 text-sm text-slate-300`}>
            {management.management_notes.map((note, idx) => (
              <div key={`management-note-${idx}`} className="mb-2 last:mb-0">{note}</div>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function TriggerContextCard({ strategy }: { strategy?: Strategy }) {
  const trigger = strategy?.trigger_context;
  const validation = strategy?.entry_validation;
  return (
    <Card className={PANEL}>
      <CardHeader>
        <CardTitle className="text-2xl text-white">Trigger Context</CardTitle>
        <p className="text-sm text-slate-400">The deeper engine logic stays here when you want to inspect why the trade is or is not active.</p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Metric label="Trigger State" value={prettyLabel(trigger?.state)} />
          <Metric label="Ladder Stage" value={prettyLabel(trigger?.ladder_stage)} />
          <Metric label="Confirmation" value={`${formatNumber(trigger?.confirmation_level, 0)} / 10`} />
          <Metric label="Band Break" value={prettyLabel(trigger?.trigger_band_break_method)} />
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          <div className={`${SUBPANEL} p-4`}>
            <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Trigger reasons</div>
            {trigger?.reasons?.length ? (
              <ul className="space-y-2 text-sm text-slate-300">
                {trigger.reasons.map((reason, idx) => (
                  <li key={`trigger-reason-${idx}`}>{reason}</li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-slate-400">No trigger reasons yet.</div>
            )}
          </div>
          <div className={`${SUBPANEL} p-4`}>
            <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Blocking reasons</div>
            {validation?.blocking_reasons?.length ? (
              <ul className="space-y-2 text-sm text-slate-300">
                {validation.blocking_reasons.map((reason, idx) => (
                  <li key={`blocking-reason-${idx}`}>{reason}</li>
                ))}
              </ul>
            ) : (
              <div className="text-sm text-slate-400">No active blockers.</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function DeepDiveMode({ data }: { data?: EngineResponse | null }) {
  const strategy = data?.consensus?.strategy;
  return (
    <div className="space-y-6">
      <Card className={PANEL}>
        <CardHeader>
          <CardTitle className="text-2xl text-white">Deep Dive</CardTitle>
          <p className="text-sm text-slate-400">All the engine internals stay here, away from the live decision surface.</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Metric label="Short-Term Bias" value={`${prettyLabel(data?.consensus?.short_term?.bias)} (${formatNumber(data?.consensus?.short_term?.average_score, 3)})`} />
            <Metric label="Higher-Timeframe Bias" value={`${prettyLabel(data?.consensus?.higher_timeframes?.bias)} (${formatNumber(data?.consensus?.higher_timeframes?.average_score, 3)})`} />
            <Metric label="Overall Bias" value={`${prettyLabel(data?.consensus?.overall?.bias)} (${formatNumber(data?.consensus?.overall?.average_score, 3)})`} />
            <Metric label="Market Phase" value={prettyLabel(strategy?.market_phase)} />
          </div>
          <div className={`${SUBPANEL} p-4 text-sm text-slate-300`}>{data?.consensus?.dashboard_summary || "Waiting for dashboard summary."}</div>
        </CardContent>
      </Card>

      <TriggerContextCard strategy={strategy} />

      <Card className={PANEL}>
        <CardHeader>
          <CardTitle className="text-2xl text-white">Timeframe Evidence</CardTitle>
          <p className="text-sm text-slate-400">Still available when you want the full technical read, but no longer cluttering the live screen.</p>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {TIMEFRAME_ORDER.map((tf) => {
              const item = data?.timeframes?.[tf];
              return (
                <div key={tf} className={`${SUBPANEL} p-4`}>
                  <div className="mb-3 flex items-center justify-between gap-2">
                    <div className="text-base font-semibold text-white">{tf}</div>
                    <Badge className={`${PILL_BASE} ${biasTone(item?.signal?.includes("bull") ? "bullish" : item?.signal?.includes("bear") ? "bearish" : "neutral")}`}>
                      {prettyLabel(item?.signal)}
                    </Badge>
                  </div>
                  <div className="space-y-2 text-sm text-slate-300">
                    <div><span className="text-slate-400">Price:</span> {formatNumber(item?.price, 4)}</div>
                    <div><span className="text-slate-400">Setup:</span> {prettyLabel(item?.setup)}</div>
                    <div><span className="text-slate-400">Structure:</span> {prettyLabel(item?.structure)}</div>
                    <div><span className="text-slate-400">Trap:</span> {prettyLabel(item?.trap_risk)}</div>
                    <div><span className="text-slate-400">RSI:</span> {formatNumber(item?.indicators?.rsi_14, 2)}</div>
                    <div><span className="text-slate-400">ATR:</span> {formatNumber(item?.indicators?.atr_14, 4)}</div>
                  </div>
                  {item?.summary ? <p className="mt-3 text-sm text-slate-400">{item.summary}</p> : null}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function XrpDashboard() {
  const [data, setData] = useState<EngineResponse | null>(null);
  const [trackerSummary, setTrackerSummary] = useState<TradeTrackerSummary | undefined>();
  const [trackerTrades, setTrackerTrades] = useState<TradeRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastFetch, setLastFetch] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState(DEFAULT_MODE);
  const [viewMode, setViewMode] = useState<ViewMode>("live");
  const loadInFlight = useRef(false);

  const loadData = useCallback(async () => {
    if (loadInFlight.current) return;
    loadInFlight.current = true;
    try {
      setLoading(true);
      setError("");

      const engineUrl = `${API_BASE}/engine/multi-state?mode=${selectedMode}`;
      const engineRes = await fetch(engineUrl, { cache: "no-store" });
      if (!engineRes.ok) {
        throw new Error(`Request failed with status ${engineRes.status}`);
      }

      const engineJson: EngineResponse = await engineRes.json();
      setData(engineJson);
      setLastFetch(new Date().toISOString());

      if (viewMode === "review") {
        const summaryUrl = `${API_BASE}/engine/trades/summary?mode=${selectedMode}&scope=canonical`;
        const tradesUrl = `${API_BASE}/engine/trades?mode=${selectedMode}&scope=canonical&limit=8`;
        const [summaryRes, tradesRes] = await Promise.allSettled([
          fetch(summaryUrl, { cache: "no-store" }),
          fetch(tradesUrl, { cache: "no-store" }),
        ]);

        if (summaryRes.status === "fulfilled" && summaryRes.value.ok) {
        const summaryJson: TradeTrackerSummaryResponse = await summaryRes.value.json();
        setTrackerSummary(summaryJson.summary);
        } else {
          setTrackerSummary(undefined);
        }

        if (tradesRes.status === "fulfilled" && tradesRes.value.ok) {
        const tradesJson: TradeTrackerResponse = await tradesRes.value.json();
        setTrackerTrades(tradesJson.trades || []);
        } else {
          setTrackerTrades([]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load engine data.");
    } finally {
      setLoading(false);
      loadInFlight.current = false;
    }
  }, [selectedMode, viewMode]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, REFRESH_MS);
    return () => clearInterval(interval);
  }, [loadData]);

  const strategy = data?.consensus?.strategy;
  const execution = strategy?.execution_plan;
  const executionReview = strategy?.execution_review || execution?.execution_review;
  const limitOrderPlan = strategy?.limit_order_plan || execution?.limit_order_plan;
  const currentPrice = useMemo(() => {
    const tf5 = data?.timeframes?.["5m"]?.price;
    const tf15 = data?.timeframes?.["15m"]?.price;
    const tf1h = data?.timeframes?.["1h"]?.price;
    return firstNonEmpty(tf5, tf15, tf1h);
  }, [data]);

  const modeOptions = data?.mode?.available || ["conservative", "balanced", "aggressive"];
  const pageBackground = useMemo(
    () => pageBackgroundForState(data?.consensus?.overall?.bias, limitOrderPlan),
    [data?.consensus?.overall?.bias, limitOrderPlan]
  );

  return (
    <div className="min-h-screen px-4 py-5 transition-[background] duration-700 md:px-8 md:py-8" style={{ background: pageBackground }}>
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-white md:text-5xl">XRP Engine Dashboard</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300 md:text-base">
              Simplified v36 trading workstation: live decision, limit-order planner, execution quality, trade management, review, and deep-dive layers.
            </p>
          </div>
          <div className="flex flex-col gap-3 xl:items-end">
            <div className="flex flex-wrap items-center gap-3">
              <Badge className={`${PILL_BASE} border-slate-700 bg-slate-900/80 text-slate-200`}>
                <Clock3 className="mr-2 h-3.5 w-3.5" />
                Last fetch: {lastFetch ? formatTime(lastFetch) : "Waiting"}
              </Badge>
              <Badge className={`${PILL_BASE} border-cyan-500/25 bg-cyan-500/10 text-cyan-300`}>
                Version: {data?.engine_version || "—"}
              </Badge>
              <button
                onClick={loadData}
                className="inline-flex items-center rounded-2xl border border-slate-700 bg-slate-900/85 px-4 py-2.5 text-sm font-semibold text-slate-100 shadow-[0_10px_24px_rgba(2,6,23,0.45)] transition hover:border-cyan-500/30 hover:bg-slate-800/90"
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin text-cyan-300" : "text-slate-300"}`} />
                Refresh
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {modeOptions.map((mode) => (
                <button
                  key={mode}
                  onClick={() => setSelectedMode(mode)}
                  className={`rounded-2xl border px-3 py-2 text-sm font-semibold transition ${
                    selectedMode === mode
                      ? "border-cyan-400/50 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700 bg-slate-900/70 text-slate-300 hover:border-slate-500"
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className={`${PANEL} p-4`}>
          <div className="flex flex-wrap items-center gap-2">
            {VIEW_OPTIONS.map((view) => (
              <button
                key={view}
                onClick={() => setViewMode(view)}
                className={`rounded-2xl border px-4 py-2 text-sm font-semibold transition ${
                  viewMode === view
                    ? "border-cyan-400/50 bg-cyan-500/15 text-cyan-200"
                    : "border-slate-700 bg-slate-900/70 text-slate-300 hover:border-slate-500"
                }`}
              >
                {view === "live" ? "Live" : view === "review" ? "Review" : "Deep Dive"}
              </button>
            ))}
          </div>
        </div>

        {error ? (
          <Alert className="border-red-500/30 bg-red-950/30 text-red-200">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}

        {viewMode === "live" ? (
          <>
            <SectionHeading icon={Zap} title="Live" subtitle="Direction, timing, entry quality, and the price plan — that is all you need to act." />
            <DecisionBar strategy={strategy} execution={execution} review={executionReview} />
            <ExecutionVisualCard
              data={data}
              strategy={strategy}
              execution={execution}
              review={executionReview}
              plan={limitOrderPlan}
              currentPrice={currentPrice}
            />
            <div className="grid gap-6 xl:grid-cols-2">
              <PricePlanCard currentPrice={currentPrice} strategy={strategy} execution={execution} review={executionReview} />
              <EntryQualityCard review={executionReview} />
            </div>
            <LimitOrderPlannerCard plan={limitOrderPlan} />
            <ManagementCard management={execution?.execution_management} />
            <Card className={PANEL}>
              <CardHeader>
                <CardTitle className="text-2xl text-white">Quick Read</CardTitle>
                <p className="text-sm text-slate-400">The live layer is now reduced to one summary and one backup instruction.</p>
              </CardHeader>
              <CardContent className="grid gap-4 xl:grid-cols-2">
                <div className={`${SUBPANEL} p-4`}>
                  <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Plain-English read</div>
                  <p className="text-base leading-7 text-slate-200">{strategy?.summary || "Waiting for a clear read."}</p>
                </div>
                <div className={`${SUBPANEL} p-4`}>
                  <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Playbook</div>
                  {execution?.playbook?.length ? (
                    <ul className="space-y-2 text-sm text-slate-300">
                      {execution.playbook.slice(0, 3).map((step, idx) => (
                        <li key={`playbook-${idx}`}>{step}</li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-sm text-slate-400">No live playbook yet.</div>
                  )}
                </div>
              </CardContent>
            </Card>
          </>
        ) : null}

        {viewMode === "review" ? (
          <>
            <SectionHeading icon={BarChart3} title="Review" subtitle="Use this to understand how the engine has actually been performing, not to make the live decision." />
            <ReviewMode summary={trackerSummary} trades={trackerTrades} />
          </>
        ) : null}

        {viewMode === "deep" ? (
          <>
            <SectionHeading icon={Layers3} title="Deep Dive" subtitle="All the technical context stays here when you want to inspect the engine more deeply." />
            <DeepDiveMode data={data} />
          </>
        ) : null}
      </div>
    </div>
  );
}

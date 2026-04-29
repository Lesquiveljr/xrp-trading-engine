"use client";

import React from "react";

type Props = {
  values: number[];
  score?: number;
};

export default function MiniSparkline({ values, score }: Props) {
  if (!values || values.length < 2) return null;

  const width = 240;
  const height = 60;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const mappedPoints = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return { x, y };
  });

  const points = mappedPoints.map((p) => `${p.x},${p.y}`).join(" ");
  const lastPoint = mappedPoints[mappedPoints.length - 1];

  let color = "#64748b";
  if ((score ?? 0) > 0) color = "#16a34a";
  if ((score ?? 0) < 0) color = "#dc2626";

  return (
    <svg width={width} height={height} className="mt-2 opacity-90">

      {/* grid lines */}
      <line x1="0" x2={width} y1="10" y2="10" stroke="#f1f5f9" strokeWidth="1" />
      <line x1="0" x2={width} y1={height/2} y2={height/2} stroke="#e2e8f0" strokeWidth="1" />
      <line x1="0" x2={width} y1={height-10} y2={height-10} stroke="#f1f5f9" strokeWidth="1" />

      {/* sparkline */}
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />

      {/* last price dot */}
      <circle cx={lastPoint.x} cy={lastPoint.y} r={3.5} fill={color} />

    </svg>
  );
}    


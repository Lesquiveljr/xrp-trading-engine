# XRP Engine Dashboard

XRP Engine Dashboard is a live XRP-USD trading-analysis workstation built with a FastAPI strategy engine and a Next.js dashboard.

The engine separates short-term scalp execution from longer-horizon directional reads, generates limit/retest plans, tracks execution quality, and logs market snapshots for later analysis. The dashboard focuses the live view around a simple question: act, wait, or stand aside.

## Stack

- Backend: FastAPI, pandas, NumPy, Coinbase Exchange candles
- Frontend: Next.js, React, Tailwind CSS
- Analysis: JSONL snapshot logs plus `scripts/analyze_engine_logs.py`

## Local Development

Start the backend:

```bash
python -m uvicorn app_v36_limit_order_planner:app --host 127.0.0.1 --port 8000
```

Start the dashboard:

```bash
cd xrp-dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:3000`.

## Deployment

Deploy the backend to Render using `render.yaml`.

Set this Render environment variable after the Vercel deployment exists:

```text
FRONTEND_ORIGINS=https://your-vercel-project.vercel.app
```

Deploy the frontend to Vercel from the `xrp-dashboard` directory.

Set this Vercel environment variable:

```text
NEXT_PUBLIC_ENGINE_URL=https://your-render-service.onrender.com
```

## Notes

This project is for educational and showcase purposes. It is not financial advice and does not place trades.

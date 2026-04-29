import { NextRequest } from "next/server";

const ENGINE_URL =
  process.env.ENGINE_URL || process.env.NEXT_PUBLIC_ENGINE_URL || "http://127.0.0.1:8000";

type RouteContext = {
  params: Promise<{
    path?: string[];
  }>;
};

export async function GET(request: NextRequest, context: RouteContext) {
  const { path = [] } = await context.params;
  const upstreamUrl = new URL(`/engine/${path.join("/")}`, ENGINE_URL);
  upstreamUrl.search = request.nextUrl.search;

  try {
    const upstream = await fetch(upstreamUrl, {
      cache: "no-store",
      headers: {
        accept: request.headers.get("accept") || "application/json",
      },
    });

    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") || "application/json",
      },
    });
  } catch (error) {
    return Response.json(
      {
        error: "Unable to reach engine API.",
        detail: error instanceof Error ? error.message : "Unknown proxy error",
      },
      { status: 502 },
    );
  }
}

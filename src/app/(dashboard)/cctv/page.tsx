"use client";

import { useQuery } from "@tanstack/react-query";

const VISION = process.env.NEXT_PUBLIC_VISION_URL ?? "http://localhost:8000";

type Camera = { id: string; online: boolean };

export default function CctvPage() {
  const { data: cams = [], isLoading, isError } = useQuery<Camera[]>({
    queryKey: ["cameras"],
    queryFn: async () => {
      const res = await fetch(`${VISION}/cameras`);
      if (!res.ok) throw new Error("Vision service unreachable");
      return res.json();
    },
    refetchInterval: 5000,
  });

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">CCTV Monitoring</h1>
        <p className="text-sm text-muted-foreground">Live feeds from connected cameras</p>
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">Connecting to cameras…</p>}
      {isError && (
        <p className="text-sm text-red-500">
          Vision service is not reachable. Make sure the Docker container is running.
        </p>
      )}
      {!isLoading && !isError && cams.length === 0 && (
        <p className="text-sm text-muted-foreground">No cameras configured.</p>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {cams.map((c) => (
          <div key={c.id} className="overflow-hidden rounded-xl border bg-card">
            <div className="flex items-center justify-between px-3 py-2 text-sm">
              <span className="font-medium">{c.id}</span>
              <span className={c.online ? "text-green-500" : "text-red-500"}>
                ● {c.online ? "Live" : "Offline"}
              </span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${VISION}/stream/${c.id}`}
              alt={`Camera ${c.id}`}
              className="aspect-video w-full bg-black object-cover"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

"use client";

import { useQuery } from "@tanstack/react-query";

const VISION = process.env.NEXT_PUBLIC_VISION_URL ?? "http://localhost:8000";

type Camera = { id: string; online: boolean; persons?: number; fps?: number };

export default function CctvPage() {
  const { data: cams = [], isLoading, isError } = useQuery<Camera[]>({
    queryKey: ["cameras"],
    queryFn: async () => {
      const res = await fetch(`${VISION}/cameras`);
      if (!res.ok) throw new Error("Vision service unreachable");
      return res.json();
    },
    refetchInterval: 2000,
  });

  const totalPersons = cams.reduce((sum, c) => sum + (c.persons ?? 0), 0);

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">CCTV Monitoring</h1>
          <p className="text-sm text-muted-foreground">Live feeds with AI person detection</p>
        </div>
        {cams.length > 0 && (
          <div className="flex gap-3 text-sm">
            <span className="rounded-lg border bg-card px-3 py-1.5">
              Cameras online: <b>{cams.filter((c) => c.online).length}/{cams.length}</b>
            </span>
            <span className="rounded-lg border bg-card px-3 py-1.5">
              People detected: <b>{totalPersons}</b>
            </span>
          </div>
        )}
      </div>

      {isLoading && <p className="text-sm text-muted-foreground">Connecting to cameras…</p>}
      {isError && (
        <p className="text-sm text-red-500">
          Vision service is not reachable. Make sure it is running on port 8000.
        </p>
      )}
      {!isLoading && !isError && cams.length === 0 && (
        <p className="text-sm text-muted-foreground">No cameras configured.</p>
      )}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {cams.map((c) => (
          <div key={c.id} className="overflow-hidden rounded-xl border bg-card">
            <div className="flex items-center justify-between gap-2 px-3 py-2 text-sm">
              <span className="font-medium">{c.id}</span>
              <div className="flex items-center gap-3">
                {c.online && typeof c.persons === "number" && (
                  <span className="text-muted-foreground">
                    👤 {c.persons} {c.persons === 1 ? "person" : "persons"}
                  </span>
                )}
                <span className={c.online ? "text-green-500" : "text-red-500"}>
                  ● {c.online ? "Live" : "Offline"}
                </span>
              </div>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${VISION}/stream/${c.id}`}
              alt={`Camera ${c.id}`}
              className="aspect-video w-full bg-black object-contain"
            />
            {c.online && typeof c.fps === "number" && (
              <div className="px-3 py-1.5 text-xs text-muted-foreground">
                Detection speed: {c.fps} FPS
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

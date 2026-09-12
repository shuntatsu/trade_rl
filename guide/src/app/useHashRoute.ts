import { useCallback, useEffect, useMemo, useState } from "react";

export type GuideRoute = {
  topicId: string;
  step?: string;
  symbol?: string;
};

export type GuideNavigateTarget = GuideRoute | string;

function normalizedSelection(value: string | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

export function normalizeHashRoute(
  hash: string,
  validIds: readonly string[],
  home: string,
): GuideRoute {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const queryIndex = raw.indexOf("?");
  const rawTopic = queryIndex >= 0 ? raw.slice(0, queryIndex) : raw;
  const rawQuery = queryIndex >= 0 ? raw.slice(queryIndex + 1) : "";

  let topicId: string;
  try {
    topicId = decodeURIComponent(rawTopic);
  } catch {
    return { topicId: home };
  }
  if (!validIds.includes(topicId)) return { topicId: home };

  const params = new URLSearchParams(rawQuery);
  const step = normalizedSelection(params.get("step"));
  const symbol = normalizedSelection(params.get("symbol"));
  return {
    topicId,
    ...(step ? { step } : {}),
    ...(symbol ? { symbol } : {}),
  };
}

export function formatHashRoute(route: GuideRoute): string {
  const query = new URLSearchParams();
  if (route.step) query.set("step", route.step);
  if (route.symbol) query.set("symbol", route.symbol);
  const suffix = query.toString();
  return `#${encodeURIComponent(route.topicId)}${suffix ? `?${suffix}` : ""}`;
}

export function useHashRoute(
  validIds: readonly string[],
  home: string,
): [GuideRoute, (target: GuideNavigateTarget) => void] {
  const ids = useMemo(() => [...validIds], [validIds]);
  const read = useCallback(
    () => normalizeHashRoute(window.location.hash, ids, home),
    [home, ids],
  );
  const [route, setRoute] = useState<GuideRoute>(() =>
    typeof window === "undefined" ? { topicId: home } : read(),
  );

  useEffect(() => {
    const onHashChange = () => setRoute(read());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [read]);

  const navigate = useCallback(
    (target: GuideNavigateTarget) => {
      const requested = typeof target === "string" ? { topicId: target } : target;
      const next = ids.includes(requested.topicId)
        ? requested
        : { topicId: home };
      const nextHash = formatHashRoute(next);
      if (window.location.hash === nextHash) {
        setRoute(next);
        return;
      }
      window.location.hash = nextHash;
    },
    [home, ids],
  );

  return [route, navigate];
}

import { useCallback, useEffect, useMemo, useState } from "react";

export type GuideRoute = {
  topicId: string;
  heading?: string;
  symbol?: string;
};

export type GuideNavigateTarget = GuideRoute | string;

const LEGACY_STEP_HEADINGS: Record<string, Record<string, string>> = {
  "implementation-replay": {
    observe: "1-観測を作る",
    decide: "2-戦略がintentを返す",
    "intent-target": "3-希望数量とproposalへ変換する",
    "risk-constrain": "4-hard-riskを適用する",
    "execute-interval": "5-約定と会計を行う",
    "update-book": "6-bookstateを引き継ぐ",
    advance: "6-bookstateを引き継ぐ",
    finalize: "7-結果を確定する",
  },
  "implementation-ppo": {
    "encode-observation": "1-観測を符号化する",
    "policy-action": "2-policyがactionを選ぶ",
    "env-intent": "3-actionを売買意図へ変換する",
    "env-risk": "4-hard-riskを通す",
    "env-execute": "5-約定-会計を通す",
    "env-reward": "6-net-returnからrewardを作る",
    "fit-strategy": "学習後も同じ観測契約を使う",
    "runtime-decide": "学習後も同じ観測契約を使う",
  },
};

function normalizedSelection(value: string | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function normalizedRoute(route: GuideRoute): GuideRoute {
  if (route.heading) return { topicId: route.topicId, heading: route.heading };
  if (route.symbol) return { topicId: route.topicId, symbol: route.symbol };
  return { topicId: route.topicId };
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
  const heading = normalizedSelection(params.get("heading"));
  if (heading) return { topicId, heading };
  const symbol = normalizedSelection(params.get("symbol"));
  if (symbol) return { topicId, symbol };

  const legacyStep = normalizedSelection(params.get("step"));
  const mapped = legacyStep ? LEGACY_STEP_HEADINGS[topicId]?.[legacyStep] : undefined;
  return mapped ? { topicId, heading: mapped } : { topicId };
}

export function formatHashRoute(route: GuideRoute): string {
  const normalized = normalizedRoute(route);
  const query = new URLSearchParams();
  if (normalized.heading) query.set("heading", normalized.heading);
  else if (normalized.symbol) query.set("symbol", normalized.symbol);
  const suffix = query.toString();
  return `#${encodeURIComponent(normalized.topicId)}${suffix ? `?${suffix}` : ""}`;
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
        ? normalizedRoute(requested)
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

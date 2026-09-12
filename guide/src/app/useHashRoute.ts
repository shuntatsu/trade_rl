import { useCallback, useEffect, useMemo, useState } from "react";

export function normalizeHashRoute(
  hash: string,
  validIds: readonly string[],
  home: string,
): string {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  let decoded = raw;
  try {
    decoded = decodeURIComponent(raw);
  } catch {
    return home;
  }
  return validIds.includes(decoded) ? decoded : home;
}

export function useHashRoute(
  validIds: readonly string[],
  home: string,
): [string, (id: string) => void] {
  const ids = useMemo(() => [...validIds], [validIds]);
  const read = useCallback(
    () => normalizeHashRoute(window.location.hash, ids, home),
    [home, ids],
  );
  const [route, setRoute] = useState<string>(() =>
    typeof window === "undefined" ? home : read(),
  );

  useEffect(() => {
    const onHashChange = () => setRoute(read());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [read]);

  const navigate = useCallback(
    (id: string) => {
      const next = ids.includes(id) ? id : home;
      if (window.location.hash === `#${encodeURIComponent(next)}`) {
        setRoute(next);
        return;
      }
      window.location.hash = next;
    },
    [home, ids],
  );

  return [route, navigate];
}

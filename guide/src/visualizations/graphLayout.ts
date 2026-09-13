export type DirectedEdge = {
  from: string;
  to: string;
};

export type GraphLayout = {
  layers: string[][];
};

export function layerDirectedGraph(
  nodeIds: readonly string[],
  edges: readonly DirectedEdge[],
): GraphLayout {
  const order = new Map(nodeIds.map((id, index) => [id, index]));
  if (order.size !== nodeIds.length) {
    throw new Error("directed graph contains duplicate node ids");
  }

  const indegree = new Map(nodeIds.map((id) => [id, 0]));
  const outgoing = new Map(nodeIds.map((id) => [id, [] as string[]]));

  for (const edge of edges) {
    if (!indegree.has(edge.from) || !indegree.has(edge.to)) {
      throw new Error(`directed graph edge references unknown node: ${edge.from} -> ${edge.to}`);
    }
    outgoing.get(edge.from)?.push(edge.to);
    indegree.set(edge.to, (indegree.get(edge.to) ?? 0) + 1);
  }

  let current = nodeIds.filter((id) => indegree.get(id) === 0);
  const layers: string[][] = [];
  let visited = 0;

  while (current.length > 0) {
    const layer = [...current].sort(
      (left, right) => (order.get(left) ?? 0) - (order.get(right) ?? 0),
    );
    layers.push(layer);
    visited += layer.length;

    const next = new Set<string>();
    for (const id of layer) {
      for (const target of outgoing.get(id) ?? []) {
        const remaining = (indegree.get(target) ?? 0) - 1;
        indegree.set(target, remaining);
        if (remaining === 0) next.add(target);
      }
    }
    current = [...next];
  }

  if (visited !== nodeIds.length) {
    throw new Error("directed graph contains a cycle");
  }

  return { layers };
}

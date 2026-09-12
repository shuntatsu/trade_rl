export type SourceSection = {
  path: string;
  heading: string;
  sha256: string;
};

export type GuideSection = {
  title: string;
  body: string[];
};

export type ArchitectureNode = {
  id: string;
  label: string;
  subtitle: string;
  description: string;
  input: string;
  output: string;
  not_owned: string;
};

export type ArchitectureVisualization = {
  kind: "architecture";
  nodes: ArchitectureNode[];
  edges: Array<{ from: string; to: string }>;
};

export type FlowStep = {
  id: string;
  label: string;
  summary?: string;
  detail: string;
  phase?: string;
};

export type DataFlowVisualization = {
  kind: "data-flow";
  steps: FlowStep[];
};

export type EconomicsVisualization = {
  kind: "economics";
  components: Array<{
    id: string;
    label: string;
    role: string;
    description: string;
    status: string;
  }>;
  assumptions: Array<{ label: string; state: string }>;
};

export type ObservationVisualization = {
  kind: "observation";
  segments: Array<{
    id: string;
    label: string;
    description: string;
    weight: number;
  }>;
  excluded: string[];
};

export type ExperimentLoopVisualization = {
  kind: "experiment-loop";
  steps: FlowStep[];
  failure_states: string[];
};

export type ResearchStatusVisualization = {
  kind: "research-status";
  groups: Array<{
    id: string;
    label: string;
    items: string[];
  }>;
};

export type GuideVisualization =
  | ArchitectureVisualization
  | DataFlowVisualization
  | EconomicsVisualization
  | ObservationVisualization
  | ExperimentLoopVisualization
  | ResearchStatusVisualization;

export type GuideTopic = {
  id: string;
  title: string;
  nav_label: string;
  summary: string;
  keywords: string[];
  source_sections: SourceSection[];
  sections: GuideSection[];
  visualization: GuideVisualization;
};

export type GuideManifest = {
  schema_version: "interactive-guide-v1";
  home: string;
  topics: string[];
};

type JsonRecord = Record<string, unknown>;

function record(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function string(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function strings(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new Error(`${label} must be a string array`);
  }
  return [...value];
}

function uniqueIds(items: JsonRecord[], label: string): Set<string> {
  const ids = items.map((item, index) => string(item.id, `${label}[${index}].id`));
  if (new Set(ids).size !== ids.length) {
    throw new Error(`${label} contains duplicate ids`);
  }
  return new Set(ids);
}

function parseSources(value: unknown): SourceSection[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("source_sections must be a non-empty array");
  }
  return value.map((item, index) => {
    const source = record(item, `source_sections[${index}]`);
    const sha256 = string(source.sha256, `source_sections[${index}].sha256`);
    if (!/^[0-9a-f]{64}$/.test(sha256)) {
      throw new Error(`source_sections[${index}].sha256 must be SHA-256`);
    }
    return {
      path: string(source.path, `source_sections[${index}].path`),
      heading: string(source.heading, `source_sections[${index}].heading`),
      sha256,
    };
  });
}

function parseSections(value: unknown): GuideSection[] {
  if (!Array.isArray(value)) {
    throw new Error("sections must be an array");
  }
  return value.map((item, index) => {
    const section = record(item, `sections[${index}]`);
    return {
      title: string(section.title, `sections[${index}].title`),
      body: strings(section.body, `sections[${index}].body`),
    };
  });
}

function parseArchitecture(raw: JsonRecord): ArchitectureVisualization {
  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) {
    throw new Error("architecture visualization requires nodes and edges");
  }
  const nodeRecords = raw.nodes.map((item, index) => record(item, `nodes[${index}]`));
  const nodeIds = uniqueIds(nodeRecords, "nodes");
  const nodes = nodeRecords.map((node, index) => ({
    id: string(node.id, `nodes[${index}].id`),
    label: string(node.label, `nodes[${index}].label`),
    subtitle: string(node.subtitle, `nodes[${index}].subtitle`),
    description: string(node.description, `nodes[${index}].description`),
    input: string(node.input, `nodes[${index}].input`),
    output: string(node.output, `nodes[${index}].output`),
    not_owned: string(node.not_owned, `nodes[${index}].not_owned`),
  }));
  const edges = raw.edges.map((item, index) => {
    const edge = record(item, `edges[${index}]`);
    const from = string(edge.from, `edges[${index}].from`);
    const to = string(edge.to, `edges[${index}].to`);
    if (!nodeIds.has(from) || !nodeIds.has(to)) {
      throw new Error(`edge reference is unknown: ${from} -> ${to}`);
    }
    return { from, to };
  });
  return { kind: "architecture", nodes, edges };
}

function parseSteps(value: unknown, label: string): FlowStep[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array`);
  }
  const records = value.map((item, index) => record(item, `${label}[${index}]`));
  uniqueIds(records, label);
  return records.map((item, index) => ({
    id: string(item.id, `${label}[${index}].id`),
    label: string(item.label, `${label}[${index}].label`),
    summary: typeof item.summary === "string" ? item.summary : undefined,
    detail: string(item.detail, `${label}[${index}].detail`),
    phase: typeof item.phase === "string" ? item.phase : undefined,
  }));
}

function parseVisualization(value: unknown): GuideVisualization {
  const raw = record(value, "visualization");
  const kind = string(raw.kind, "visualization.kind");
  if (kind === "architecture") return parseArchitecture(raw);
  if (kind === "data-flow") {
    return { kind, steps: parseSteps(raw.steps, "steps") };
  }
  if (kind === "economics") {
    if (!Array.isArray(raw.components) || !Array.isArray(raw.assumptions)) {
      throw new Error("economics visualization requires components and assumptions");
    }
    const components = raw.components.map((item, index) => {
      const component = record(item, `components[${index}]`);
      return {
        id: string(component.id, `components[${index}].id`),
        label: string(component.label, `components[${index}].label`),
        role: string(component.role, `components[${index}].role`),
        description: string(component.description, `components[${index}].description`),
        status: string(component.status, `components[${index}].status`),
      };
    });
    uniqueIds(components, "components");
    const assumptions = raw.assumptions.map((item, index) => {
      const assumption = record(item, `assumptions[${index}]`);
      return {
        label: string(assumption.label, `assumptions[${index}].label`),
        state: string(assumption.state, `assumptions[${index}].state`),
      };
    });
    return { kind, components, assumptions };
  }
  if (kind === "observation") {
    if (!Array.isArray(raw.segments)) throw new Error("observation requires segments");
    const segments = raw.segments.map((item, index) => {
      const segment = record(item, `segments[${index}]`);
      const weight = segment.weight;
      if (typeof weight !== "number" || !Number.isFinite(weight) || weight <= 0) {
        throw new Error(`segments[${index}].weight must be positive`);
      }
      return {
        id: string(segment.id, `segments[${index}].id`),
        label: string(segment.label, `segments[${index}].label`),
        description: string(segment.description, `segments[${index}].description`),
        weight,
      };
    });
    uniqueIds(segments, "segments");
    return { kind, segments, excluded: strings(raw.excluded, "excluded") };
  }
  if (kind === "experiment-loop") {
    return {
      kind,
      steps: parseSteps(raw.steps, "steps"),
      failure_states: strings(raw.failure_states, "failure_states"),
    };
  }
  if (kind === "research-status") {
    if (!Array.isArray(raw.groups)) throw new Error("research-status requires groups");
    const groups = raw.groups.map((item, index) => {
      const group = record(item, `groups[${index}]`);
      return {
        id: string(group.id, `groups[${index}].id`),
        label: string(group.label, `groups[${index}].label`),
        items: strings(group.items, `groups[${index}].items`),
      };
    });
    uniqueIds(groups, "groups");
    return { kind, groups };
  }
  throw new Error(`unknown visualization kind: ${kind}`);
}

export function parseTopic(value: unknown): GuideTopic {
  const raw = record(value, "topic");
  return {
    id: string(raw.id, "topic.id"),
    title: string(raw.title, "topic.title"),
    nav_label: string(raw.nav_label, "topic.nav_label"),
    summary: string(raw.summary, "topic.summary"),
    keywords: strings(raw.keywords, "topic.keywords"),
    source_sections: parseSources(raw.source_sections),
    sections: parseSections(raw.sections),
    visualization: parseVisualization(raw.visualization),
  };
}

export function parseGuideManifest(value: unknown): GuideManifest {
  const raw = record(value, "manifest");
  const schemaVersion = string(raw.schema_version, "manifest.schema_version");
  if (schemaVersion !== "interactive-guide-v1") {
    throw new Error(`unsupported manifest schema: ${schemaVersion}`);
  }
  const topics = strings(raw.topics, "manifest.topics");
  if (topics.length === 0 || new Set(topics).size !== topics.length) {
    throw new Error("manifest topics must be non-empty and unique");
  }
  const home = string(raw.home, "manifest.home");
  if (!topics.includes(home)) throw new Error("manifest home must reference a topic");
  return { schema_version: "interactive-guide-v1", home, topics };
}

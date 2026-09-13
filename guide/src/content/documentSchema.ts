import { extractMarkdownHeadings, type MarkdownHeading } from "./markdown";

export type SourceSection = {
  path: string;
  heading: string;
  sha256: string;
};

export type CodeVariableReference = {
  name: string;
  label_ja: string;
  description_ja: string;
};

export type CodeReference = {
  id: string;
  symbol: string;
  kind: "function" | "class" | "method";
  source_sha256: string;
  label_ja: string;
  description_ja: string;
  variables: CodeVariableReference[];
  tests: string[];
};

export type GuidePageRole = "overview" | "detail" | "status" | "reference";

export type DocumentGuideTopic = {
  id: string;
  title: string;
  nav_label: string;
  summary: string;
  role: GuidePageRole;
  keywords: string[];
  source_sections: SourceSection[];
  code_references: CodeReference[];
  markdown: string;
  headings: MarkdownHeading[];
};

export type GuideManifestGroup = {
  id: string;
  label: string;
  topics: string[];
};

export type DocumentGuideManifest = {
  schema_version: "document-guide-v2";
  home: string;
  groups: GuideManifestGroup[];
  reading_order: string[];
};

type JsonRecord = Record<string, unknown>;

function record(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value.trim();
}

function textArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new Error(`${label} must be a string array`);
  }
  return value.map((item) => item.trim());
}

function sha256(value: unknown, label: string): string {
  const digest = text(value, label);
  if (!/^[0-9a-f]{64}$/.test(digest)) throw new Error(`${label} must be SHA-256`);
  return digest;
}

function unique(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`${label} must be unique`);
  }
}

function parseSources(value: unknown): SourceSection[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("source_sections must be a non-empty array");
  }
  return value.map((item, index) => {
    const source = record(item, `source_sections[${index}]`);
    return {
      path: text(source.path, `source_sections[${index}].path`),
      heading: text(source.heading, `source_sections[${index}].heading`),
      sha256: sha256(source.sha256, `source_sections[${index}].sha256`),
    };
  });
}

function parseCodeReferences(value: unknown): CodeReference[] {
  if (!Array.isArray(value)) throw new Error("code_references must be an array");
  const raw = value.map((item, index) => record(item, `code_references[${index}]`));
  const ids = raw.map((item, index) => text(item.id, `code_references[${index}].id`));
  unique(ids, "code reference ids");

  return raw.map((reference, index) => {
    const kind = text(reference.kind, `code_references[${index}].kind`);
    if (kind !== "function" && kind !== "class" && kind !== "method") {
      throw new Error(`code_references[${index}].kind is invalid`);
    }
    const rawVariables = reference.variables ?? [];
    if (!Array.isArray(rawVariables)) {
      throw new Error(`code_references[${index}].variables must be an array`);
    }
    const variables = rawVariables.map((item, variableIndex) => {
      const variable = record(
        item,
        `code_references[${index}].variables[${variableIndex}]`,
      );
      return {
        name: text(
          variable.name,
          `code_references[${index}].variables[${variableIndex}].name`,
        ),
        label_ja: text(
          variable.label_ja,
          `code_references[${index}].variables[${variableIndex}].label_ja`,
        ),
        description_ja: text(
          variable.description_ja,
          `code_references[${index}].variables[${variableIndex}].description_ja`,
        ),
      };
    });
    unique(
      variables.map((variable) => variable.name),
      `code_references[${index}] variable names`,
    );
    return {
      id: ids[index],
      symbol: text(reference.symbol, `code_references[${index}].symbol`),
      kind,
      source_sha256: sha256(
        reference.source_sha256,
        `code_references[${index}].source_sha256`,
      ),
      label_ja: text(reference.label_ja, `code_references[${index}].label_ja`),
      description_ja: text(
        reference.description_ja,
        `code_references[${index}].description_ja`,
      ),
      variables,
      tests: textArray(reference.tests ?? [], `code_references[${index}].tests`),
    };
  });
}

function parseRole(value: unknown): GuidePageRole {
  const role = text(value, "topic.role");
  if (role !== "overview" && role !== "detail" && role !== "status" && role !== "reference") {
    throw new Error(`unsupported Guide page role: ${role}`);
  }
  return role;
}

export function parseDocumentTopic(value: unknown, markdown: string): DocumentGuideTopic {
  const raw = record(value, "topic metadata");
  const role = parseRole(raw.role);
  const codeReferences = parseCodeReferences(raw.code_references ?? []);
  if ((role === "overview" || role === "status") && codeReferences.length !== 0) {
    throw new Error(`${role} page must not expose code_references`);
  }
  if (role === "reference" && codeReferences.length === 0) {
    throw new Error("reference page requires code_references");
  }
  if (typeof markdown !== "string" || markdown.trim().length === 0) {
    throw new Error("Markdown page must be non-empty");
  }

  return {
    id: text(raw.id, "topic.id"),
    title: text(raw.title, "topic.title"),
    nav_label: text(raw.nav_label, "topic.nav_label"),
    summary: text(raw.summary, "topic.summary"),
    role,
    keywords: textArray(raw.keywords, "topic.keywords"),
    source_sections: parseSources(raw.source_sections),
    code_references: codeReferences,
    markdown,
    headings: extractMarkdownHeadings(markdown),
  };
}

export function parseDocumentGuideManifest(value: unknown): DocumentGuideManifest {
  const raw = record(value, "manifest");
  const schemaVersion = text(raw.schema_version, "manifest.schema_version");
  if (schemaVersion !== "document-guide-v2") {
    throw new Error(`unsupported manifest schema: ${schemaVersion}`);
  }
  if (!Array.isArray(raw.groups) || raw.groups.length === 0) {
    throw new Error("manifest groups must be a non-empty array");
  }
  const groups = raw.groups.map((item, index) => {
    const group = record(item, `manifest.groups[${index}]`);
    const topics = textArray(group.topics, `manifest.groups[${index}].topics`);
    if (topics.length === 0) throw new Error("manifest group topics must be non-empty");
    unique(topics, `manifest.groups[${index}].topics`);
    return {
      id: text(group.id, `manifest.groups[${index}].id`),
      label: text(group.label, `manifest.groups[${index}].label`),
      topics,
    };
  });
  unique(
    groups.map((group) => group.id),
    "manifest group ids",
  );
  const topics = groups.flatMap((group) => group.topics);
  unique(topics, "manifest grouped topics");

  const readingOrder = textArray(raw.reading_order, "manifest.reading_order");
  if (readingOrder.length === 0) throw new Error("manifest reading_order must be non-empty");
  unique(readingOrder, "manifest reading_order");
  for (const id of readingOrder) {
    if (!topics.includes(id)) throw new Error(`reading_order references unknown topic: ${id}`);
  }
  const home = text(raw.home, "manifest.home");
  if (!topics.includes(home)) throw new Error("manifest home must reference a topic");
  if (readingOrder[0] !== home) throw new Error("manifest reading_order must start at home");

  return {
    schema_version: "document-guide-v2",
    home,
    groups,
    reading_order: readingOrder,
  };
}

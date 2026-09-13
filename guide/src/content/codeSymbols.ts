import generatedRaw from "../../.generated/code-symbols-runtime.json";

export type CodeSymbol = {
  qualified_name: string;
  kind: "function" | "class" | "method";
  path: string;
  start_line: number;
  end_line: number;
  signature: string;
  source_sha256: string;
  local_names: string[];
};

export type CodeSymbolIndex = {
  schema_version: "guide-code-symbols-v1";
  source_revision: string;
  symbols: CodeSymbol[];
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

function integer(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return value;
}

function stringArray(value: unknown, label: string): string[] {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    throw new Error(`${label} must be a string array`);
  }
  return [...value];
}

function parseCodeSymbol(value: unknown, index: number): CodeSymbol {
  const raw = record(value, `symbols[${index}]`);
  const kind = string(raw.kind, `symbols[${index}].kind`);
  if (kind !== "function" && kind !== "class" && kind !== "method") {
    throw new Error(`symbols[${index}].kind is invalid`);
  }
  const sourceSha256 = string(raw.source_sha256, `symbols[${index}].source_sha256`);
  if (!/^[0-9a-f]{64}$/.test(sourceSha256)) {
    throw new Error(`symbols[${index}].source_sha256 must be SHA-256`);
  }
  return {
    qualified_name: string(raw.qualified_name, `symbols[${index}].qualified_name`),
    kind,
    path: string(raw.path, `symbols[${index}].path`),
    start_line: integer(raw.start_line, `symbols[${index}].start_line`),
    end_line: integer(raw.end_line, `symbols[${index}].end_line`),
    signature: string(raw.signature, `symbols[${index}].signature`),
    source_sha256: sourceSha256,
    local_names: stringArray(raw.local_names, `symbols[${index}].local_names`),
  };
}

function parseCodeSymbolIndex(value: unknown): CodeSymbolIndex {
  const raw = record(value, "code symbol index");
  if (raw.schema_version !== "guide-code-symbols-v1") {
    throw new Error(`unsupported code symbol schema: ${String(raw.schema_version)}`);
  }
  const sourceRevision = string(raw.source_revision, "source_revision");
  if (!/^[0-9a-f]{40}$/.test(sourceRevision)) {
    throw new Error("source_revision must be a 40-character commit SHA");
  }
  if (!Array.isArray(raw.symbols)) {
    throw new Error("symbols must be an array");
  }
  const symbols = raw.symbols.map(parseCodeSymbol);
  const names = symbols.map((symbol) => symbol.qualified_name);
  if (new Set(names).size !== names.length) {
    throw new Error("code symbol index contains duplicate qualified names");
  }
  return {
    schema_version: "guide-code-symbols-v1",
    source_revision: sourceRevision,
    symbols,
  };
}

export const codeSymbolIndex = parseCodeSymbolIndex(generatedRaw);

const symbolsByName = new Map(
  codeSymbolIndex.symbols.map((symbol) => [symbol.qualified_name, symbol]),
);

export function getCodeSymbol(qualifiedName: string): CodeSymbol | undefined {
  return symbolsByName.get(qualifiedName);
}

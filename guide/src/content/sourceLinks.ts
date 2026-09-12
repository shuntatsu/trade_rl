import { codeSymbolIndex, type CodeSymbol } from "./codeSymbols";

export function buildCodeSourceUrl(symbol: CodeSymbol, repositoryUrl: string): string {
  const repository = repositoryUrl.replace(/\/$/, "");
  return `${repository}/blob/${codeSymbolIndex.source_revision}/${symbol.path}#L${symbol.start_line}-L${symbol.end_line}`;
}

from pathlib import Path

path = Path("docs/AGENTS.md")
text = path.read_text(encoding="utf-8")
start = text.index("## Interactive Guide update contract\n")
end = text.index("## Local repository tooling\n")
replacement = '''## Human Guide update contract

Root `guide/` は人間向けの**非正本**説明層であり、技術仕様・研究状態のauthorityにはしない。画面ではMarkdown本文・静的flow・表を主役にし、実identifier・型・file pathは記事末尾の実装参照で副表示する。人間向け本文は `guide/content/pages/*.md`、machine metadata / source fingerprint / CodeReferenceは `guide/content/meta/*.json` に置き、UI componentへ研究固有の真実を直接埋め込まない。

Guide manifestは `document-guide-v2` とし、navigation groupとdefault `reading_order`を明示する。`overview` / `status` は `code_references=[]` を必須にし、`reference` はcode referenceを1件以上持つ。replay/PPOの主要処理順は初期DOMのMarkdownだけで読める状態を維持し、interactive step selectionを理解の前提にしない。MDX、raw HTML実行、custom Markdown UI directiveを導入しない。

正本Markdownとの対応はsection fingerprintで、Python実装との対応はAST由来の `guide/.generated/code-symbols.json` でfail-closedに検証する。browser用 `.generated/code-symbols-runtime.json` はreview済み `content/meta/*.json` の参照symbolだけへ射影する。生成indexはproduction moduleをimport/executeせず、current treeへcommitしない。

Guideがbindしている `docs/architecture/*` または `docs/research/current-status.md` のsectionを変更した場合、対応pageの説明が新しい正本と一致することを人間が確認してから、対象topicだけを明示的にrefreshする。Guideが参照するPython symbolを変更した場合も、日本語説明・variables・関連testを読み直してから対象topicだけ `--refresh-code` する。refreshを単なるCI通過手段として実行しない。

```bash
python3 guide/tools/content_contract.py --refresh-code <topic-id>
python3 guide/tools/content_contract.py --refresh <topic-id>
python3 guide/tools/content_contract.py --check
npm --prefix guide run check
npm --prefix guide run e2e
```

`--refresh-code` は検証済みCodeReferenceのsource digestだけを更新し、説明を自動生成しない。`--refresh` はMarkdown section fingerprintだけを更新する。正本sectionやsourceが変わっていない通常のGuide UI変更ではfingerprintを更新しない。

Guide source linkはfloating `main` ではなくbuildした**exact revision**へ固定する。CIはPR head SHA、Pages build/smokeはverified `workflow_run.head_sha` を `GUIDE_SOURCE_REV` として渡す。production JavaScript chunkは500,000 bytes以下をhard budgetとし、Markdown renderer追加を理由にwarning thresholdを上げない。

Guide側の説明と正本が食い違う場合は、正本をGuideへ合わせず、現行source・contract test・正本docsから契約を再確認してGuideを修正する。

'''
text = text[:start] + replacement + text[end:]
old = "| Guideがbindする正本section / Python symbol | 対応する `guide/content/topics/*.json`, `--refresh` / `--refresh-code`, `guide/tools/content_contract.py --check` |"
new = "| Guideがbindする正本section / Python symbol | 対応する `guide/content/pages/*.md` と `guide/content/meta/*.json`, `--refresh` / `--refresh-code`, `guide/tools/content_contract.py --check` |"
if old not in text:
    raise SystemExit("Guide update matrix marker missing")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

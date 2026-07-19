interface WorkspacePageProps {
  title: string
  description: string
}

export function WorkspacePage({ title, description }: WorkspacePageProps) {
  return (
    <section className="workspace-page" aria-labelledby="workspace-title">
      <div className="workspace-page__header">
        <div>
          <h1 id="workspace-title">{title}</h1>
          <p>{description}</p>
        </div>
        <span className="workspace-page__state">FOUNDATION READY</span>
      </div>
      <div className="workspace-stage-grid">
        <article>
          <span>01</span>
          <h2>入力</h2>
          <p>既存の研究artifactと設定を安全に読み込みます。</p>
        </article>
        <article>
          <span>02</span>
          <h2>検証</h2>
          <p>因果性、identity、実行条件をUIから確認します。</p>
        </article>
        <article>
          <span>03</span>
          <h2>操作</h2>
          <p>既存workflowへ型付きAPIを通じて接続します。</p>
        </article>
      </div>
    </section>
  )
}

import {
  Activity,
  BarChart3,
  Beaker,
  Database,
  Gauge,
  GitCompareArrows,
  MonitorCog,
  ScrollText,
  Settings,
} from 'lucide-react'

export type WorkspaceId =
  | 'dashboard'
  | 'data'
  | 'experiments'
  | 'runs'
  | 'live'
  | 'compare'
  | 'evidence'
  | 'serving'
  | 'settings'

const items = [
  { id: 'dashboard' as const, label: 'ダッシュボード', compactLabel: 'Home', icon: Gauge },
  { id: 'data' as const, label: 'Data Lab', compactLabel: 'Data', icon: Database },
  { id: 'experiments' as const, label: '実験', compactLabel: 'Exp', icon: Beaker },
  { id: 'runs' as const, label: 'Run Center', compactLabel: 'Runs', icon: BarChart3 },
  { id: 'live' as const, label: 'Live Training', compactLabel: 'Live', icon: Activity },
  { id: 'compare' as const, label: '比較', compactLabel: 'Compare', icon: GitCompareArrows },
  { id: 'evidence' as const, label: 'Evidence Explorer', compactLabel: 'Evidence', icon: ScrollText },
  { id: 'serving' as const, label: 'Serving Monitor', compactLabel: 'Serving', icon: MonitorCog },
  { id: 'settings' as const, label: '設定', compactLabel: 'Settings', icon: Settings },
]

interface SidebarProps {
  active: WorkspaceId
  onSelect: (id: WorkspaceId) => void
}

export function Sidebar({ active, onSelect }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">Trade RL Studio</div>
      <nav aria-label="メインナビゲーション" className="nav-list">
        {items.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              type="button"
              aria-current={active === item.id ? 'page' : undefined}
              className={`nav-item ${active === item.id ? 'nav-item--active' : ''}`}
              onClick={() => onSelect(item.id)}
            >
              <Icon aria-hidden="true" size={18} strokeWidth={1.8} />
              <span className="nav-item__label">{item.label}</span>
              <span className="nav-item__compact">{item.compactLabel}</span>
            </button>
          )
        })}
      </nav>
      <div className="sidebar-status" aria-label="研究運用状態">
        <strong>NO-GO</strong>
        <span>研究目的運用</span>
        <small>注文機能はありません</small>
      </div>
    </aside>
  )
}

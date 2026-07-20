import { LockKeyhole } from 'lucide-react'

export function StatusBar() {
  return (
    <footer className="statusbar">
      <span><strong>システムメッセージ:</strong> すべてのサービスは正常に動作中です。</span>
      <span className="statusbar-local"><LockKeyhole size={13} aria-hidden="true" /> Local Only · 127.0.0.1</span>
    </footer>
  )
}

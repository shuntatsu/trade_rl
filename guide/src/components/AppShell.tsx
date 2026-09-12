import { Github, Moon, Sun } from "lucide-react";
import type { ReactNode } from "react";

import type { ThemePreference } from "../app/useTheme";
import type { GuideTopic } from "../content/schema";
import { MobileNav } from "./MobileNav";
import { SearchPalette } from "./SearchPalette";
import { Sidebar } from "./Sidebar";

export function AppShell({
  topics,
  activeId,
  onNavigate,
  theme,
  resolvedTheme,
  onThemeChange,
  children,
}: {
  topics: GuideTopic[];
  activeId: string;
  onNavigate: (id: string) => void;
  theme: ThemePreference;
  resolvedTheme: "light" | "dark";
  onThemeChange: (theme: ThemePreference) => void;
  children: ReactNode;
}) {
  const toggleTheme = () => onThemeChange(resolvedTheme === "dark" ? "light" : "dark");

  return (
    <div className="app-layout">
      <Sidebar topics={topics} activeId={activeId} onNavigate={onNavigate} />
      <div className="app-column">
        <header className="topbar">
          <MobileNav topics={topics} activeId={activeId} onNavigate={onNavigate} />
          <SearchPalette topics={topics} onNavigate={onNavigate} />
          <div className="topbar__actions">
            <a
              className="icon-button"
              href="https://github.com/shuntatsu/trade_rl"
              target="_blank"
              rel="noreferrer"
              aria-label="GitHub repository"
            >
              <Github size={18} />
            </a>
            <button
              type="button"
              className="icon-button"
              onClick={toggleTheme}
              aria-label={resolvedTheme === "dark" ? "ライトモードへ" : "ダークモードへ"}
              title={`Theme: ${theme}`}
            >
              {resolvedTheme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>
        <main className="main-content">{children}</main>
      </div>
    </div>
  );
}

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "studio/src/live/MarketReplayChart.tsx",
    "import type { TrainingTelemetryRecord } from '../data/types'\n",
    "import type { TrainingTelemetryRecord } from '../data/types'\n"
    "import { selectTelemetryTrack } from './telemetryTracks'\n",
)
replace_once(
    "studio/src/live/MarketReplayChart.tsx",
    "export function MarketReplayChart({ records, cursorSequence, compressed }: MarketReplayChartProps) {\n"
    "  const selected = (compressed ? records.filter((record) => record.eventType !== 'rollout') : records)\n"
    "    .filter((record) => record.close !== null)\n"
    "    .slice(-96)\n",
    "export function MarketReplayChart({ records, cursorSequence, compressed }: MarketReplayChartProps) {\n"
    "  const track = selectTelemetryTrack(records, cursorSequence)\n"
    "  const trackRecords = track?.records ?? []\n"
    "  const selected = (compressed ? trackRecords.filter((record) => record.eventType !== 'rollout') : trackRecords)\n"
    "    .filter((record) => record.close !== null)\n"
    "    .slice(-96)\n",
)
replace_once(
    "studio/src/live/MarketReplayChart.tsx",
    "            <g key={`${record.sequence}-${record.environmentId}`}>\n",
    "            <g\n"
    "              key={`${record.sequence}-${record.environmentId}`}\n"
    "              data-sequence={record.sequence}\n"
    "              data-track-key={track?.key ?? ''}\n"
    "            >\n",
)

replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "import { MarketReplayChart } from '../live/MarketReplayChart'\n"
    "import { useTrainingTelemetry } from '../live/useTrainingTelemetry'\n",
    "import { MarketReplayChart } from '../live/MarketReplayChart'\n"
    "import { deriveTelemetryTracks } from '../live/telemetryTracks'\n"
    "import { useTrainingTelemetry } from '../live/useTrainingTelemetry'\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "  const [jobId, setJobId] = useState<string | null>(null)\n"
    "  const [seed, setSeed] = useState<number | null>(null)\n"
    "  const [checkpointEvidenceId, setCheckpointEvidenceId] = useState<string | null>(null)\n",
    "  const [jobId, setJobId] = useState<string | null>(null)\n"
    "  const [seed, setSeed] = useState<number | null>(null)\n"
    "  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<number | null>(null)\n"
    "  const [selectedTrackKey, setSelectedTrackKey] = useState<string | null>(null)\n"
    "  const [checkpointEvidenceId, setCheckpointEvidenceId] = useState<string | null>(null)\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "  }, [seed, seedKey, telemetry.status?.selectedSeed])\n\n"
    "  useEffect(() => {\n"
    "    if (telemetry.records.length === 0) {\n"
    "      setCursor(0)\n"
    "      return\n"
    "    }\n"
    "    setCursor((current) => replayMode === 'live' || current === 0\n"
    "      ? telemetry.records.length - 1\n"
    "      : Math.min(current, telemetry.records.length - 1))\n"
    "  }, [replayMode, telemetry.records.length])\n\n"
    "  useEffect(() => {\n"
    "    if (!playing || replayMode === 'live' || telemetry.records.length < 2) return undefined\n"
    "    const timer = window.setInterval(() => {\n"
    "      setCursor((current) => current >= telemetry.records.length - 1 ? 0 : current + 1)\n"
    "    }, Math.max(90, 700 / speed))\n"
    "    return () => window.clearInterval(timer)\n"
    "  }, [playing, replayMode, speed, telemetry.records.length])\n\n"
    "  const selectedJob = jobs.find((job) => job.id === jobId) ?? null\n"
    "  const activeRecord = telemetry.records[Math.min(cursor, Math.max(0, telemetry.records.length - 1))] ?? null\n"
    "  const latestRecord = telemetry.records.at(-1) ?? null\n",
    "  }, [seed, seedKey, telemetry.status?.selectedSeed])\n\n"
    "  const tracks = useMemo(\n"
    "    () => deriveTelemetryTracks(telemetry.records),\n"
    "    [telemetry.records],\n"
    "  )\n"
    "  const environmentIds = useMemo(\n"
    "    () => [...new Set(tracks.map((track) => track.environmentId))].sort((left, right) => left - right),\n"
    "    [tracks],\n"
    "  )\n"
    "  const selectedTrack = tracks.find((track) => track.key === selectedTrackKey) ?? null\n"
    "  const selectedRecords = selectedTrack?.records ?? []\n\n"
    "  useEffect(() => {\n"
    "    if (tracks.length === 0) {\n"
    "      setSelectedEnvironmentId(null)\n"
    "      setSelectedTrackKey(null)\n"
    "      return\n"
    "    }\n"
    "    const current = tracks.find((track) => track.key === selectedTrackKey)\n"
    "    if (current) {\n"
    "      if (selectedEnvironmentId !== current.environmentId) {\n"
    "        setSelectedEnvironmentId(current.environmentId)\n"
    "      }\n"
    "      return\n"
    "    }\n"
    "    const environmentTracks = selectedEnvironmentId === null\n"
    "      ? []\n"
    "      : tracks.filter((track) => track.environmentId === selectedEnvironmentId)\n"
    "    const fallback = environmentTracks.at(-1) ?? tracks.at(-1) ?? null\n"
    "    setSelectedEnvironmentId(fallback?.environmentId ?? null)\n"
    "    setSelectedTrackKey(fallback?.key ?? null)\n"
    "  }, [selectedEnvironmentId, selectedTrackKey, tracks])\n\n"
    "  useEffect(() => {\n"
    "    if (selectedRecords.length === 0) {\n"
    "      setCursor(0)\n"
    "      return\n"
    "    }\n"
    "    setCursor((current) => replayMode === 'live' || current === 0\n"
    "      ? selectedRecords.length - 1\n"
    "      : Math.min(current, selectedRecords.length - 1))\n"
    "  }, [replayMode, selectedRecords.length, selectedTrackKey])\n\n"
    "  useEffect(() => {\n"
    "    if (!playing || replayMode === 'live' || selectedRecords.length < 2) return undefined\n"
    "    const timer = window.setInterval(() => {\n"
    "      setCursor((current) => current >= selectedRecords.length - 1 ? 0 : current + 1)\n"
    "    }, Math.max(90, 700 / speed))\n"
    "    return () => window.clearInterval(timer)\n"
    "  }, [playing, replayMode, selectedRecords.length, selectedTrackKey, speed])\n\n"
    "  const selectedJob = jobs.find((job) => job.id === jobId) ?? null\n"
    "  const activeRecord = selectedRecords[Math.min(cursor, Math.max(0, selectedRecords.length - 1))] ?? null\n"
    "  const latestRecord = selectedRecords.at(-1) ?? null\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "  const firstPortfolio = telemetry.records.find((record) => record.portfolioValue !== null)?.portfolioValue ?? null\n",
    "  const firstPortfolio = selectedRecords.find((record) => record.portfolioValue !== null)?.portfolioValue ?? null\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "  const recentEvents = useMemo(\n"
    "    () => telemetry.records.filter((record) => record.eventType !== 'rollout').slice(-8).reverse(),\n"
    "    [telemetry.records],\n"
    "  )\n"
    "  const equityValues = telemetry.records.map((record) => record.portfolioValue)\n"
    "  const baselineValues = telemetry.records.map((record) => record.baselinePortfolioValue)\n"
    "  const drawdownValues = telemetry.records.map((record) => record.drawdown === null ? null : -record.drawdown * 100)\n",
    "  const recentEvents = useMemo(\n"
    "    () => selectedRecords.filter((record) => record.eventType !== 'rollout').slice(-8).reverse(),\n"
    "    [selectedRecords],\n"
    "  )\n"
    "  const equityValues = selectedRecords.map((record) => record.portfolioValue)\n"
    "  const baselineValues = selectedRecords.map((record) => record.baselinePortfolioValue)\n"
    "  const drawdownValues = selectedRecords.map((record) => record.drawdown === null ? null : -record.drawdown * 100)\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "  const jump = (amount: number) => {\n"
    "    setPlaying(false)\n"
    "    setCursor((current) => Math.max(0, Math.min(telemetry.records.length - 1, current + amount)))\n"
    "  }\n",
    "  const chooseEnvironment = (environmentId: number | null) => {\n"
    "    setSelectedEnvironmentId(environmentId)\n"
    "    const latest = environmentId === null\n"
    "      ? tracks.at(-1) ?? null\n"
    "      : tracks.filter((track) => track.environmentId === environmentId).at(-1) ?? null\n"
    "    setSelectedTrackKey(latest?.key ?? null)\n"
    "  }\n\n"
    "  const jump = (amount: number) => {\n"
    "    setPlaying(false)\n"
    "    setCursor((current) => Math.max(0, Math.min(selectedRecords.length - 1, current + amount)))\n"
    "  }\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "          <label className=\"live-job-select\">Seed\n"
    "            <select value={effectiveSeed ?? ''} onChange={(event) => setSeed(event.target.value === '' ? null : Number(event.target.value))} aria-label=\"Live Training seed\">\n"
    "              {(telemetry.status?.availableSeeds.length ?? 0) === 0 ? <option value=\"\">seed待機中</option> : null}\n"
    "              {telemetry.status?.availableSeeds.map((value) => <option key={value} value={value}>Seed {value}</option>)}\n"
    "            </select>\n"
    "          </label>\n"
    "          <div className=\"live-segment-group\" aria-label=\"リプレイモード\">\n",
    "          <label className=\"live-job-select\">Seed\n"
    "            <select value={effectiveSeed ?? ''} onChange={(event) => setSeed(event.target.value === '' ? null : Number(event.target.value))} aria-label=\"Live Training seed\">\n"
    "              {(telemetry.status?.availableSeeds.length ?? 0) === 0 ? <option value=\"\">seed待機中</option> : null}\n"
    "              {telemetry.status?.availableSeeds.map((value) => <option key={value} value={value}>Seed {value}</option>)}\n"
    "            </select>\n"
    "          </label>\n"
    "          <label className=\"live-job-select\">Environment\n"
    "            <select\n"
    "              value={selectedEnvironmentId ?? ''}\n"
    "              onChange={(event) => chooseEnvironment(event.target.value === '' ? null : Number(event.target.value))}\n"
    "              aria-label=\"Live Training environment\"\n"
    "            >\n"
    "              {environmentIds.length === 0 ? <option value=\"\">env待機中</option> : null}\n"
    "              {environmentIds.map((value) => <option key={value} value={value}>Env {value}</option>)}\n"
    "            </select>\n"
    "          </label>\n"
    "          <label className=\"live-job-select\">Episode\n"
    "            <select\n"
    "              value={selectedTrack?.key ?? ''}\n"
    "              onChange={(event) => setSelectedTrackKey(event.target.value || null)}\n"
    "              aria-label=\"Live Training episode\"\n"
    "            >\n"
    "              {selectedEnvironmentId === null ? <option value=\"\">episode待機中</option> : null}\n"
    "              {tracks.filter((track) => track.environmentId === selectedEnvironmentId).map((track) => (\n"
    "                <option key={track.key} value={track.key}>\n"
    "                  {track.inferred ? `Legacy inferred ${track.legacyOrdinal}` : `Episode ${track.episodeId}`}\n"
    "                </option>\n"
    "              ))}\n"
    "            </select>\n"
    "          </label>\n"
    "          <div className=\"live-segment-group\" aria-label=\"リプレイモード\">\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "          <div className=\"live-buffer\"><Database size={14} aria-hidden=\"true\" /><strong>{telemetry.records.length}</strong> steps buffered</div>\n",
    "          <div className=\"live-buffer\"><Database size={14} aria-hidden=\"true\" /><span>{selectedRecords.length} / {telemetry.records.length} records</span></div>\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "          <MarketReplayChart records={telemetry.records} cursorSequence={activeRecord?.sequence ?? null} compressed={compressed} />\n",
    "          <MarketReplayChart records={selectedRecords} cursorSequence={activeRecord?.sequence ?? null} compressed={compressed} />\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "            <button type=\"button\" className=\"live-latest\" onClick={() => { setCursor(Math.max(0, telemetry.records.length - 1)); setReplayMode('live') }}>最新へ</button>\n",
    "            <button type=\"button\" className=\"live-latest\" onClick={() => { setCursor(Math.max(0, selectedRecords.length - 1)); setReplayMode('live') }}>最新へ</button>\n",
)
replace_once(
    "studio/src/pages/LiveTrainingPage.tsx",
    "              <button type=\"button\" key={`${record.sequence}-${record.environmentId}`} aria-label={`Step ${record.globalStep} ${label}`} onClick={() => { setPlaying(false); setCursor(Math.max(0, telemetry.records.findIndex((item) => item.sequence === record.sequence))) }}>\n",
    "              <button type=\"button\" key={`${record.sequence}-${record.environmentId}`} aria-label={`Step ${record.globalStep} ${label}`} onClick={() => { setPlaying(false); setCursor(Math.max(0, selectedRecords.findIndex((item) => item.sequence === record.sequence))) }}>\n",
)
replace_once(
    "studio/src/liveTraining.css",
    ".live-header-controls { display: flex; align-items: end; justify-content: flex-end; gap: 10px; min-width: 0; }\n",
    ".live-header-controls { display: flex; flex-wrap: wrap; align-items: end; justify-content: flex-end; gap: 6px 10px; min-width: 0; }\n",
)

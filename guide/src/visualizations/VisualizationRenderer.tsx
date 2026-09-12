import type { GuideVisualization } from "../content/schema";
import { ArchitectureFlow } from "./ArchitectureFlow";
import { EconomicsAuthorityPath } from "./EconomicsAuthorityPath";
import { ExperimentLoop } from "./ExperimentLoop";
import { FlowStepper } from "./FlowStepper";
import { ObservationVector } from "./ObservationVector";
import { ResearchStatusBoard } from "./ResearchStatusBoard";

export function VisualizationRenderer({
  visualization,
}: {
  visualization: GuideVisualization;
}) {
  switch (visualization.kind) {
    case "architecture":
      return <ArchitectureFlow visualization={visualization} />;
    case "data-flow":
      return <FlowStepper steps={visualization.steps} />;
    case "economics":
      return <EconomicsAuthorityPath visualization={visualization} />;
    case "observation":
      return <ObservationVector visualization={visualization} />;
    case "experiment-loop":
      return <ExperimentLoop visualization={visualization} />;
    case "research-status":
      return <ResearchStatusBoard visualization={visualization} />;
  }
}

import type { GuideVisualization } from "../content/schema";
import { ArchitectureFlow } from "./ArchitectureFlow";
import { CodeMap } from "./CodeMap";
import { EconomicsAuthorityPath } from "./EconomicsAuthorityPath";
import { ExperimentLoop } from "./ExperimentLoop";
import { FlowStepper } from "./FlowStepper";
import { ObservationVector } from "./ObservationVector";
import { ResearchStatusBoard } from "./ResearchStatusBoard";
import { SequenceDiagram } from "./SequenceDiagram";

const ignoreSelection = () => undefined;

export function VisualizationRenderer({
  visualization,
  selectedStep,
  onSelectStep,
  selectedNode,
  onSelectNode,
}: {
  visualization: GuideVisualization;
  selectedStep?: string;
  onSelectStep?: (stepId: string) => void;
  selectedNode?: string;
  onSelectNode?: (nodeId: string) => void;
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
    case "sequence":
      return (
        <SequenceDiagram
          visualization={visualization}
          selectedStep={selectedStep}
          onSelectStep={onSelectStep ?? ignoreSelection}
        />
      );
    case "code-map":
      return (
        <CodeMap
          visualization={visualization}
          selectedNode={selectedNode}
          onSelectNode={onSelectNode ?? ignoreSelection}
        />
      );
  }
}

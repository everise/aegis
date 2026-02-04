/**
 * Agent selector component.
 * Shows available agents with a dropdown selector styled like model-selector.
 */

import { FC } from "react";
import { Bot } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface Agent {
  id: string;
  name: string;
  description: string;
  icon?: string;
}

const AVAILABLE_AGENTS: Agent[] = [
  {
    id: "image-generation",
    name: "Image Generation",
    description: "Generate, evaluate, and refine images using AI",
  },
  // Future agents can be added here
  // {
  //   id: "code-assistant",
  //   name: "Code Assistant",
  //   description: "Help with coding tasks",
  // },
];

interface AgentSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
}

export const AgentSelector: FC<AgentSelectorProps> = ({
  value,
  onValueChange,
}) => {
  const selectedAgent =
    AVAILABLE_AGENTS.find((a) => a.id === value) || AVAILABLE_AGENTS[0];

  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger
        className={cn(
          "flex w-fit items-center justify-between gap-2",
          "whitespace-nowrap rounded-md text-sm outline-none",
          "transition-colors focus-visible:ring-2 focus-visible:ring-ring/50",
          "border border-input bg-transparent hover:bg-accent hover:text-accent-foreground",
          "h-9 px-3 py-2"
        )}
      >
        <span className="flex items-center gap-2">
          <Bot className="size-4 shrink-0" />
          <span className="truncate font-medium">{selectedAgent?.name}</span>
        </span>
      </SelectTrigger>
      <SelectContent>
        {AVAILABLE_AGENTS.map((agent) => (
          <SelectItem key={agent.id} value={agent.id} className="py-2">
            <div className="flex items-center gap-2">
              <Bot className="size-4 shrink-0" />
              <div className="flex flex-col">
                <span className="font-medium">{agent.name}</span>
                <span className="text-xs text-muted-foreground">
                  {agent.description}
                </span>
              </div>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export { AVAILABLE_AGENTS };
export type { Agent };

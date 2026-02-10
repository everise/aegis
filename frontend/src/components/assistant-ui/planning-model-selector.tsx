/**
 * Planning model selector component.
 * Fetches available models from the backend and switches the active model.
 */

import { FC, useCallback, useEffect, useState } from "react";
import { Brain, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { planningModelsApi, type PlanningModel } from "@/services/api";
import { getModelIcon } from "@/components/icons/model-icons";

interface PlanningModelSelectorProps {
  value: string;
  onValueChange: (value: string) => void;
}

export const PlanningModelSelector: FC<PlanningModelSelectorProps> = ({
  value,
  onValueChange,
}) => {
  const [models, setModels] = useState<PlanningModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen] = useState(false);

  // Fetch available models from backend on mount
  useEffect(() => {
    let cancelled = false;

    const fetchModels = async () => {
      try {
        const data = await planningModelsApi.list();
        if (cancelled) return;
        setModels(data.models);
        // If no value set yet, sync with backend active model
        if (!value && data.active_model_id) {
          onValueChange(data.active_model_id);
        }
      } catch (err) {
        console.error("Failed to load planning models:", err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchModels();
    return () => {
      cancelled = true;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelect = useCallback(
    async (modelId: string) => {
      try {
        await planningModelsApi.setActive(modelId);
        onValueChange(modelId);
      } catch (err) {
        console.error("Failed to switch planning model:", err);
      }
      setOpen(false);
    },
    [onValueChange],
  );

  const selectedModel = models.find((m) => m.id === value) || models[0];

  if (loading) {
    return (
      <div className="flex h-9 items-center gap-2 rounded-md border border-input bg-transparent px-3 py-2 text-sm text-muted-foreground">
        <span className="animate-pulse">Loading…</span>
        <Brain className="size-4 opacity-50" />
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className={cn(
          "inline-flex h-9 items-center gap-2 rounded-md px-3 py-2",
          "whitespace-nowrap text-sm font-medium outline-none",
          "border border-input bg-transparent",
          "transition-colors hover:bg-accent hover:text-accent-foreground",
          "focus-visible:ring-2 focus-visible:ring-ring/50",
        )}
      >
        {(() => {
          const Icon = selectedModel ? getModelIcon(selectedModel.id) : undefined;
          return Icon ? <Icon className="size-4 shrink-0" /> : null;
        })()}
        <span className="truncate">{selectedModel?.name ?? "Select Model"}</span>
        <ChevronDown
          className={cn(
            "size-4 shrink-0 opacity-50 transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <>
          {/* Backdrop */}
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />

          {/* Menu */}
          <div
            className={cn(
              "absolute left-0 top-full z-50 mt-1 min-w-[220px]",
              "rounded-md border bg-popover p-1 shadow-md",
              "animate-in fade-in-0 zoom-in-95",
            )}
          >
            {models.map((model) => {
              const ModelIcon = getModelIcon(model.id);
              return (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => handleSelect(model.id)}
                  className={cn(
                    "flex w-full items-start gap-2 rounded-sm px-2 py-2 text-left text-sm",
                    "outline-none transition-colors hover:bg-accent hover:text-accent-foreground",
                    model.id === value && "bg-accent/50",
                  )}
                >
                  {ModelIcon ? (
                    <ModelIcon className="mt-0.5 size-4 shrink-0" />
                  ) : (
                    <Brain className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                  )}
                  <div className="flex flex-col gap-0.5">
                    <span className="font-medium">{model.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {model.provider} · {model.description}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

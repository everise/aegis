/**
 * Assistant UI Thread component with shadcn styling.
 * Based on assistant-ui's shadcn example.
 */

"use client";

import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  makeAssistantToolUI,
  useMessage,
  useThread,
} from "@assistant-ui/react";
import { type FC, useRef, useCallback, useContext, useSyncExternalStore } from "react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  PlusIcon,
  RefreshCwIcon,
  SquareIcon,
  SearchIcon,
  WrenchIcon,
  Loader2Icon,
  SparklesIcon,
  PaletteIcon,
  ZapIcon,
  BrainIcon,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { toast, Toaster } from "sonner";

import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { BranchContext } from "@/pages/ChatPage";
import { SessionIdContext } from "@/pages/ChatPage";
import { subscribeToBranchState, getBranchStateVersion } from "@/hooks/useAegisRuntime";
import ContextCounter from "@/components/assistant-ui/context-counter";

// Suggestion prompts for empty state - 2 shown side by side
const SUGGESTIONS = [
  {
    text1: "What's the weather",
    text2: "in San Francisco?",
  },
  {
    text1: "Explain React hooks",
    text2: "like useState and useEffect",
  },
];

// Tool UI components for displaying agent tool calls
// Each skill has its own distinctive style and icon

const ToolFallback = makeAssistantToolUI({
  toolName: "*", // Fallback for all tools
  render: function ToolFallbackUI({ args, result, status }) {
    const params = (args as Record<string, unknown>)?.params as Record<string, unknown> || {};
    const isRunning = status?.type === "running";
    const isComplete = status?.type === "complete";
    
    // Get display text
    let displayText = params.prompt as string || "Processing...";
    
    // Get result info
    let resultInfo = "";
    if (isComplete && result) {
      const r = result as Record<string, unknown>;
      if (r.image_url) {
        resultInfo = "Completed";
      } else if (r.feedback) {
        resultInfo = r.feedback as string;
      } else {
        resultInfo = "Done";
      }
    }
    
    return (
      <div className="my-3 overflow-hidden rounded-xl border border-border/50 bg-gradient-to-r from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 shadow-sm">
        <div className="flex items-center gap-3 px-4 py-3">
          <div className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl",
            "bg-slate-200 dark:bg-slate-700"
          )}>
            {isRunning ? (
              <Loader2Icon className="size-5 animate-spin text-slate-600 dark:text-slate-300" />
            ) : (
              <WrenchIcon className="size-5 text-slate-600 dark:text-slate-300" />
            )}
          </div>
          <div className="flex flex-col gap-0.5 min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                Tool
              </span>
              {isRunning && (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-400">
                  Running
                </span>
              )}
              {isComplete && (
                <span className="inline-flex items-center gap-1 rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs font-medium text-green-700 dark:text-green-400">
                  <CheckIcon className="size-3" />
                  Done
                </span>
              )}
            </div>
            <span className="text-sm text-foreground truncate">{displayText}</span>
            {resultInfo && (
              <span className="text-xs text-muted-foreground">{resultInfo}</span>
            )}
          </div>
        </div>
      </div>
    );
  },
});

const TextToImageTool = makeAssistantToolUI({
  toolName: "text_to_image",
  render: function TextToImageToolUI({ args, result, status }) {
    const params = (args as Record<string, unknown>)?.params as Record<string, unknown> || {};
    const prompt = params.prompt as string || "Generating image...";
    const isRunning = status?.type === "running";
    const isComplete = status?.type === "complete";
    const r = result as Record<string, unknown> | undefined;
    const imageUrl = r?.image_url as string | undefined;
    
    return (
      <div className="my-3 overflow-hidden rounded-xl border border-purple-200 dark:border-purple-800 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-950/50 dark:to-pink-950/50 shadow-sm">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-purple-100 dark:border-purple-800/50">
          <div className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl",
            "bg-gradient-to-br from-purple-500 to-pink-500 shadow-lg shadow-purple-500/25"
          )}>
            {isRunning ? (
              <Loader2Icon className="size-5 animate-spin text-white" />
            ) : (
              <SparklesIcon className="size-5 text-white" />
            )}
          </div>
          <div className="flex flex-col gap-0.5 min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-purple-600 dark:text-purple-400">
                Text to Image
              </span>
              {isRunning && (
                <span className="inline-flex items-center gap-1 rounded-full bg-purple-100 dark:bg-purple-900/30 px-2 py-0.5 text-xs font-medium text-purple-700 dark:text-purple-400 animate-pulse">
                  <Loader2Icon className="size-3 animate-spin" />
                  Generating
                </span>
              )}
              {isComplete && (
                <span className="inline-flex items-center gap-1 rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs font-medium text-green-700 dark:text-green-400">
                  <CheckIcon className="size-3" />
                  Generated
                </span>
              )}
            </div>
            <span className="text-sm text-foreground line-clamp-2">{prompt}</span>
          </div>
        </div>
        {/* Show generated image */}
        {isComplete && imageUrl && (
          <div className="p-3">
            <img 
              src={imageUrl} 
              alt={prompt}
              className="w-full max-w-md rounded-lg shadow-md"
            />
          </div>
        )}
        {isRunning && (
          <div className="px-4 py-3 flex items-center gap-2 text-sm text-purple-600 dark:text-purple-400">
            <div className="flex gap-1">
              <span className="size-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="size-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="size-2 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span>Creating your masterpiece...</span>
          </div>
        )}
      </div>
    );
  },
});

const EvaluateImageTool = makeAssistantToolUI({
  toolName: "evaluate_image",
  render: function EvaluateImageToolUI({ result, status }) {
    const isRunning = status?.type === "running";
    const isComplete = status?.type === "complete";
    const r = result as Record<string, unknown> | undefined;
    const overallScore = r?.overall_score as number | undefined;
    const scores = r?.scores as Record<string, number> | undefined;
    const feedback = r?.feedback as string || "";
    const scorePercent = overallScore ? Math.round(overallScore * 100) : null;
    
    // Determine score color
    const getScoreColor = (score: number) => {
      if (score >= 0.8) return "text-green-600 dark:text-green-400";
      if (score >= 0.6) return "text-amber-600 dark:text-amber-400";
      return "text-red-600 dark:text-red-400";
    };
    
    return (
      <div className="my-3 overflow-hidden rounded-xl border border-blue-200 dark:border-blue-800 bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-950/50 dark:to-cyan-950/50 shadow-sm">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-blue-100 dark:border-blue-800/50">
          <div className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl",
            "bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg shadow-blue-500/25"
          )}>
            {isRunning ? (
              <Loader2Icon className="size-5 animate-spin text-white" />
            ) : (
              <SearchIcon className="size-5 text-white" />
            )}
          </div>
          <div className="flex flex-col gap-0.5 min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-blue-600 dark:text-blue-400">
                Quality Evaluation
              </span>
              {isRunning && (
                <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 dark:bg-blue-900/30 px-2 py-0.5 text-xs font-medium text-blue-700 dark:text-blue-400 animate-pulse">
                  Analyzing
                </span>
              )}
            </div>
            <span className="text-sm text-foreground">Analyzing image quality and aesthetics</span>
          </div>
        </div>
        {/* Show scores */}
        {isComplete && scorePercent !== null && (
          <div className="px-4 py-3 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Overall Score</span>
              <span className={cn("text-2xl font-bold", getScoreColor(overallScore!))}>
                {scorePercent}%
              </span>
            </div>
            {scores && (
              <div className="grid grid-cols-3 gap-2 text-xs">
                {Object.entries(scores).map(([key, value]) => (
                  <div key={key} className="flex flex-col items-center p-2 rounded-lg bg-white/50 dark:bg-black/20">
                    <span className="text-muted-foreground capitalize">{key.replace(/_/g, " ")}</span>
                    <span className={cn("font-semibold", getScoreColor(value))}>
                      {Math.round(value * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
            {feedback && (
              <p className="text-sm text-muted-foreground italic">&quot;{feedback}&quot;</p>
            )}
          </div>
        )}
        {isRunning && (
          <div className="px-4 py-3 flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400">
            <BrainIcon className="size-4 animate-pulse" />
            <span>Evaluating quality metrics...</span>
          </div>
        )}
      </div>
    );
  },
});

const RepairImageTool = makeAssistantToolUI({
  toolName: "repair_image",
  render: function RepairImageToolUI({ args, result, status }) {
    const params = (args as Record<string, unknown>)?.params as Record<string, unknown> || {};
    const prompt = params.prompt as string || "Repairing image...";
    const isRunning = status?.type === "running";
    const isComplete = status?.type === "complete";
    const r = result as Record<string, unknown> | undefined;
    const imageUrl = r?.image_url as string | undefined;
    
    return (
      <div className="my-3 overflow-hidden rounded-xl border border-orange-200 dark:border-orange-800 bg-gradient-to-r from-orange-50 to-amber-50 dark:from-orange-950/50 dark:to-amber-950/50 shadow-sm">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-orange-100 dark:border-orange-800/50">
          <div className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl",
            "bg-gradient-to-br from-orange-500 to-amber-500 shadow-lg shadow-orange-500/25"
          )}>
            {isRunning ? (
              <Loader2Icon className="size-5 animate-spin text-white" />
            ) : (
              <WrenchIcon className="size-5 text-white" />
            )}
          </div>
          <div className="flex flex-col gap-0.5 min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-orange-600 dark:text-orange-400">
                Image Repair
              </span>
              {isRunning && (
                <span className="inline-flex items-center gap-1 rounded-full bg-orange-100 dark:bg-orange-900/30 px-2 py-0.5 text-xs font-medium text-orange-700 dark:text-orange-400 animate-pulse">
                  Repairing
                </span>
              )}
              {isComplete && (
                <span className="inline-flex items-center gap-1 rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs font-medium text-green-700 dark:text-green-400">
                  <CheckIcon className="size-3" />
                  Fixed
                </span>
              )}
            </div>
            <span className="text-sm text-foreground line-clamp-2">{prompt}</span>
          </div>
        </div>
        {/* Show repaired image */}
        {isComplete && imageUrl && (
          <div className="p-3">
            <img 
              src={imageUrl} 
              alt="Repaired image"
              className="w-full max-w-md rounded-lg shadow-md"
            />
          </div>
        )}
        {isRunning && (
          <div className="px-4 py-3 flex items-center gap-2 text-sm text-orange-600 dark:text-orange-400">
            <PaletteIcon className="size-4 animate-pulse" />
            <span>Enhancing and repairing...</span>
          </div>
        )}
      </div>
    );
  },
});

const FinishTool = makeAssistantToolUI({
  toolName: "finish",
  render: function FinishToolUI({ args, result }) {
    const r = result as Record<string, unknown> | undefined;
    const message = r?.message as string || (args as Record<string, unknown>)?.message as string || "Task completed successfully";
    const imageUrl = r?.image_url as string | undefined;
    
    return (
      <div className="my-3 overflow-hidden rounded-xl border border-green-200 dark:border-green-800 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/50 dark:to-emerald-950/50 shadow-sm">
        <div className="flex items-center gap-3 px-4 py-3">
          <div className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-xl",
            "bg-gradient-to-br from-green-500 to-emerald-500 shadow-lg shadow-green-500/25"
          )}>
            <ZapIcon className="size-5 text-white" />
          </div>
          <div className="flex flex-col gap-0.5 min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-green-600 dark:text-green-400">
                Completed
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-green-100 dark:bg-green-900/30 px-2 py-0.5 text-xs font-medium text-green-700 dark:text-green-400">
                <CheckIcon className="size-3" />
                Success
              </span>
            </div>
            <span className="text-sm text-foreground">{message}</span>
          </div>
        </div>
        {/* Show final image if present */}
        {imageUrl && (
          <div className="p-3 border-t border-green-100 dark:border-green-800/50">
            <img 
              src={imageUrl} 
              alt="Final result"
              className="w-full max-w-md rounded-lg shadow-md"
            />
          </div>
        )}
      </div>
    );
  },
});

export const Thread: FC = () => {
  return (
    <TooltipProvider>
      <ThreadPrimitive.Root
        className="aui-thread-root @container flex h-full flex-col items-center bg-background"
        style={{ "--thread-max-width": "44rem" } as React.CSSProperties}
      >
        <ThreadPrimitive.Viewport className="aui-thread-viewport relative flex w-full flex-1 flex-col items-center overflow-y-scroll scroll-smooth px-4 pt-4">
          <ThreadWelcome />
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
          {/* Footer with Composer - sticky at bottom, centered with max-width */}
          <div className="aui-thread-viewport-footer sticky bottom-0 mt-auto flex w-full max-w-[44rem] flex-col gap-4 overflow-visible rounded-t-3xl bg-background pb-4 md:pb-6">
            <ThreadScrollToBottom />
            <Composer />
          </div>
        </ThreadPrimitive.Viewport>
        {/* Register tool UIs */}
        <TextToImageTool />
        <EvaluateImageTool />
        <RepairImageTool />
        <FinishTool />
        <ToolFallback />
      </ThreadPrimitive.Root>
      <Toaster position="top-center" richColors />
    </TooltipProvider>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom className="absolute -top-10 left-1/2 -translate-x-1/2 rounded-full border border-input bg-background p-2 shadow-md transition-opacity disabled:pointer-events-none disabled:opacity-0">
      <ArrowDownIcon className="size-4" />
      <span className="sr-only">Scroll to bottom</span>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <ThreadPrimitive.Empty>
      <div className="aui-thread-welcome-root flex w-full max-w-[44rem] grow flex-col justify-end">
        {/* Suggestion Buttons - 2 columns side by side, at bottom of welcome area */}
        <div className="aui-thread-welcome-suggestions grid w-full grid-cols-2 gap-2 pb-4">
          {SUGGESTIONS.map((suggestion, index) => (
            <div
              key={index}
              className="aui-thread-welcome-suggestion-display animate-in fade-in slide-in-from-bottom-2 fill-mode-both duration-200"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <ThreadPrimitive.Suggestion
                prompt={`${suggestion.text1} ${suggestion.text2}`}
                autoSend
                asChild
              >
                <button
                  type="button"
                  className={cn(
                    "aui-thread-welcome-suggestion",
                    "inline-flex shrink-0 font-medium outline-none",
                    "focus-visible:ring-2 focus-visible:ring-ring/50",
                    "hover:bg-muted hover:text-accent-foreground",
                    "h-auto w-full flex-col items-start justify-start gap-1",
                    "rounded-2xl border px-4 py-3 text-left text-sm transition-colors"
                  )}
                >
                  <span className="aui-thread-welcome-suggestion-text-1 font-medium">
                    {suggestion.text1}
                  </span>
                  <span className="aui-thread-welcome-suggestion-text-2 text-muted-foreground">
                    {suggestion.text2}
                  </span>
                </button>
              </ThreadPrimitive.Suggestion>
            </div>
          ))}
        </div>
      </div>
    </ThreadPrimitive.Empty>
  );
};

const Composer: FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sessionId = useContext(SessionIdContext);

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <ComposerPrimitive.Root className="aui-composer-root focus-within:border-ring relative flex w-full flex-col rounded-2xl border bg-background shadow-sm transition-colors">
      {/* Hidden file input for attachments */}
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="image/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) {
            console.log("File selected:", file.name);
            // TODO: Handle file upload
          }
        }}
      />

      {/* Input area */}
      <ComposerPrimitive.Input
        autoFocus
        placeholder="Send a message..."
        rows={1}
        className="aui-composer-input placeholder:text-muted-foreground min-h-[48px] max-h-40 flex-grow resize-none border-none bg-transparent px-4 py-3 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />

      {/* Bottom toolbar */}
      <div className="flex items-center justify-between px-3 pb-3">
        {/* Left side - attachment button */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-8 rounded-lg"
              onClick={handleAttachClick}
            >
              <PlusIcon className="size-4" />
              <span className="sr-only">Attach file</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>Attach file</TooltipContent>
        </Tooltip>

        {/* Right side - context counter + send/stop button */}
        <div className="flex items-center gap-1">
          {/* Context Counter */}
          <ContextCounter sessionId={sessionId} />

          {/* Send/Stop button */}
          <ThreadPrimitive.If running={false}>
            <ComposerPrimitive.Send asChild>
              <Button
                size="icon"
                variant="ghost"
                className="size-8 rounded-lg transition-opacity"
              >
                <ArrowUpIcon className="size-4" />
                <span className="sr-only">Send message</span>
              </Button>
            </ComposerPrimitive.Send>
          </ThreadPrimitive.If>
          <ThreadPrimitive.If running>
            <ComposerPrimitive.Cancel asChild>
              <Button
                size="icon"
                variant="ghost"
                className="relative size-8 rounded-lg transition-opacity"
              >
                {/* Spinning ring around stop button */}
                <span className="absolute inset-0 animate-spin rounded-full border-2 border-transparent border-t-primary" />
                <SquareIcon className="size-3 fill-current" />
                <span className="sr-only">Stop generation</span>
              </Button>
            </ComposerPrimitive.Cancel>
          </ThreadPrimitive.If>
        </div>
      </div>
    </ComposerPrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="aui-user-message grid w-full max-w-[44rem] auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] gap-y-2 py-4">
      <UserActionBar />
      <div className="bg-muted text-foreground col-start-2 row-start-1 max-w-xl break-words rounded-3xl px-5 py-2.5">
        <MessagePrimitive.Content />
      </div>
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return null;
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="aui-assistant-message relative grid w-full max-w-[44rem] grid-cols-[auto_auto_1fr] grid-rows-[auto_1fr] py-4">
      <Avatar className="col-start-1 row-span-full row-start-1 mr-4 size-8">
        <AvatarFallback className="bg-primary text-primary-foreground text-xs">
          A
        </AvatarFallback>
      </Avatar>
      <div className="text-foreground col-span-2 col-start-2 row-start-1 my-1.5 max-w-xl break-words leading-7">
        <MessagePrimitive.Content
          components={{
            Text: MarkdownText,
          }}
        />
      </div>
      <AssistantActionBar />
      <BranchPicker className="col-start-2 row-start-2 -ml-2 mr-2" />
    </MessagePrimitive.Root>
  );
};

const MarkdownText: FC<{ text: string }> = ({ text }) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      className="prose prose-sm dark:prose-invert max-w-none"
      components={{
        img: ({ src, alt }) => (
          <img
            src={src}
            alt={alt || "Generated image"}
            className="rounded-lg max-w-full h-auto my-4"
          />
        ),
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        ul: ({ children }) => (
          <ul className="list-disc pl-4 mb-2">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-4 mb-2">{children}</ol>
        ),
        code: ({ className, children }) => {
          const isInline = !className;
          if (isInline) {
            return (
              <code className="bg-muted px-1 py-0.5 rounded text-sm">
                {children}
              </code>
            );
          }
          return (
            <pre className="bg-muted p-3 rounded-lg overflow-x-auto my-2">
              <code className="text-sm">{children}</code>
            </pre>
          );
        },
      }}
    >
      {text}
    </ReactMarkdown>
  );
};

const AssistantActionBar: FC = () => {
  const message = useMessage();
  
  // Convert message content to markdown for copying
  const getMarkdownContent = useCallback(() => {
    const content = message.content;
    if (!content) return "";
    
    const parts: string[] = [];
    
    for (const part of content) {
      if (part.type === "text") {
        parts.push(part.text);
      } else if (part.type === "tool-call") {
        const toolCall = part as { toolName: string; args?: { thought?: string; params?: { prompt?: string } }; result?: unknown };
        const thought = toolCall.args?.thought;
        const prompt = toolCall.args?.params?.prompt;
        const result = toolCall.result as Record<string, unknown> | undefined;
        
        if (thought) {
          parts.push(`**Thinking:** ${thought}`);
        }
        
        parts.push(`**Tool:** ${toolCall.toolName}`);
        
        if (prompt) {
          parts.push(`**Prompt:** ${prompt}`);
        }
        
        if (result) {
          if (result.image_url) {
            parts.push(`![Generated Image](${result.image_url})`);
          }
          if (result.overall_score !== undefined) {
            parts.push(`**Score:** ${Math.round((result.overall_score as number) * 100)}%`);
          }
          if (result.feedback) {
            parts.push(`**Feedback:** ${result.feedback}`);
          }
        }
        parts.push("");
      }
    }
    
    return parts.join("\n");
  }, [message.content]);
  
  const handleCopy = useCallback(async () => {
    const markdown = getMarkdownContent();
    try {
      await navigator.clipboard.writeText(markdown);
      toast.success("已复制到剪贴板", {
        description: "内容已复制为Markdown格式",
        duration: 2000,
      });
    } catch {
      toast.error("复制失败", {
        description: "无法访问剪贴板",
        duration: 2000,
      });
    }
  }, [getMarkdownContent]);
  
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="text-muted-foreground col-start-3 row-start-2 -ml-1 flex gap-1"
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon" className="size-8" onClick={handleCopy}>
            <CopyIcon className="size-4" />
            <span className="sr-only">Copy message</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent>复制</TooltipContent>
      </Tooltip>
      <Tooltip>
        <ActionBarPrimitive.Reload asChild>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="size-8">
              <RefreshCwIcon className="size-4" />
              <span className="sr-only">Regenerate</span>
            </Button>
          </TooltipTrigger>
        </ActionBarPrimitive.Reload>
        <TooltipContent>重新生成</TooltipContent>
      </Tooltip>
    </ActionBarPrimitive.Root>
  );
};

const BranchPicker: FC<{ className?: string }> = ({ className }) => {
  const branchContext = useContext(BranchContext);
  const message = useMessage();
  const messages = useThread((t) => t.messages);
  
  // Subscribe to branch state changes to trigger re-renders
  useSyncExternalStore(subscribeToBranchState, getBranchStateVersion);
  
  // Find the user message ID for the current assistant message
  const getUserMessageId = useCallback(() => {
    if (!messages) return null;
    const currentId = message.id;
    
    // If current message is a user message, return its ID
    if (message.role === 'user') {
      return currentId;
    }
    
    // Find the user message before this assistant message
    const currentIndex = messages.findIndex((m) => m.id === currentId);
    if (currentIndex <= 0) return null;
    
    for (let i = currentIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        return messages[i].id;
      }
    }
    return null;
  }, [messages, message]);
  
  const userMessageId = getUserMessageId();
  const branchInfo = userMessageId && branchContext ? branchContext.getBranchInfoForMessage(userMessageId) : null;
  
  // If no branch info or only one branch, try the built-in branch picker
  if (!branchInfo) {
    return (
      <BranchPickerPrimitive.Root
        hideWhenSingleBranch
        className={cn(
          "inline-flex items-center text-xs text-muted-foreground",
          className
        )}
      >
        <BranchPickerPrimitive.Previous asChild>
          <Button variant="ghost" size="icon" className="size-6">
            <ChevronLeftIcon className="size-4" />
          </Button>
        </BranchPickerPrimitive.Previous>
        <span className="font-medium">
          <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
        </span>
        <BranchPickerPrimitive.Next asChild>
          <Button variant="ghost" size="icon" className="size-6">
            <ChevronRightIcon className="size-4" />
          </Button>
        </BranchPickerPrimitive.Next>
      </BranchPickerPrimitive.Root>
    );
  }
  
  // Custom branch picker for our branch management
  const handlePrev = () => {
    if (userMessageId && branchContext) {
      branchContext.handleSwitchBranch(userMessageId, 'prev');
    }
  };
  
  const handleNext = () => {
    if (userMessageId && branchContext) {
      branchContext.handleSwitchBranch(userMessageId, 'next');
    }
  };
  
  return (
    <div className={cn("inline-flex items-center text-xs text-muted-foreground", className)}>
      <Button 
        variant="ghost" 
        size="icon" 
        className="size-6"
        onClick={handlePrev}
        disabled={branchInfo.current <= 1}
      >
        <ChevronLeftIcon className="size-4" />
      </Button>
      <span className="font-medium">
        {branchInfo.current} / {branchInfo.count}
      </span>
      <Button 
        variant="ghost" 
        size="icon" 
        className="size-6"
        onClick={handleNext}
        disabled={branchInfo.current >= branchInfo.count}
      >
        <ChevronRightIcon className="size-4" />
      </Button>
    </div>
  );
};

export default Thread;

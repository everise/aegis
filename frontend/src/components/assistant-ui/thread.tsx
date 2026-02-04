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
} from "@assistant-ui/react";
import { type FC, useRef } from "react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
      </ThreadPrimitive.Root>
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

        {/* Right side - send button */}
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
      <BranchPicker className="col-span-full col-start-1 row-start-2 -mr-1 justify-end" />
    </MessagePrimitive.Root>
  );
};

const UserActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      className="col-start-1 mr-3 mt-2.5 flex flex-col items-end"
    >
      <Tooltip>
        <ActionBarPrimitive.Edit asChild>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="size-8">
              <PencilIcon className="size-4" />
              <span className="sr-only">Edit message</span>
            </Button>
          </TooltipTrigger>
        </ActionBarPrimitive.Edit>
        <TooltipContent>Edit message</TooltipContent>
      </Tooltip>
    </ActionBarPrimitive.Root>
  );
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
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="text-muted-foreground col-start-3 row-start-2 -ml-1 flex gap-1"
    >
      <Tooltip>
        <ActionBarPrimitive.Copy asChild>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="size-8">
              <MessagePrimitive.If copied>
                <CheckIcon className="size-4" />
              </MessagePrimitive.If>
              <MessagePrimitive.If copied={false}>
                <CopyIcon className="size-4" />
              </MessagePrimitive.If>
              <span className="sr-only">Copy message</span>
            </Button>
          </TooltipTrigger>
        </ActionBarPrimitive.Copy>
        <TooltipContent>Copy</TooltipContent>
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
        <TooltipContent>Regenerate</TooltipContent>
      </Tooltip>
    </ActionBarPrimitive.Root>
  );
};

const BranchPicker: FC<{ className?: string }> = ({ className }) => {
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
};

export default Thread;

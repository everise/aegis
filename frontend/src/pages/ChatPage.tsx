/**
 * Main Chat Page with assistant-ui integration.
 * Styled to match assistant-ui shadcn example.
 */

import { useState, useCallback, useRef, useSyncExternalStore, createContext } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { Plus, PanelLeft, Settings, MoreHorizontal, Archive, Trash2 } from "lucide-react";

import { Thread } from "@/components/assistant-ui/thread";
import { AgentSelector } from "@/components/assistant-ui/agent-selector";
import { useAegisRuntime, subscribeToRunningState, isSessionRunning, subscribeToBranchState, getBranchStateVersion, getRunningStateVersion } from "@/hooks/useAegisRuntime";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

interface Session {
  id: number;
  status: string;
  task_type: string | null;
  created_at: string;
}

// Branch context for passing branch functions to Thread component
export interface BranchContextType {
  handleSwitchBranch: (userMessageId: string, direction: 'prev' | 'next') => void;
  getBranchInfoForMessage: (userMessageId: string) => { count: number; current: number } | null;
}

export const BranchContext = createContext<BranchContextType | null>(null);

export default function ChatPage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedAgent, setSelectedAgent] = useState("image-generation");
  const [sessionsLoaded, setSessionsLoaded] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [deletingIds, setDeletingIds] = useState<Set<number>>(new Set());
  const newSessionKey = useRef(0);

  const currentSessionId = sessionId ? parseInt(sessionId, 10) : null;

  // Subscribe to running state changes to trigger re-renders
  useSyncExternalStore(
    subscribeToRunningState,
    getRunningStateVersion
  );

  // Subscribe to branch state changes to trigger re-renders for branch picker
  useSyncExternalStore(
    subscribeToBranchState,
    getBranchStateVersion
  );

  // Load sessions on mount
  const loadSessions = useCallback(async () => {
    try {
      const response = await fetch("/api/v1/sessions?page_size=50");
      if (response.ok) {
        const data = await response.json();
        setSessions(data.sessions || []);
        setSessionsLoaded(true);
      }
    } catch (error) {
      console.error("Failed to load sessions:", error);
    }
  }, []);

  // Load sessions on first render
  if (!sessionsLoaded) {
    loadSessions();
  }

  const handleSessionCreated = useCallback(
    (newSessionId: number) => {
      navigate(`/chat/${newSessionId}`);
      loadSessions();
    },
    [navigate, loadSessions]
  );

  const { runtime, handleSwitchBranch, getBranchInfoForMessage } = useAegisRuntime({
    sessionId: currentSessionId,
    onSessionCreated: handleSessionCreated,
  });

  const handleNewSession = async () => {
    try {
      // Create new session immediately
      const response = await fetch("/api/v1/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_type: "image_generation" }),
      });
      if (!response.ok) throw new Error("Failed to create session");
      const session = await response.json();
      
      // Add to sessions list at the top
      setSessions((prev) => [session, ...prev]);
      
      // Increment key to force runtime reset
      newSessionKey.current += 1;
      
      // Navigate to new session
      navigate(`/chat/${session.id}`);
    } catch (error) {
      console.error("Failed to create new session:", error);
    }
  };

  const handleDeleteSession = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    // Add to deleting set for animation
    setDeletingIds((prev) => new Set(prev).add(id));
    
    // Wait for animation
    await new Promise((resolve) => setTimeout(resolve, 200));
    
    try {
      await fetch(`/api/v1/sessions/${id}`, { method: "DELETE" });
      // Remove from sessions list immediately (optimistic update)
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (currentSessionId === id) {
        navigate("/chat");
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      // Restore on error
      loadSessions();
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const getSessionTitle = (session: Session) => {
    // Could be enhanced to show first message or task description
    return `Session #${session.id}`;
  };

  return (
    <TooltipProvider>
      <div className="flex h-screen w-full bg-background">
        {/* Sidebar */}
        <aside
          className={cn(
            "flex h-screen flex-col bg-muted/30 transition-all duration-200",
            sidebarOpen ? "w-64 opacity-100" : "w-0 opacity-0 overflow-hidden"
          )}
        >
          {/* Sidebar Header */}
          <div className="flex h-14 shrink-0 items-center px-4">
            <div className="flex items-center gap-2 px-2 font-medium text-sm">
              <span className="size-5 flex items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
                A
              </span>
              <span className="text-foreground/90">Aegis</span>
            </div>
          </div>

          {/* New Thread Button */}
          <div className="shrink-0 px-3 pb-2">
            <button
              type="button"
              onClick={handleNewSession}
              data-active={!currentSessionId}
              className={cn(
                "aui-thread-list-new inline-flex shrink-0 items-center whitespace-nowrap font-medium outline-none transition-all",
                "border bg-background shadow-xs hover:bg-muted",
                "h-9 w-full justify-start gap-2 rounded-lg px-3 text-sm",
                "focus-visible:ring-2 focus-visible:ring-ring/50",
                !currentSessionId && "bg-muted"
              )}
            >
              <Plus className="size-4" />
              New Thread
            </button>
          </div>

          {/* Sessions List - takes all remaining space, pushing Training to bottom */}
          <div className="min-h-0 flex-1 overflow-y-auto px-3">
            <div className="aui-thread-list-root flex flex-col gap-1">
              {/* Session Items */}
              {sessions.map((session) => (
                <div
                  key={session.id}
                  data-active={currentSessionId === session.id}
                  className={cn(
                    "aui-thread-list-item group flex h-9 items-center gap-2 rounded-lg transition-all duration-200",
                    "hover:bg-muted focus-visible:bg-muted focus-visible:outline-none",
                    currentSessionId === session.id && "bg-muted",
                    deletingIds.has(session.id) && "opacity-0 scale-95 -translate-x-2"
                  )}
                >
                  <button
                    type="button"
                    onClick={() => navigate(`/chat/${session.id}`)}
                    className="aui-thread-list-item-trigger flex h-full min-w-0 flex-1 items-center truncate px-3 text-start text-sm"
                  >
                    {getSessionTitle(session)}
                  </button>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        type="button"
                        className={cn(
                          "aui-thread-list-item-more mr-2 size-7 p-0 rounded-md inline-flex items-center justify-center",
                          "transition-opacity",
                          "hover:bg-accent hover:text-accent-foreground",
                          "focus-visible:ring-2 focus-visible:ring-ring/50",
                          isSessionRunning(session.id)
                            ? "opacity-100"
                            : cn(
                                "opacity-0 group-hover:opacity-100",
                                currentSessionId === session.id && "opacity-100"
                              )
                        )}
                      >
                        <MoreHorizontal 
                          className={cn(
                            "size-4",
                            isSessionRunning(session.id) && "animate-bounce"
                          )}
                        />
                        <span className="sr-only">More options</span>
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" side="right">
                      <DropdownMenuItem>
                        <Archive className="size-4" />
                        Archive
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        variant="destructive"
                        onClick={(e) => handleDeleteSession(session.id, e)}
                      >
                        <Trash2 className="size-4" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ))}
            </div>
          </div>

          {/* Sidebar Footer - always at bottom */}
          <div className="shrink-0 p-3 mt-auto">
            <button
              type="button"
              onClick={() => navigate("/training")}
              className={cn(
                "flex h-9 w-full items-center gap-2 rounded-lg px-3 text-sm",
                "transition-colors hover:bg-muted"
              )}
            >
              <Settings className="size-4" />
              Training
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Header */}
          <header className="flex h-14 shrink-0 items-center gap-2 px-4">
            {/* Sidebar Toggle */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-9"
                  onClick={() => setSidebarOpen(!sidebarOpen)}
                >
                  <PanelLeft className="size-4" />
                  <span className="sr-only">
                    {sidebarOpen ? "Hide sidebar" : "Show sidebar"}
                  </span>
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                {sidebarOpen ? "Hide sidebar" : "Show sidebar"}
              </TooltipContent>
            </Tooltip>

            {/* Agent Selector */}
            <AgentSelector
              value={selectedAgent}
              onValueChange={setSelectedAgent}
            />
          </header>

          {/* Chat Thread */}
          <main className="flex-1 overflow-hidden">
            <AssistantRuntimeProvider key={`${currentSessionId ?? "new"}-${newSessionKey.current}`} runtime={runtime}>
              <BranchContext.Provider value={{ handleSwitchBranch, getBranchInfoForMessage }}>
                <Thread />
              </BranchContext.Provider>
            </AssistantRuntimeProvider>
          </main>
        </div>
      </div>
    </TooltipProvider>
  );
}

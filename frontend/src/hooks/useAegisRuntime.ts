/**
 * Custom runtime hook for connecting to Aegis backend.
 * Uses useExternalStoreRuntime pattern to bridge our backend with assistant-ui.
 */

import { useCallback, useState, useEffect, useRef } from "react";
import { useExternalStoreRuntime } from "@assistant-ui/react";
import type { AppendMessage, ThreadMessageLike, TextContentPart, ToolCallContentPart } from "@assistant-ui/react";

const API_BASE_URL = "/api/v1";

interface AegisMessage {
  id: number;
  session_id: number;
  role: "user" | "assistant" | "system";
  content: string;
  plan_json: unknown | null;
  created_at: string;
}

interface UseAegisRuntimeOptions {
  sessionId: number | null;
  onSessionCreated?: (sessionId: number) => void;
}

// Content item for building message content
interface ContentItem {
  type: "text" | "tool-call";
  data: TextContentPart | ToolCallContentPart;
}

// Branch for storing multiple assistant responses to the same user message
interface AssistantBranch {
  id: string;
  message: ThreadMessageLike;
}

// Branch group: all assistant responses for a single user message
interface BranchGroup {
  userMessageId: string;
  branches: AssistantBranch[];
  currentIndex: number;
}

/** SSE `memory_compressed` payload */
export interface CompressionEvent {
  tokens_before: number;
  tokens_after: number;
  original_count: number;
  compressed_count: number;
  ratio: number;
  strategy: string;
}

/** SSE `memory_stats` payload */
export interface MemoryStatsEvent {
  total_tokens: number;
  max_tokens: number;
  usage_ratio: number;
  message_count: number;
  compression_count: number;
  image_url_count: number;
}

/** SSE `api_token_usage` payload — actual API tokens from OpenRouter */
export interface ApiTokenUsageEvent {
  planning: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  skills: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  total: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

// Session state for managing multiple concurrent sessions
interface SessionState {
  eventSource: EventSource | null;
  isRunning: boolean;
  messages: ThreadMessageLike[];
  contentItems: ContentItem[];
  currentThinkingText: string;
  finalImageUrl: string | null;
  assistantMessageId: string | null;
  // Branch management
  branchGroups: Map<string, BranchGroup>; // userMessageId -> branch group
  // Memory compression events
  compressionEvent: CompressionEvent | null;
  memoryStatsEvent: MemoryStatsEvent | null;
  apiTokenUsage: ApiTokenUsageEvent | null;
}

// Global session state manager (persists across component re-renders)
const sessionStates = new Map<number, SessionState>();

// Subscribers for running state changes
type RunningStateListener = () => void;
const runningStateListeners = new Set<RunningStateListener>();
let runningStateVersion = 0;

// Subscribers for branch state changes
type BranchStateListener = () => void;
const branchStateListeners = new Set<BranchStateListener>();
let branchStateVersion = 0;

// Subscribers for compression state changes
type CompressionStateListener = () => void;
const compressionStateListeners = new Set<CompressionStateListener>();
let compressionStateVersion = 0;

function notifyRunningStateChange() {
  runningStateVersion++;
  runningStateListeners.forEach((listener) => listener());
}

function notifyBranchStateChange() {
  branchStateVersion++;
  branchStateListeners.forEach((listener) => listener());
}

function notifyCompressionStateChange() {
  compressionStateVersion++;
  compressionStateListeners.forEach((listener) => listener());
}

export function subscribeToRunningState(listener: RunningStateListener): () => void {
  runningStateListeners.add(listener);
  return () => runningStateListeners.delete(listener);
}

export function getRunningStateVersion(): number {
  return runningStateVersion;
}

export function subscribeToBranchState(listener: BranchStateListener): () => void {
  branchStateListeners.add(listener);
  return () => branchStateListeners.delete(listener);
}

export function getBranchStateVersion(): number {
  return branchStateVersion;
}

export function subscribeToCompressionState(listener: CompressionStateListener): () => void {
  compressionStateListeners.add(listener);
  return () => compressionStateListeners.delete(listener);
}

export function getCompressionStateVersion(): number {
  return compressionStateVersion;
}

export function getCompressionEvent(sessionId: number): CompressionEvent | null {
  return sessionStates.get(sessionId)?.compressionEvent ?? null;
}

export function getMemoryStatsEvent(sessionId: number): MemoryStatsEvent | null {
  return sessionStates.get(sessionId)?.memoryStatsEvent ?? null;
}

export function getApiTokenUsage(sessionId: number): ApiTokenUsageEvent | null {
  return sessionStates.get(sessionId)?.apiTokenUsage ?? null;
}

export function isSessionRunning(sessionId: number): boolean {
  return sessionStates.get(sessionId)?.isRunning ?? false;
}

export function getRunningSessionIds(): number[] {
  const running: number[] = [];
  sessionStates.forEach((state, id) => {
    if (state.isRunning) running.push(id);
  });
  return running;
}

// ── Image config global state ───────────────────────────────────
let _imageAspectRatio = "1:1";
let _imageSize = "1K";

export function setImageAspectRatio(ratio: string) {
  _imageAspectRatio = ratio;
}
export function setImageSize(size: string) {
  _imageSize = size;
}
export function getImageAspectRatio(): string {
  return _imageAspectRatio;
}
export function getImageSize(): string {
  return _imageSize;
}

function getSessionState(sessionId: number): SessionState {
  if (!sessionStates.has(sessionId)) {
    sessionStates.set(sessionId, {
      eventSource: null,
      isRunning: false,
      messages: [],
      contentItems: [],
      currentThinkingText: "",
      finalImageUrl: null,
      assistantMessageId: null,
      branchGroups: new Map(),
      compressionEvent: null,
      memoryStatsEvent: null,
      apiTokenUsage: null,
    });
  }
  return sessionStates.get(sessionId)!;
}

function setSessionRunning(sessionId: number, running: boolean) {
  const state = getSessionState(sessionId);
  if (state.isRunning !== running) {
    state.isRunning = running;
    notifyRunningStateChange();
  }
}

// Branch management functions
export function getBranchInfo(sessionId: number, userMessageId: string): { count: number; current: number } | null {
  const state = sessionStates.get(sessionId);
  if (!state) return null;
  const group = state.branchGroups.get(userMessageId);
  if (!group || group.branches.length <= 1) return null;
  return { count: group.branches.length, current: group.currentIndex + 1 };
}

export function switchBranch(sessionId: number, userMessageId: string, direction: 'prev' | 'next'): { messages: ThreadMessageLike[] | null; indexOnly: boolean } {
  const state = sessionStates.get(sessionId);
  if (!state) return { messages: null, indexOnly: false };
  const group = state.branchGroups.get(userMessageId);
  if (!group || group.branches.length <= 1) return { messages: null, indexOnly: false };
  
  // Check if we're currently generating for this user message
  const isGeneratingForThisMessage = state.isRunning && 
    state.assistantMessageId && 
    group.branches.some(b => b.id === state.assistantMessageId);
  
  if (direction === 'prev' && group.currentIndex > 0) {
    group.currentIndex--;
  } else if (direction === 'next' && group.currentIndex < group.branches.length - 1) {
    group.currentIndex++;
  } else {
    return { messages: null, indexOnly: false }; // No change
  }
  
  // If generating, only update index but don't rebuild messages
  if (isGeneratingForThisMessage) {
    notifyBranchStateChange(); // Update UI to show new index
    return { messages: null, indexOnly: true };
  }
  
  // Rebuild messages with new branch selection
  return { messages: rebuildMessagesWithBranches(state), indexOnly: false };
}

function rebuildMessagesWithBranches(state: SessionState): ThreadMessageLike[] {
  const result: ThreadMessageLike[] = [];
  let i = 0;
  
  while (i < state.messages.length) {
    const msg = state.messages[i];
    result.push(msg);
    
    if (msg.role === 'user') {
      const group = state.branchGroups.get(msg.id!);
      if (group && group.branches.length > 0) {
        // Add the current branch's assistant message
        result.push(group.branches[group.currentIndex].message);
        // Skip all assistant messages that follow this user message in the original array
        i++;
        while (i < state.messages.length && state.messages[i].role === 'assistant') {
          i++;
        }
        continue;
      }
    }
    i++;
  }
  
  return result;
}

// Plan JSON structure from backend

// Plan JSON structure from backend
interface PlanStep {
  type: string;
  data?: {
    thought?: string;
    action?: string;
    action_input?: {
      skill?: string;
      params?: Record<string, unknown>;
      result?: string;
      image_url?: string;
      message?: string;
    };
    observation?: {
      status?: string;
      result?: Record<string, unknown>;
    };
    result?: Record<string, unknown>;
  };
}

interface PlanJson {
  steps?: PlanStep[];
  final_result?: Record<string, unknown>;
}

function convertToThreadMessage(msg: AegisMessage): ThreadMessageLike {
  // For user messages, just return text content
  if (msg.role === "user") {
    return {
      id: String(msg.id),
      role: "user",
      content: [{ type: "text" as const, text: msg.content }],
      createdAt: new Date(msg.created_at),
    };
  }
  
  // For assistant messages, try to parse plan_json for rich content
  const planJson = msg.plan_json as PlanJson | null;
  
  if (planJson?.steps && planJson.steps.length > 0) {
    const parts: (TextContentPart | ToolCallContentPart)[] = [];
    let toolCallIndex = 0;
    
    for (const step of planJson.steps) {
      if (step.type === "thought") {
        const thought = step.data?.thought;
        const actionInput = step.data?.action_input;
        const skillName = actionInput?.skill || step.data?.action || "thinking";
        
        // Add thinking text
        if (thought) {
          parts.push({
            type: "text" as const,
            text: `💭 **Thinking:** ${thought}`,
          });
        }
        
        // Add tool call
        const args = {
          thought: thought || "",
          params: actionInput?.params || {},
        };
        const argsJson = JSON.stringify(args);
        
        parts.push({
          type: "tool-call" as const,
          toolCallId: `tool-${msg.id}-${toolCallIndex++}`,
          toolName: skillName,
          args: JSON.parse(argsJson),
          argsText: argsJson,
        });
      } else if (step.type === "observation") {
        // Update last tool call with result
        const obs = step.data?.observation;
        if (obs && parts.length > 0) {
          // Find last tool call and add result
          for (let i = parts.length - 1; i >= 0; i--) {
            const part = parts[i];
            if (part.type === "tool-call") {
              parts[i] = {
                ...part,
                result: obs.result || obs,
              };
              break;
            }
          }
        }
      } else if (step.type === "finished") {
        // Add completion text
        parts.push({
          type: "text" as const,
          text: `\n✅ **Task completed!**`,
        });
      }
    }
    
    // Add final image if present
    if (planJson.final_result?.image_url) {
      parts.push({
        type: "text" as const,
        text: `![Generated Image](${planJson.final_result.image_url})`,
      });
    }
    
    if (parts.length > 0) {
      return {
        id: String(msg.id),
        role: "assistant",
        content: parts,
        createdAt: new Date(msg.created_at),
      };
    }
  }
  
  // Fallback: just return text content
  return {
    id: String(msg.id),
    role: msg.role === "system" ? "assistant" : msg.role,
    content: [{ type: "text" as const, text: msg.content }],
    createdAt: new Date(msg.created_at),
  };
}

export function useAegisRuntime({ sessionId, onSessionCreated }: UseAegisRuntimeOptions) {
  const [messages, setMessages] = useState<ThreadMessageLike[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<number | null>(sessionId);
  const sessionStateRef = useRef<SessionState | null>(null);
  // Use ref to track current session for SSE callbacks (avoids stale closure)
  const currentSessionIdRef = useRef<number | null>(sessionId);

  // Load messages when session changes
  const loadMessages = useCallback(async (sid: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/${sid}/messages?limit=100`);
      if (!response.ok) throw new Error("Failed to load messages");
      const data = await response.json();
      const threadMessages = (data.messages || []).map(convertToThreadMessage);
      setMessages(threadMessages);
      
      // Update session state
      const state = getSessionState(sid);
      state.messages = threadMessages;
    } catch (error) {
      console.error("Failed to load messages:", error);
    }
  }, []);

  // Create new session if needed
  const ensureSession = useCallback(async (): Promise<number> => {
    if (currentSessionId) return currentSessionId;

    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_type: "image_generation" }),
      });
      if (!response.ok) throw new Error("Failed to create session");
      const session = await response.json();
      setCurrentSessionId(session.id);
      onSessionCreated?.(session.id);
      return session.id;
    } catch (error) {
      console.error("Failed to create session:", error);
      throw error;
    }
  }, [currentSessionId, onSessionCreated]);

  // Handle new message submission
  const onNew = useCallback(
    async (message: AppendMessage) => {
      if (message.content.length === 0) return;

      const textContent = message.content
        .filter((part): part is { type: "text"; text: string } => part.type === "text")
        .map((part) => part.text)
        .join("\n");

      if (!textContent.trim()) return;

      setIsRunning(true);

      try {
        const sid = await ensureSession();
        const state = getSessionState(sid);
        setSessionRunning(sid, true);
        state.contentItems = [];
        state.currentThinkingText = "";
        state.finalImageUrl = null;
        sessionStateRef.current = state;

        // Add user message optimistically
        const userMessage: ThreadMessageLike = {
          id: `temp-${Date.now()}`,
          role: "user",
          content: [{ type: "text" as const, text: textContent }],
          createdAt: new Date(),
        };
        setMessages((prev) => [...prev, userMessage]);
        state.messages = [...state.messages, userMessage];

        // Call streaming chat endpoint (include image config)
        const ar = encodeURIComponent(getImageAspectRatio());
        const sz = encodeURIComponent(getImageSize());
        const eventSource = new EventSource(
          `${API_BASE_URL}/${sid}/chat/stream?message=${encodeURIComponent(textContent)}&aspect_ratio=${ar}&image_size=${sz}`
        );
        state.eventSource = eventSource;

        const assistantMessageId = `assistant-${Date.now()}`;
        state.assistantMessageId = assistantMessageId;

        // Helper to build content array from content items
        const buildContent = (): (TextContentPart | ToolCallContentPart)[] => {
          const parts: (TextContentPart | ToolCallContentPart)[] = [];
          
          for (const item of state.contentItems) {
            parts.push(item.data);
          }
          
          // Add current thinking text if any
          if (state.currentThinkingText) {
            parts.push({
              type: "text" as const,
              text: state.currentThinkingText,
            });
          }
          
          // Add final result image if available
          if (state.finalImageUrl) {
            parts.push({
              type: "text" as const,
              text: `![Generated Image](${state.finalImageUrl})`,
            });
          }
          
          return parts.length > 0 ? parts : [{ type: "text" as const, text: "" }];
        };

        // Add placeholder for assistant message
        const assistantMessage: ThreadMessageLike = {
          id: assistantMessageId,
          role: "assistant" as const,
          content: buildContent(),
          createdAt: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        state.messages = [...state.messages, assistantMessage];

        // Update messages helper
        const updateMessages = () => {
          const newMessages = state.messages.map((m) =>
            m.id === assistantMessageId
              ? { ...m, content: buildContent() }
              : m
          );
          state.messages = newMessages;
          // Only update UI if this is the current session (use ref for latest value)
          if (currentSessionIdRef.current === sid) {
            setMessages(newMessages);
          }
        };

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === "thinking") {
              state.currentThinkingText = "🤔 Thinking...";
              updateMessages();
            } else if (data.type === "thought_delta") {
              // Incremental token from the LLM — typewriter effect
              const delta: string = data.data?.delta ?? "";
              if (state.currentThinkingText === "🤔 Thinking...") {
                // Replace placeholder with first real token
                state.currentThinkingText = "💭 " + delta;
              } else {
                state.currentThinkingText += delta;
              }
              updateMessages();
            } else if (data.type === "thought") {
              state.currentThinkingText = "";
              
              const { thought, action, action_input } = data.data || {};
              const skillName = action_input?.skill || action || "thinking";
              const toolCallId = `tool-${Date.now()}-${state.contentItems.length}`;
              
              if (thought) {
                state.contentItems.push({
                  type: "text",
                  data: {
                    type: "text" as const,
                    text: `💭 **Thinking:** ${thought}`,
                  },
                });
              }
              
              const args = {
                thought: thought || "",
                params: action_input?.params || {},
              };
              const argsJson = JSON.stringify(args);
              
              state.contentItems.push({
                type: "tool-call",
                data: {
                  type: "tool-call" as const,
                  toolCallId,
                  toolName: skillName,
                  args: JSON.parse(argsJson),
                  argsText: argsJson,
                },
              });
              
              updateMessages();
            } else if (data.type === "executing") {
              updateMessages();
            } else if (data.type === "observation") {
              const obs = data.data?.observation || {};
              const lastToolCallItem = [...state.contentItems].reverse().find(item => item.type === "tool-call");
              if (lastToolCallItem && lastToolCallItem.data.type === "tool-call") {
                const tc = lastToolCallItem.data as ToolCallContentPart;
                lastToolCallItem.data = {
                  ...tc,
                  result: obs.result || obs.error || "Completed",
                };
              }
              updateMessages();
            } else if (data.type === "finished") {
              const result = data.data?.result || {};
              if (result.image_url) {
                state.finalImageUrl = result.image_url;
              }
              state.contentItems.push({
                type: "text",
                data: {
                  type: "text" as const,
                  text: `\n✅ **Task completed!**`,
                },
              });
              updateMessages();
            } else if (data.type === "completed") {
              eventSource.close();
              state.eventSource = null;
              setSessionRunning(sid, false);
              // Only update UI if this is the current session (use ref for latest value)
              if (currentSessionIdRef.current === sid) {
                setIsRunning(false);
              }
              // Reload messages from server to get persisted data
              setTimeout(() => loadMessages(sid), 1000);
            } else if (data.type === "memory_compressed") {
              state.compressionEvent = data.data as CompressionEvent;
              notifyCompressionStateChange();
            } else if (data.type === "memory_stats") {
              state.memoryStatsEvent = data.data as MemoryStatsEvent;
              notifyCompressionStateChange();
            } else if (data.type === "api_token_usage") {
              state.apiTokenUsage = data.data as ApiTokenUsageEvent;
              notifyCompressionStateChange();
            } else if (data.type === "error") {
              const errorMessages = state.messages.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, content: [{ type: "text" as const, text: `Error: ${data.data?.message || "Unknown error"}` }] }
                  : m
              );
              state.messages = errorMessages;
              if (currentSessionIdRef.current === sid) {
                setMessages(errorMessages);
              }
              eventSource.close();
              state.eventSource = null;
              setSessionRunning(sid, false);
              if (currentSessionIdRef.current === sid) {
                setIsRunning(false);
              }
            }
          } catch (e) {
            console.error("Error parsing SSE data:", e);
          }
        };

        eventSource.onerror = () => {
          eventSource.close();
          state.eventSource = null;
          setSessionRunning(sid, false);
          if (currentSessionIdRef.current === sid) {
            setIsRunning(false);
            loadMessages(sid);
          }
        };
      } catch (error) {
        console.error("Failed to send message:", error);
        setIsRunning(false);
      }
    },
    [ensureSession, loadMessages]
  );

  // Reset session when sessionId prop changes
  useEffect(() => {
    // Always update ref to latest sessionId
    currentSessionIdRef.current = sessionId;
    
    if (sessionId === null) {
      // New chat - reset state
      setCurrentSessionId(null);
      setMessages([]);
      setIsRunning(false);
      sessionStateRef.current = null;
    } else if (sessionId !== currentSessionId) {
      // Switching to different session
      setCurrentSessionId(sessionId);
      
      // Check if the session has running state
      const state = getSessionState(sessionId);
      sessionStateRef.current = state;
      
      if (state.isRunning && state.eventSource) {
        // Session is still running, restore state
        setMessages(state.messages);
        setIsRunning(true);
      } else {
        // Load messages from server
        setIsRunning(false);
        loadMessages(sessionId);
      }
    }
  }, [sessionId]); // Only depend on sessionId to trigger on navigation

  // Handle cancel/stop generation
  const onCancel = useCallback(async () => {
    if (currentSessionId) {
      const state = getSessionState(currentSessionId);
      if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
      }
      setSessionRunning(currentSessionId, false);
    }
    setIsRunning(false);
  }, [currentSessionId]);

  // Handle reload/regenerate - creates a new branch instead of replacing
  const onReload = useCallback(
    async (parentId: string | null) => {
      if (!currentSessionId) return;
      
      // Find the user message that we want to regenerate from
      const parentIndex = messages.findIndex((m) => m.id === parentId);
      if (parentIndex === -1) return;
      
      const parentMessage = messages[parentIndex];
      if (parentMessage.role !== "user") return;
      
      const userMessageId = parentMessage.id!;
      
      // Get the text content from the user message
      let textContent = "";
      const content = parentMessage.content;
      if (typeof content === "string") {
        textContent = content;
      } else if (Array.isArray(content)) {
        textContent = content
          .filter((part): part is TextContentPart => part.type === "text")
          .map((part) => part.text)
          .join("\n");
      }
      
      if (!textContent.trim()) return;
      
      const sid = currentSessionId;
      const state = getSessionState(sid);
      
      // Save existing assistant message to branch group if not already saved
      const existingAssistantMsg = messages[parentIndex + 1];
      if (existingAssistantMsg && existingAssistantMsg.role === "assistant") {
        let group = state.branchGroups.get(userMessageId);
        if (!group) {
          group = {
            userMessageId,
            branches: [],
            currentIndex: 0,
          };
          state.branchGroups.set(userMessageId, group);
        }
        
        // Add existing message to branches if not already there
        const existingBranchIndex = group.branches.findIndex(b => b.id === existingAssistantMsg.id);
        if (existingBranchIndex === -1) {
          group.branches.push({
            id: existingAssistantMsg.id!,
            message: existingAssistantMsg,
          });
        }
      }
      
      setIsRunning(true);
      setSessionRunning(sid, true);
      state.contentItems = [];
      state.currentThinkingText = "";
      state.finalImageUrl = null;
      sessionStateRef.current = state;

      // Call streaming chat endpoint (include image config)
      const ar = encodeURIComponent(getImageAspectRatio());
      const sz = encodeURIComponent(getImageSize());
      const eventSource = new EventSource(
        `${API_BASE_URL}/${sid}/chat/stream?message=${encodeURIComponent(textContent)}&aspect_ratio=${ar}&image_size=${sz}`
      );
      state.eventSource = eventSource;

      const assistantMessageId = `assistant-branch-${Date.now()}`;
      state.assistantMessageId = assistantMessageId;

      // Helper to build content array from content items
      const buildContent = (): (TextContentPart | ToolCallContentPart)[] => {
        const parts: (TextContentPart | ToolCallContentPart)[] = [];
        
        for (const item of state.contentItems) {
          parts.push(item.data);
        }
        
        if (state.currentThinkingText) {
          parts.push({
            type: "text" as const,
            text: state.currentThinkingText,
          });
        }
        
        if (state.finalImageUrl) {
          parts.push({
            type: "text" as const,
            text: `![Generated Image](${state.finalImageUrl})`,
          });
        }
        
        return parts.length > 0 ? parts : [{ type: "text" as const, text: "" }];
      };

      // Create new assistant message for the new branch
      const newAssistantMessage: ThreadMessageLike = {
        id: assistantMessageId,
        role: "assistant" as const,
        content: buildContent(),
        createdAt: new Date(),
      };
      
      // Add new branch and switch to it
      let group = state.branchGroups.get(userMessageId);
      if (!group) {
        group = {
          userMessageId,
          branches: [],
          currentIndex: 0,
        };
        state.branchGroups.set(userMessageId, group);
      }
      group.branches.push({
        id: assistantMessageId,
        message: newAssistantMessage,
      });
      group.currentIndex = group.branches.length - 1;
      
      // Notify branch state change
      notifyBranchStateChange();
      
      // Build messages array: keep messages up to and including user message, then add new assistant
      const newMessages = [...messages.slice(0, parentIndex + 1), newAssistantMessage];
      state.messages = newMessages;
      setMessages(newMessages);

      // Update messages helper
      const updateMessages = () => {
        // Update the branch's message
        const branchGroup = state.branchGroups.get(userMessageId);
        if (branchGroup) {
          const branchIndex = branchGroup.branches.findIndex(b => b.id === assistantMessageId);
          if (branchIndex !== -1) {
            branchGroup.branches[branchIndex].message = {
              ...branchGroup.branches[branchIndex].message,
              content: buildContent(),
            };
          }
        }
        
        const updatedMessages = state.messages.map((m) =>
          m.id === assistantMessageId
            ? { ...m, content: buildContent() }
            : m
        );
        state.messages = updatedMessages;
        if (currentSessionIdRef.current === sid) {
          setMessages(updatedMessages);
        }
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "thinking") {
            state.currentThinkingText = "🤔 Thinking...";
            updateMessages();
          } else if (data.type === "thought") {
            state.currentThinkingText = "";
            
            const { thought, action, action_input } = data.data || {};
            const skillName = action_input?.skill || action || "thinking";
            const toolCallId = `tool-${Date.now()}-${state.contentItems.length}`;
            
            if (thought) {
              state.contentItems.push({
                type: "text",
                data: {
                  type: "text" as const,
                  text: `💭 **Thinking:** ${thought}`,
                },
              });
            }
            
            const args = {
              thought: thought || "",
              params: action_input?.params || {},
            };
            const argsJson = JSON.stringify(args);
            
            state.contentItems.push({
              type: "tool-call",
              data: {
                type: "tool-call" as const,
                toolCallId,
                toolName: skillName,
                args: JSON.parse(argsJson),
                argsText: argsJson,
              },
            });
            
            updateMessages();
          } else if (data.type === "executing") {
            updateMessages();
          } else if (data.type === "observation") {
            const obs = data.data?.observation || {};
            const lastToolCallItem = [...state.contentItems].reverse().find(item => item.type === "tool-call");
            if (lastToolCallItem && lastToolCallItem.data.type === "tool-call") {
              const tc = lastToolCallItem.data as ToolCallContentPart;
              lastToolCallItem.data = {
                ...tc,
                result: obs.result || obs.error || "Completed",
              };
            }
            updateMessages();
          } else if (data.type === "finished") {
            const result = data.data?.result || {};
            if (result.image_url) {
              state.finalImageUrl = result.image_url;
            }
            state.contentItems.push({
              type: "text",
              data: {
                type: "text" as const,
                text: `\n✅ **Task completed!**`,
              },
            });
            updateMessages();
          } else if (data.type === "completed") {
            eventSource.close();
            state.eventSource = null;
            setSessionRunning(sid, false);
            if (currentSessionIdRef.current === sid) {
              setIsRunning(false);
            }
            // Notify branch state change to update branch picker UI
            notifyBranchStateChange();
          } else if (data.type === "memory_compressed") {
            state.compressionEvent = data.data as CompressionEvent;
            notifyCompressionStateChange();
          } else if (data.type === "memory_stats") {
            state.memoryStatsEvent = data.data as MemoryStatsEvent;
            notifyCompressionStateChange();
          } else if (data.type === "error") {
            const errorMessages = state.messages.map((m) =>
              m.id === assistantMessageId
                ? { ...m, content: [{ type: "text" as const, text: `Error: ${data.data?.message || "Unknown error"}` }] }
                : m
            );
            state.messages = errorMessages;
            if (currentSessionIdRef.current === sid) {
              setMessages(errorMessages);
            }
            eventSource.close();
            state.eventSource = null;
            setSessionRunning(sid, false);
            if (currentSessionIdRef.current === sid) {
              setIsRunning(false);
            }
          }
        } catch (e) {
          console.error("Error parsing SSE data:", e);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        state.eventSource = null;
        setSessionRunning(sid, false);
        if (currentSessionIdRef.current === sid) {
          setIsRunning(false);
        }
      };
    },
    [currentSessionId, messages]
  );
  
  // Handle branch switching
  const handleSwitchBranch = useCallback((userMessageId: string, direction: 'prev' | 'next') => {
    if (!currentSessionId) return;
    const result = switchBranch(currentSessionId, userMessageId, direction);
    
    // If indexOnly is true, the index was updated but we shouldn't change messages (generating)
    if (result.indexOnly) {
      return;
    }
    
    if (result.messages) {
      const state = getSessionState(currentSessionId);
      state.messages = result.messages;
      setMessages(result.messages);
    }
  }, [currentSessionId]);
  
  // Get branch info for a specific user message
  const getBranchInfoForMessage = useCallback((userMessageId: string) => {
    if (!currentSessionId) return null;
    return getBranchInfo(currentSessionId, userMessageId);
  }, [currentSessionId]);

  // setMessages callback for branch switching
  const handleSetMessages = useCallback((newMessages: ThreadMessageLike[]) => {
    if (!currentSessionId) return;
    const state = getSessionState(currentSessionId);
    state.messages = newMessages;
    setMessages(newMessages);
  }, [currentSessionId]);

  const runtime = useExternalStoreRuntime({
    messages,
    setMessages: handleSetMessages,
    isRunning,
    onNew,
    onCancel,
    onReload,
    convertMessage: (message: ThreadMessageLike) => message,
  });

  return {
    runtime,
    handleSwitchBranch,
    getBranchInfoForMessage,
  };
}

export type { UseAegisRuntimeOptions };

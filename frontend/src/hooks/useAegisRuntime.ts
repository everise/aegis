/**
 * Custom runtime hook for connecting to Aegis backend.
 * Uses useExternalStoreRuntime pattern to bridge our backend with assistant-ui.
 */

import { useCallback, useState, useEffect } from "react";
import { useExternalStoreRuntime } from "@assistant-ui/react";
import type { AppendMessage, ThreadMessageLike } from "@assistant-ui/react";

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

function convertToThreadMessage(msg: AegisMessage): ThreadMessageLike {
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

  // Load messages when session changes
  const loadMessages = useCallback(async (sid: number) => {
    try {
      const response = await fetch(`${API_BASE_URL}/${sid}/messages?limit=100`);
      if (!response.ok) throw new Error("Failed to load messages");
      const data = await response.json();
      const threadMessages = (data.messages || []).map(convertToThreadMessage);
      setMessages(threadMessages);
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

        // Add user message optimistically
        const userMessage: ThreadMessageLike = {
          id: `temp-${Date.now()}`,
          role: "user",
          content: [{ type: "text" as const, text: textContent }],
          createdAt: new Date(),
        };
        setMessages((prev) => [...prev, userMessage]);

        // Call streaming chat endpoint
        const eventSource = new EventSource(
          `${API_BASE_URL}/${sid}/chat/stream?message=${encodeURIComponent(textContent)}`
        );

        const assistantMessageId = `assistant-${Date.now()}`;
        let assistantContent = "";

        // Add placeholder for assistant message
        setMessages((prev) => [
          ...prev,
          {
            id: assistantMessageId,
            role: "assistant" as const,
            content: [{ type: "text" as const, text: "" }],
            createdAt: new Date(),
          },
        ]);

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === "step_update" || data.type === "thinking") {
              const thought = data.data?.thought || data.data?.step?.thought || "";
              const action = data.data?.action || data.data?.step?.action || "";
              const observation = data.data?.observation || "";

              let content = "";
              if (thought) content += `**Thinking:** ${thought}\n\n`;
              if (action) content += `**Action:** ${action}\n\n`;
              if (observation) content += `**Observation:** ${JSON.stringify(observation)}\n\n`;

              if (content) {
                assistantContent = content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, content: [{ type: "text" as const, text: assistantContent }] }
                      : m
                  )
                );
              }
            } else if (data.type === "completed") {
              const result = data.data?.final_result || data.data?.result;
              if (result) {
                let finalContent = assistantContent;
                if (result.image_url) {
                  finalContent += `\n\n![Generated Image](${result.image_url})`;
                }
                if (result.message) {
                  finalContent += `\n\n${result.message}`;
                }
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, content: [{ type: "text" as const, text: finalContent || "Task completed." }] }
                      : m
                  )
                );
              }
              eventSource.close();
              setIsRunning(false);
            } else if (data.type === "error") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: [{ type: "text" as const, text: `Error: ${data.data?.message || "Unknown error"}` }] }
                    : m
                )
              );
              eventSource.close();
              setIsRunning(false);
            }
          } catch (e) {
            console.error("Error parsing SSE data:", e);
          }
        };

        eventSource.onerror = () => {
          eventSource.close();
          setIsRunning(false);
          loadMessages(sid);
        };
      } catch (error) {
        console.error("Failed to send message:", error);
        setIsRunning(false);
      }
    },
    [ensureSession, loadMessages]
  );

  // Update when sessionId prop changes
  useEffect(() => {
    if (sessionId && sessionId !== currentSessionId) {
      setCurrentSessionId(sessionId);
      loadMessages(sessionId);
    }
  }, [sessionId, currentSessionId, loadMessages]);

  const runtime = useExternalStoreRuntime({
    messages,
    isRunning,
    onNew,
    convertMessage: (message: ThreadMessageLike) => message,
  });

  return runtime;
}

export type { UseAegisRuntimeOptions };

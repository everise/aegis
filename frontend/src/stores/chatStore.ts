/**
 * Zustand store for chat state management.
 */

import { create } from 'zustand';
import type { Session, Message, ExecutionPlan, PlanStep } from '@/types';
import { sessionsApi, messagesApi } from '@/services/api';

interface ChatState {
  // Current session
  currentSession: Session | null;
  sessions: Session[];
  
  // Messages
  messages: Message[];
  isLoadingMessages: boolean;
  
  // Planning state
  currentPlan: ExecutionPlan | null;
  isPlanning: boolean;
  planningStep: PlanStep | null;
  
  // SSE connection
  eventSource: EventSource | null;
  
  // Actions
  createSession: (taskType?: string) => Promise<Session>;
  loadSessions: () => Promise<void>;
  selectSession: (sessionId: number) => Promise<void>;
  deleteSession: (sessionId: number) => Promise<void>;
  
  loadMessages: (sessionId: number) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  sendMessageStream: (content: string) => void;
  
  // SSE handlers
  handleSSEEvent: (event: MessageEvent) => void;
  closeEventSource: () => void;
  
  // Reset
  reset: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  // Initial state
  currentSession: null,
  sessions: [],
  messages: [],
  isLoadingMessages: false,
  currentPlan: null,
  isPlanning: false,
  planningStep: null,
  eventSource: null,

  // Session actions
  createSession: async (taskType = 'text_to_image') => {
    const session = await sessionsApi.create(taskType);
    set((state) => ({
      sessions: [session, ...state.sessions],
      currentSession: session,
      messages: [],
      currentPlan: null,
    }));
    return session;
  },

  loadSessions: async () => {
    const response = await sessionsApi.list();
    set({ sessions: response.sessions });
  },

  selectSession: async (sessionId: number) => {
    const session = await sessionsApi.get(sessionId);
    set({ currentSession: session, messages: [], currentPlan: null });
    await get().loadMessages(sessionId);
  },

  deleteSession: async (sessionId: number) => {
    await sessionsApi.delete(sessionId);
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== sessionId),
      currentSession: state.currentSession?.id === sessionId ? null : state.currentSession,
      messages: state.currentSession?.id === sessionId ? [] : state.messages,
    }));
  },

  // Message actions
  loadMessages: async (sessionId: number) => {
    set({ isLoadingMessages: true });
    try {
      const response = await messagesApi.list(sessionId);
      set({ messages: response.messages });
    } finally {
      set({ isLoadingMessages: false });
    }
  },

  sendMessage: async (content: string) => {
    const { currentSession } = get();
    if (!currentSession) return;

    // Add user message optimistically
    const tempUserMessage: Message = {
      id: Date.now(),
      session_id: currentSession.id,
      role: 'user',
      content,
      plan_json: null,
      created_at: new Date().toISOString(),
    };
    set((state) => ({ messages: [...state.messages, tempUserMessage] }));

    set({ isPlanning: true });
    try {
      const response = await messagesApi.chat(currentSession.id, content);
      
      // Add assistant message with plan
      const assistantMessage: Message = {
        id: response.message_id,
        session_id: currentSession.id,
        role: 'assistant',
        content: response.plan.final_result?.image_url as string || 'Task completed',
        plan_json: response.plan,
        created_at: new Date().toISOString(),
      };
      
      set((state) => ({
        messages: [...state.messages, assistantMessage],
        currentPlan: response.plan,
      }));
    } finally {
      set({ isPlanning: false });
    }
  },

  sendMessageStream: (content: string) => {
    const { currentSession, closeEventSource } = get();
    if (!currentSession) return;

    // Close existing connection
    closeEventSource();

    // Add user message
    const tempUserMessage: Message = {
      id: Date.now(),
      session_id: currentSession.id,
      role: 'user',
      content,
      plan_json: null,
      created_at: new Date().toISOString(),
    };
    set((state) => ({ 
      messages: [...state.messages, tempUserMessage],
      isPlanning: true,
      currentPlan: {
        session_id: currentSession.id,
        user_message: content,
        steps: [],
        status: 'thinking',
        final_result: null,
      },
    }));

    // Create SSE connection
    const eventSource = messagesApi.chatStream(currentSession.id, content);
    
    eventSource.onmessage = get().handleSSEEvent;
    
    eventSource.onerror = () => {
      set({ isPlanning: false });
      eventSource.close();
    };

    set({ eventSource });
  },

  handleSSEEvent: (event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      const eventType = data.type || event.type;

      switch (eventType) {
        case 'thinking':
          set({ planningStep: { step_number: data.step_number, thought: '', action: '', action_input: {}, observation: null, status: 'thinking', error: null } });
          break;

        case 'thought':
          set((state) => ({
            currentPlan: state.currentPlan ? {
              ...state.currentPlan,
              steps: [...state.currentPlan.steps, {
                step_number: data.step_number,
                thought: data.thought,
                action: data.action,
                action_input: data.action_input,
                observation: null,
                status: 'executing',
                error: null,
              }],
            } : null,
            planningStep: {
              step_number: data.step_number,
              thought: data.thought,
              action: data.action,
              action_input: data.action_input,
              observation: null,
              status: 'executing',
              error: null,
            },
          }));
          break;

        case 'observation':
          set((state) => ({
            currentPlan: state.currentPlan ? {
              ...state.currentPlan,
              steps: state.currentPlan.steps.map((step) =>
                step.step_number === data.step_number
                  ? { ...step, observation: data.observation, status: 'observing' }
                  : step
              ),
            } : null,
          }));
          break;

        case 'finished':
          set((state) => ({
            currentPlan: state.currentPlan ? {
              ...state.currentPlan,
              status: 'completed',
              final_result: data.result,
            } : null,
            isPlanning: false,
          }));
          get().closeEventSource();
          
          // Add assistant message
          const { currentSession, currentPlan } = get();
          if (currentSession && currentPlan) {
            const assistantMessage: Message = {
              id: Date.now(),
              session_id: currentSession.id,
              role: 'assistant',
              content: data.result?.image_url || 'Task completed',
              plan_json: currentPlan,
              created_at: new Date().toISOString(),
            };
            set((state) => ({ messages: [...state.messages, assistantMessage] }));
          }
          break;

        case 'error':
          set((state) => ({
            currentPlan: state.currentPlan ? {
              ...state.currentPlan,
              status: 'failed',
            } : null,
            isPlanning: false,
          }));
          get().closeEventSource();
          break;
      }
    } catch (e) {
      console.error('Error handling SSE event:', e);
    }
  },

  closeEventSource: () => {
    const { eventSource } = get();
    if (eventSource) {
      eventSource.close();
      set({ eventSource: null });
    }
  },

  reset: () => {
    get().closeEventSource();
    set({
      currentSession: null,
      messages: [],
      currentPlan: null,
      isPlanning: false,
      planningStep: null,
    });
  },
}));

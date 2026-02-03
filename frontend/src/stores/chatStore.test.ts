/**
 * Tests for chat store.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useChatStore } from '@/stores/chatStore';

// Mock the API
vi.mock('@/services/api', () => ({
  sessionsApi: {
    create: vi.fn().mockResolvedValue({
      id: 1,
      status: 'active',
      task_type: 'text_to_image',
      metadata: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      message_count: 0,
    }),
    list: vi.fn().mockResolvedValue({ sessions: [], total: 0, page: 1, page_size: 20 }),
    get: vi.fn().mockResolvedValue({
      id: 1,
      status: 'active',
      task_type: 'text_to_image',
      metadata: {},
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      message_count: 0,
    }),
    delete: vi.fn().mockResolvedValue(undefined),
  },
  messagesApi: {
    list: vi.fn().mockResolvedValue({ messages: [], total: 0 }),
    chat: vi.fn().mockResolvedValue({
      message_id: 1,
      plan: { steps: [], status: 'completed', final_result: null },
      status: 'completed',
    }),
  },
}));

describe('useChatStore', () => {
  beforeEach(() => {
    useChatStore.getState().reset();
  });

  it('should have initial state', () => {
    const state = useChatStore.getState();
    
    expect(state.currentSession).toBeNull();
    expect(state.sessions).toEqual([]);
    expect(state.messages).toEqual([]);
    expect(state.isPlanning).toBe(false);
  });

  it('should create a session', async () => {
    const { createSession } = useChatStore.getState();
    
    const session = await createSession();
    
    expect(session.id).toBe(1);
    expect(useChatStore.getState().currentSession).toEqual(session);
  });

  it('should load sessions', async () => {
    const { loadSessions } = useChatStore.getState();
    
    await loadSessions();
    
    expect(useChatStore.getState().sessions).toEqual([]);
  });

  it('should reset state', () => {
    const store = useChatStore.getState();
    
    // Set some state
    useChatStore.setState({
      currentSession: { id: 1 } as any,
      messages: [{ id: 1 }] as any,
    });
    
    store.reset();
    
    const newState = useChatStore.getState();
    expect(newState.currentSession).toBeNull();
    expect(newState.messages).toEqual([]);
  });
});

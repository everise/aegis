/**
 * API client service for Aegis backend.
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig, AxiosResponse } from 'axios';
import type {
  Session,
  SessionListResponse,
  Message,
  MessageListResponse,
  ExecutionPlan,
  SkillInfo,
  SkillExecution,
  TrainingJob,
  TrainingStatus,
  TrainingConfig,
  Prompt,
  PromptListResponse,
} from '@/types';

const API_BASE_URL = '/api/v1';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error: unknown) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token');
    }
    return Promise.reject(error);
  }
);

// Sessions API
export const sessionsApi = {
  create: async (taskType?: string): Promise<Session> => {
    const response = await api.post<Session>('/sessions', { task_type: taskType });
    return response.data;
  },

  list: async (page = 1, pageSize = 20): Promise<SessionListResponse> => {
    const response = await api.get<SessionListResponse>('/sessions', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  get: async (sessionId: number): Promise<Session> => {
    const response = await api.get<Session>(`/sessions/${sessionId}`);
    return response.data;
  },

  delete: async (sessionId: number): Promise<void> => {
    await api.delete(`/sessions/${sessionId}`);
  },

  complete: async (sessionId: number): Promise<void> => {
    await api.post(`/sessions/${sessionId}/complete`);
  },
};

// Messages API
export const messagesApi = {
  list: async (sessionId: number, limit = 50): Promise<MessageListResponse> => {
    const response = await api.get<MessageListResponse>(`/${sessionId}/messages`, {
      params: { limit },
    });
    return response.data;
  },

  create: async (sessionId: number, content: string): Promise<Message> => {
    const response = await api.post<Message>(`/${sessionId}/messages`, { content });
    return response.data;
  },

  chat: async (sessionId: number, message: string): Promise<{ message_id: number; plan: ExecutionPlan; status: string }> => {
    const response = await api.post<{ message_id: number; plan: ExecutionPlan; status: string }>(`/${sessionId}/chat`, { content: message });
    return response.data;
  },

  chatStream: (sessionId: number, message: string): EventSource => {
    const url = `${API_BASE_URL}/${sessionId}/chat/stream?message=${encodeURIComponent(message)}`;
    return new EventSource(url);
  },
};

// Skills API
export const skillsApi = {
  list: async (): Promise<{ skills: SkillInfo[]; total: number }> => {
    const response = await api.get<{ skills: SkillInfo[]; total: number }>('/skills');
    return response.data;
  },

  get: async (skillName: string): Promise<SkillInfo> => {
    const response = await api.get<SkillInfo>(`/skills/${skillName}`);
    return response.data;
  },

  execute: async (
    skillName: string,
    params: Record<string, unknown>,
    messageId?: number
  ): Promise<SkillExecution> => {
    const response = await api.post<SkillExecution>('/skills/execute', {
      skill_name: skillName,
      params,
      message_id: messageId,
    });
    return response.data;
  },

  getExecution: async (executionId: number): Promise<SkillExecution> => {
    const response = await api.get<SkillExecution>(`/skills/executions/${executionId}`);
    return response.data;
  },
};

// Training API
export const trainingApi = {
  createJob: async (config?: Partial<TrainingConfig>): Promise<TrainingJob> => {
    const response = await api.post<TrainingJob>('/training/jobs', { config });
    return response.data;
  },

  listJobs: async (limit = 20): Promise<TrainingJob[]> => {
    const response = await api.get<TrainingJob[]>('/training/jobs', { params: { limit } });
    return response.data;
  },

  getJob: async (jobId: number): Promise<TrainingJob> => {
    const response = await api.get<TrainingJob>(`/training/jobs/${jobId}`);
    return response.data;
  },

  startJob: async (jobId: number): Promise<void> => {
    await api.post(`/training/jobs/${jobId}/start`);
  },

  pauseJob: async (jobId: number): Promise<void> => {
    await api.post(`/training/jobs/${jobId}/pause`);
  },

  resumeJob: async (jobId: number): Promise<void> => {
    await api.post(`/training/jobs/${jobId}/resume`);
  },

  cancelJob: async (jobId: number): Promise<void> => {
    await api.post(`/training/jobs/${jobId}/cancel`);
  },

  getStatus: async (): Promise<TrainingStatus> => {
    const response = await api.get<TrainingStatus>('/training/status');
    return response.data;
  },

  getMetrics: async (lastN = 100): Promise<{ metrics: Record<string, unknown>[]; count: number }> => {
    const response = await api.get<{ metrics: Record<string, unknown>[]; count: number }>('/training/metrics', { params: { last_n: lastN } });
    return response.data;
  },

  getBufferStats: async (): Promise<Record<string, unknown>> => {
    const response = await api.get<Record<string, unknown>>('/training/buffer/stats');
    return response.data;
  },
};

export default api;

// Context API
export interface ContextStatsResponse {
  session_id: number;
  total_vectors: number;
  user_message_count: number;
  assistant_message_count: number;
  system_message_count: number;
  total_tokens_estimate: number;
  context_window_usage: number;
  max_context_tokens: number;
  oldest_message_time: string | null;
  newest_message_time: string | null;
  collection_name: string;
  similar_context_count: number;
}

export const contextApi = {
  getStats: async (sessionId: number): Promise<ContextStatsResponse> => {
    const response = await api.get<ContextStatsResponse>(`/context/${sessionId}/stats`);
    return response.data;
  },
  getMemoryStats: async (sessionId: number): Promise<MemoryStatsResponse> => {
    const response = await api.get<MemoryStatsResponse>(`/context/${sessionId}/memory/stats`);
    return response.data;
  },
};

// Memory Stats API (working memory with compression)
export interface MemoryStatsResponse {
  session_id: number;
  message_count: number;
  compressed_count: number;
  total_tokens: number;
  max_tokens: number;
  usage_ratio: number;
  image_url_count: number;
  compression_count: number;
}

// Planning Models API
export interface PlanningModel {
  id: string;
  name: string;
  provider: string;
  description: string;
  supports_vision: boolean;
  supports_streaming: boolean;
}

export interface PlanningModelListResponse {
  models: PlanningModel[];
  active_model_id: string | null;
}

export interface SetActiveModelResponse {
  active_model_id: string;
  model: PlanningModel;
}

export const planningModelsApi = {
  list: async (): Promise<PlanningModelListResponse> => {
    const response = await api.get<PlanningModelListResponse>('/planning-models');
    return response.data;
  },

  getActive: async (): Promise<PlanningModel> => {
    const response = await api.get<PlanningModel>('/planning-models/active');
    return response.data;
  },

  setActive: async (modelId: string): Promise<SetActiveModelResponse> => {
    const response = await api.put<SetActiveModelResponse>('/planning-models/active', {
      model_id: modelId,
    });
    return response.data;
  },
};

// Prompts API
export const promptsApi = {
  list: async (page = 1, pageSize = 20): Promise<PromptListResponse> => {
    const response = await api.get<PromptListResponse>('/prompts', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  get: async (promptId: number): Promise<Prompt> => {
    const response = await api.get<Prompt>(`/prompts/${promptId}`);
    return response.data;
  },

  create: async (name: string, content: string, isActive = false): Promise<Prompt> => {
    const response = await api.post<Prompt>('/prompts', {
      name,
      content,
      is_active: isActive,
    });
    return response.data;
  },

  update: async (
    promptId: number,
    data: { name?: string; content?: string; is_active?: boolean },
  ): Promise<Prompt> => {
    const response = await api.patch<Prompt>(`/prompts/${promptId}`, data);
    return response.data;
  },

  delete: async (promptId: number): Promise<void> => {
    await api.delete(`/prompts/${promptId}`);
  },

  activate: async (promptId: number): Promise<Prompt> => {
    const response = await api.put<Prompt>(`/prompts/${promptId}/activate`);
    return response.data;
  },
};

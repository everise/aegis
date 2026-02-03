/**
 * API client service for Aegis backend.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
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
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
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
    const response = await api.post('/sessions', { task_type: taskType });
    return response.data;
  },

  list: async (page = 1, pageSize = 20): Promise<SessionListResponse> => {
    const response = await api.get('/sessions', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  },

  get: async (sessionId: number): Promise<Session> => {
    const response = await api.get(`/sessions/${sessionId}`);
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
    const response = await api.get(`/${sessionId}/messages`, {
      params: { limit },
    });
    return response.data;
  },

  create: async (sessionId: number, content: string): Promise<Message> => {
    const response = await api.post(`/${sessionId}/messages`, { content });
    return response.data;
  },

  chat: async (sessionId: number, message: string): Promise<{ message_id: number; plan: ExecutionPlan; status: string }> => {
    const response = await api.post(`/${sessionId}/chat`, { content: message });
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
    const response = await api.get('/skills');
    return response.data;
  },

  get: async (skillName: string): Promise<SkillInfo> => {
    const response = await api.get(`/skills/${skillName}`);
    return response.data;
  },

  execute: async (
    skillName: string,
    params: Record<string, unknown>,
    messageId?: number
  ): Promise<SkillExecution> => {
    const response = await api.post('/skills/execute', {
      skill_name: skillName,
      params,
      message_id: messageId,
    });
    return response.data;
  },

  getExecution: async (executionId: number): Promise<SkillExecution> => {
    const response = await api.get(`/skills/executions/${executionId}`);
    return response.data;
  },
};

// Training API
export const trainingApi = {
  createJob: async (config?: Partial<TrainingConfig>): Promise<TrainingJob> => {
    const response = await api.post('/training/jobs', { config });
    return response.data;
  },

  listJobs: async (limit = 20): Promise<TrainingJob[]> => {
    const response = await api.get('/training/jobs', { params: { limit } });
    return response.data;
  },

  getJob: async (jobId: number): Promise<TrainingJob> => {
    const response = await api.get(`/training/jobs/${jobId}`);
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
    const response = await api.get('/training/status');
    return response.data;
  },

  getMetrics: async (lastN = 100): Promise<{ metrics: Record<string, unknown>[]; count: number }> => {
    const response = await api.get('/training/metrics', { params: { last_n: lastN } });
    return response.data;
  },

  getBufferStats: async (): Promise<Record<string, unknown>> => {
    const response = await api.get('/training/buffer/stats');
    return response.data;
  },
};

export default api;

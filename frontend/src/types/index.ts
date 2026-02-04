/**
 * Type definitions for Aegis frontend.
 */

// Session types
export interface Session {
  id: number;
  status: 'active' | 'completed' | 'failed';
  task_type: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionListResponse {
  sessions: Session[];
  total: number;
  page: number;
  page_size: number;
}

// Message types
export interface Message {
  id: number;
  session_id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  plan_json: ExecutionPlan | null;
  created_at: string;
}

export interface MessageListResponse {
  messages: Message[];
  total: number;
}

// Plan types
export interface PlanStep {
  step_number: number;
  thought: string;
  action: string;
  action_input: Record<string, unknown>;
  observation: Record<string, unknown> | null;
  status: string;
  error: string | null;
}

export interface ExecutionPlan {
  session_id: number | null;
  user_message: string;
  steps: PlanStep[];
  status: 'thinking' | 'executing' | 'observing' | 'completed' | 'failed';
  final_result: Record<string, unknown> | null;
}

// Skill types
export interface SkillInfo {
  name: string;
  description: string;
  submit_endpoint: string;
  parameters?: Record<string, unknown>;
}

export interface SkillExecution {
  id: number;
  message_id: number;
  skill_name: string;
  status: 'pending' | 'submitted' | 'polling' | 'completed' | 'failed' | 'timeout';
  request_params: Record<string, unknown> | null;
  remote_task_id: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  poll_count: number;
  submitted_at: string | null;
  completed_at: string | null;
  created_at: string;
}

// Training types
export interface TrainingConfig {
  total_epochs: number;
  steps_per_epoch: number;
  batch_size: number;
  learning_rate: number;
  discount_factor: number;
  buffer_size: number;
  use_cross_policy: boolean;
  use_task_normalization: boolean;
}

export interface TrainingJob {
  id: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  policy_version: string | null;
  config: TrainingConfig | null;
  total_epochs: number;
  current_epoch: number;
  total_steps: number;
  current_step: number;
  metrics: Record<string, unknown>;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface TrainingMetrics {
  epoch: number;
  step: number;
  policy_loss: number;
  value_loss: number;
  mean_return: number;
  mean_reward: number;
  buffer_size: number;
}

export interface TrainingStatus {
  status: string;
  current_epoch: number;
  current_step: number;
  policy_version: string;
  best_return: number;
  buffer_size: number;
  config: TrainingConfig;
}

// SSE Event types
export interface SSEEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

// API Response types
export interface ApiError {
  detail: string;
}

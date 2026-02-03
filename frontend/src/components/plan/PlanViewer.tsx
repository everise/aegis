/**
 * Plan viewer component showing ReAct execution steps.
 */

import { clsx } from 'clsx';
import type { ExecutionPlan, PlanStep } from '@/types';

interface PlanViewerProps {
  plan: ExecutionPlan;
}

function StepCard({ step, isLast }: { step: PlanStep; isLast: boolean }) {
  const statusColors = {
    thinking: 'border-yellow-300 bg-yellow-50',
    executing: 'border-blue-300 bg-blue-50',
    observing: 'border-purple-300 bg-purple-50',
    completed: 'border-green-300 bg-green-50',
    failed: 'border-red-300 bg-red-50',
  };

  const statusIcons = {
    thinking: '🤔',
    executing: '⚡',
    observing: '👁️',
    completed: '✅',
    failed: '❌',
  };

  return (
    <div className="relative">
      {/* Connector line */}
      {!isLast && (
        <div className="absolute left-4 top-12 bottom-0 w-0.5 bg-gray-200" />
      )}

      <div
        className={clsx(
          'border-2 rounded-lg p-3 mb-3',
          statusColors[step.status as keyof typeof statusColors] || 'border-gray-200 bg-white'
        )}
      >
        <div className="flex items-center gap-2 mb-2">
          <span className="text-lg">
            {statusIcons[step.status as keyof typeof statusIcons] || '⏳'}
          </span>
          <span className="font-semibold text-sm">Step {step.step_number}</span>
          <span className="text-xs text-gray-500 capitalize">{step.status}</span>
        </div>

        {/* Thought */}
        {step.thought && (
          <div className="mb-2">
            <p className="text-xs text-gray-500 mb-1">Thought:</p>
            <p className="text-sm text-gray-700">{step.thought}</p>
          </div>
        )}

        {/* Action */}
        {step.action && (
          <div className="mb-2">
            <p className="text-xs text-gray-500 mb-1">Action:</p>
            <div className="bg-gray-100 rounded px-2 py-1">
              <code className="text-xs text-gray-800">{step.action}</code>
            </div>
          </div>
        )}

        {/* Action Input */}
        {step.action_input && Object.keys(step.action_input).length > 0 && (
          <div className="mb-2">
            <p className="text-xs text-gray-500 mb-1">Input:</p>
            <pre className="bg-gray-100 rounded px-2 py-1 text-xs overflow-auto max-h-20">
              {JSON.stringify(step.action_input, null, 2)}
            </pre>
          </div>
        )}

        {/* Observation */}
        {step.observation && (
          <div>
            <p className="text-xs text-gray-500 mb-1">Observation:</p>
            <pre className="bg-gray-100 rounded px-2 py-1 text-xs overflow-auto max-h-32">
              {JSON.stringify(step.observation, null, 2)}
            </pre>
          </div>
        )}

        {/* Error */}
        {step.error && (
          <div className="mt-2 text-red-600 text-xs">
            Error: {step.error}
          </div>
        )}
      </div>
    </div>
  );
}

export default function PlanViewer({ plan }: PlanViewerProps) {
  const statusColors = {
    thinking: 'text-yellow-600 bg-yellow-100',
    executing: 'text-blue-600 bg-blue-100',
    observing: 'text-purple-600 bg-purple-100',
    completed: 'text-green-600 bg-green-100',
    failed: 'text-red-600 bg-red-100',
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-gray-900">Execution Plan</h3>
          <span
            className={clsx(
              'px-2 py-1 rounded text-xs font-medium capitalize',
              statusColors[plan.status as keyof typeof statusColors] || 'text-gray-600 bg-gray-100'
            )}
          >
            {plan.status}
          </span>
        </div>
        <p className="text-sm text-gray-500 truncate" title={plan.user_message}>
          "{plan.user_message}"
        </p>
      </div>

      {/* Steps */}
      <div className="flex-1 overflow-auto p-4">
        {plan.steps.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <div className="flex justify-center gap-1 mb-2">
              <span className="thinking-dot w-2 h-2 bg-gray-400 rounded-full" />
              <span className="thinking-dot w-2 h-2 bg-gray-400 rounded-full" />
              <span className="thinking-dot w-2 h-2 bg-gray-400 rounded-full" />
            </div>
            <p className="text-sm">Thinking...</p>
          </div>
        ) : (
          plan.steps.map((step, index) => (
            <StepCard
              key={step.step_number}
              step={step}
              isLast={index === plan.steps.length - 1}
            />
          ))
        )}
      </div>

      {/* Result */}
      {plan.final_result && (
        <div className="p-4 border-t border-gray-200 bg-green-50">
          <p className="text-xs text-gray-500 mb-1">Final Result:</p>
          <pre className="text-xs overflow-auto max-h-32">
            {JSON.stringify(plan.final_result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

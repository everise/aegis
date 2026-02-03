/**
 * Training page for RL training management.
 */

import { useEffect, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { trainingApi } from '@/services/api';
import type { TrainingJob, TrainingStatus } from '@/types';

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: 'bg-yellow-100 text-yellow-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
    cancelled: 'bg-gray-100 text-gray-800',
    idle: 'bg-gray-100 text-gray-800',
  };

  return (
    <span className={clsx('px-2 py-1 rounded text-xs font-medium capitalize', colors[status] || colors.idle)}>
      {status}
    </span>
  );
}

function JobCard({ job, onStart, onCancel }: { 
  job: TrainingJob; 
  onStart: (id: number) => void;
  onCancel: (id: number) => void;
}) {
  const progress = job.total_epochs > 0 ? (job.current_epoch / job.total_epochs) * 100 : 0;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">Job #{job.id}</h3>
        <StatusBadge status={job.status} />
      </div>

      <div className="space-y-2 text-sm text-gray-600">
        <div className="flex justify-between">
          <span>Policy Version:</span>
          <span className="font-mono">{job.policy_version || 'N/A'}</span>
        </div>
        <div className="flex justify-between">
          <span>Progress:</span>
          <span>{job.current_epoch} / {job.total_epochs} epochs</span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-3 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className="h-full bg-primary-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Actions */}
      <div className="mt-4 flex gap-2">
        {job.status === 'pending' && (
          <button
            onClick={() => onStart(job.id)}
            className="px-3 py-1 bg-primary-500 text-white rounded text-sm hover:bg-primary-600"
          >
            Start
          </button>
        )}
        {job.status === 'running' && (
          <button
            onClick={() => onCancel(job.id)}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
          >
            Cancel
          </button>
        )}
      </div>

      <div className="mt-3 text-xs text-gray-400">
        Created: {new Date(job.created_at).toLocaleString()}
      </div>
    </div>
  );
}

export default function TrainingPage() {
  const queryClient = useQueryClient();

  // Fetch training status
  const { data: status } = useQuery({
    queryKey: ['trainingStatus'],
    queryFn: trainingApi.getStatus,
    refetchInterval: 5000,
  });

  // Fetch jobs
  const { data: jobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['trainingJobs'],
    queryFn: () => trainingApi.listJobs(20),
  });

  // Fetch buffer stats
  const { data: bufferStats } = useQuery({
    queryKey: ['bufferStats'],
    queryFn: trainingApi.getBufferStats,
    refetchInterval: 10000,
  });

  // Mutations
  const createJobMutation = useMutation({
    mutationFn: () => trainingApi.createJob(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['trainingJobs'] }),
  });

  const startJobMutation = useMutation({
    mutationFn: trainingApi.startJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['trainingJobs'] }),
  });

  const cancelJobMutation = useMutation({
    mutationFn: trainingApi.cancelJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['trainingJobs'] }),
  });

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">RL Training</h1>

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Trainer Status</p>
            <div className="mt-1">
              <StatusBadge status={status?.status || 'idle'} />
            </div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Current Epoch</p>
            <p className="text-2xl font-bold text-gray-900">{status?.current_epoch || 0}</p>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Buffer Size</p>
            <p className="text-2xl font-bold text-gray-900">{status?.buffer_size || 0}</p>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-gray-500">Best Return</p>
            <p className="text-2xl font-bold text-gray-900">
              {status?.best_return?.toFixed(2) || 'N/A'}
            </p>
          </div>
        </div>

        {/* Create Job Button */}
        <div className="mb-6">
          <button
            onClick={() => createJobMutation.mutate()}
            disabled={createJobMutation.isPending}
            className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:bg-gray-300"
          >
            {createJobMutation.isPending ? 'Creating...' : '+ Create Training Job'}
          </button>
        </div>

        {/* Jobs List */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Training Jobs</h2>
          
          {jobsLoading ? (
            <p className="text-gray-500">Loading jobs...</p>
          ) : jobs && jobs.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {jobs.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStart={(id) => startJobMutation.mutate(id)}
                  onCancel={(id) => cancelJobMutation.mutate(id)}
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500">No training jobs yet</p>
          )}
        </div>

        {/* Buffer Stats */}
        {bufferStats && (
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Buffer Statistics</h2>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <pre className="text-sm text-gray-700 overflow-auto">
                {JSON.stringify(bufferStats, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

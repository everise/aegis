/**
 * Training page for RL training management.
 */

import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Play, Square, Plus, Activity } from "lucide-react";
import { trainingApi } from "@/services/api";
import type { TrainingJob } from "@/types";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function StatusBadge({ status }: { status: string }) {
  const variants: Record<string, string> = {
    pending: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
    running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
    completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
    failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
    cancelled: "bg-muted text-muted-foreground",
    idle: "bg-muted text-muted-foreground",
  };

  return (
    <span
      className={cn(
        "px-2 py-1 rounded-md text-xs font-medium capitalize",
        variants[status] || variants.idle
      )}
    >
      {status}
    </span>
  );
}

function JobCard({
  job,
  onStart,
  onCancel,
}: {
  job: TrainingJob;
  onStart: (id: number) => void;
  onCancel: (id: number) => void;
}) {
  const progress =
    job.total_epochs > 0 ? (job.current_epoch / job.total_epochs) * 100 : 0;

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">Job #{job.id}</h3>
        <StatusBadge status={job.status} />
      </div>

      <div className="space-y-2 text-sm text-muted-foreground">
        <div className="flex justify-between">
          <span>Policy Version:</span>
          <span className="font-mono text-foreground">
            {job.policy_version || "N/A"}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Progress:</span>
          <span className="text-foreground">
            {job.current_epoch} / {job.total_epochs} epochs
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Actions */}
      <div className="mt-4 flex gap-2">
        {job.status === "pending" && (
          <Button size="sm" onClick={() => onStart(job.id)}>
            <Play className="h-3 w-3 mr-1" />
            Start
          </Button>
        )}
        {job.status === "running" && (
          <Button
            size="sm"
            variant="destructive"
            onClick={() => onCancel(job.id)}
          >
            <Square className="h-3 w-3 mr-1" />
            Cancel
          </Button>
        )}
      </div>

      <div className="mt-3 text-xs text-muted-foreground">
        Created: {new Date(job.created_at).toLocaleString()}
      </div>
    </div>
  );
}

export default function TrainingPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Fetch training status
  const { data: status } = useQuery({
    queryKey: ["trainingStatus"],
    queryFn: trainingApi.getStatus,
    refetchInterval: 5000,
  });

  // Fetch jobs
  const { data: jobs, isLoading: jobsLoading } = useQuery({
    queryKey: ["trainingJobs"],
    queryFn: () => trainingApi.listJobs(20),
  });

  // Fetch buffer stats
  const { data: bufferStats } = useQuery({
    queryKey: ["bufferStats"],
    queryFn: trainingApi.getBufferStats,
    refetchInterval: 10000,
  });

  // Mutations
  const createJobMutation = useMutation({
    mutationFn: () => trainingApi.createJob(),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["trainingJobs"] }),
  });

  const startJobMutation = useMutation({
    mutationFn: trainingApi.startJob,
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["trainingJobs"] }),
  });

  const cancelJobMutation = useMutation({
    mutationFn: trainingApi.cancelJob,
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["trainingJobs"] }),
  });

  return (
    <div className="h-screen bg-background overflow-auto">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center gap-4 mb-6">
          <Button variant="ghost" size="icon" onClick={() => navigate("/chat")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold">RL Training</h1>
            <p className="text-sm text-muted-foreground">
              Manage reinforcement learning training jobs
            </p>
          </div>
        </div>

        {/* Status Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="rounded-lg border bg-card p-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <Activity className="h-4 w-4" />
              Trainer Status
            </div>
            <div className="mt-1">
              <StatusBadge status={status?.status || "idle"} />
            </div>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <p className="text-sm text-muted-foreground">Current Epoch</p>
            <p className="text-2xl font-bold">{status?.current_epoch || 0}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <p className="text-sm text-muted-foreground">Buffer Size</p>
            <p className="text-2xl font-bold">{status?.buffer_size || 0}</p>
          </div>
          <div className="rounded-lg border bg-card p-4">
            <p className="text-sm text-muted-foreground">Best Return</p>
            <p className="text-2xl font-bold">
              {status?.best_return?.toFixed(2) || "N/A"}
            </p>
          </div>
        </div>

        {/* Create Job Button */}
        <div className="mb-6">
          <Button
            onClick={() => createJobMutation.mutate()}
            disabled={createJobMutation.isPending}
          >
            <Plus className="h-4 w-4 mr-2" />
            {createJobMutation.isPending ? "Creating..." : "Create Training Job"}
          </Button>
        </div>

        {/* Jobs List */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-4">Training Jobs</h2>

          {jobsLoading ? (
            <p className="text-muted-foreground">Loading jobs...</p>
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
            <p className="text-muted-foreground">No training jobs yet</p>
          )}
        </div>

        {/* Buffer Stats */}
        {bufferStats && (
          <div>
            <h2 className="text-lg font-semibold mb-4">Buffer Statistics</h2>
            <div className="rounded-lg border bg-card p-4">
              <pre className="text-sm overflow-auto">
                {JSON.stringify(bufferStats, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

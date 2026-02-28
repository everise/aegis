/**
 * Context Counter component.
 *
 * Displays a context count icon in the composer toolbar.
 * On hover/click, shows a tooltip with detailed context statistics
 * from the ChromaDB vector database and working memory compression info.
 */

import { type FC, useState, useEffect, useCallback, useRef, useSyncExternalStore } from "react";
import { DatabaseIcon, Loader2Icon, ArchiveIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  contextApi,
  type ContextStatsResponse,
  type MemoryStatsResponse,
} from "@/services/api";
import {
  subscribeToCompressionState,
  getCompressionStateVersion,
  getCompressionEvent,
  getMemoryStatsEvent,
  type CompressionEvent,
  type MemoryStatsEvent,
} from "@/hooks/useAegisRuntime";

interface ContextCounterProps {
  sessionId: number | null;
  className?: string;
}

const ContextCounter: FC<ContextCounterProps> = ({
  sessionId,
  className,
}) => {
  // Subscribe to global compression state from SSE
  const compressionVersion = useSyncExternalStore(
    subscribeToCompressionState,
    getCompressionStateVersion
  );
  const compressionEvent: CompressionEvent | null = sessionId
    ? getCompressionEvent(sessionId)
    : null;
  const memoryStatsFromSSE: MemoryStatsEvent | null = sessionId
    ? getMemoryStatsEvent(sessionId)
    : null;

  const [stats, setStats] = useState<ContextStatsResponse | null>(null);
  const [memStats, setMemStats] = useState<MemoryStatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [visible, setVisible] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Flash animation when compression occurs
  const [compressFlash, setCompressFlash] = useState(false);
  const lastCompressionRef = useRef<CompressionEvent | null>(null);

  const fetchStats = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const [ctxData, memData] = await Promise.all([
        contextApi.getStats(sessionId),
        contextApi.getMemoryStats(sessionId).catch(() => null),
      ]);
      setStats(ctxData);
      if (memData) setMemStats(memData);
    } catch (e) {
      setError("无法获取上下文信息");
      console.error("Failed to fetch context stats:", e);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Refresh stats when sessionId changes
  useEffect(() => {
    if (sessionId) {
      fetchStats();
    } else {
      setStats(null);
      setMemStats(null);
    }
  }, [sessionId, fetchStats]);

  // Poll interval
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(fetchStats, 10000);
    return () => clearInterval(interval);
  }, [sessionId, fetchStats]);

  // React to SSE compression event: flash indicator + update local stats
  useEffect(() => {
    if (compressionEvent && compressionEvent !== lastCompressionRef.current) {
      lastCompressionRef.current = compressionEvent;
      setCompressFlash(true);
      const timer = setTimeout(() => setCompressFlash(false), 2000);
      // Refresh stats after compression
      fetchStats();
      return () => clearTimeout(timer);
    }
  }, [compressionEvent, compressionVersion, fetchStats]);

  // React to SSE memory_stats event
  useEffect(() => {
    if (memoryStatsFromSSE) {
      setMemStats((prev) => ({
        session_id: prev?.session_id ?? sessionId ?? 0,
        message_count: memoryStatsFromSSE.message_count,
        compressed_count: prev?.compressed_count ?? 0,
        total_tokens: memoryStatsFromSSE.total_tokens,
        max_tokens: memoryStatsFromSSE.max_tokens,
        usage_ratio: memoryStatsFromSSE.usage_ratio,
        image_url_count: memoryStatsFromSSE.image_url_count,
        compression_count: memoryStatsFromSSE.compression_count,
      }));
    }
  }, [memoryStatsFromSSE, compressionVersion, sessionId]);

  const showTooltip = useCallback(() => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }
    setVisible(true);
    fetchStats();
  }, [fetchStats]);

  const hideTooltip = useCallback(() => {
    hideTimeoutRef.current = setTimeout(() => {
      setVisible(false);
    }, 200);
  }, []);

  const handleClick = useCallback(() => {
    setVisible((prev) => !prev);
    fetchStats();
  }, [fetchStats]);

  const usagePercent = stats
    ? Math.round(stats.context_window_usage * 100)
    : 0;
  const totalVectors = stats?.total_vectors ?? 0;
  const memUsagePercent = memStats
    ? Math.round(memStats.usage_ratio * 100)
    : 0;
  const compressionCount = memStats?.compression_count ?? 0;

  const getUsageColor = (percent: number) => {
    if (percent >= 80) return "text-red-500";
    if (percent >= 50) return "text-yellow-500";
    return "text-muted-foreground";
  };

  const getUsageBarColor = (percent: number) => {
    if (percent >= 80) return "bg-red-500";
    if (percent >= 50) return "bg-yellow-500";
    return "bg-primary";
  };

  const formatTokens = (tokens: number): string => {
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
    return tokens.toString();
  };

  return (
    <div className={cn("relative inline-flex", className)}>
      <button
        ref={triggerRef}
        type="button"
        onClick={handleClick}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        className={cn(
          "inline-flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs transition-colors",
          "hover:bg-muted/80",
          compressFlash ? "text-violet-500" : getUsageColor(usagePercent),
          compressFlash && "animate-pulse"
        )}
        title="上下文信息"
      >
        {loading ? (
          <Loader2Icon className="size-3.5 animate-spin" />
        ) : compressFlash ? (
          <ArchiveIcon className="size-3.5" />
        ) : (
          <DatabaseIcon className="size-3.5" />
        )}
        <span className="font-medium tabular-nums">
          {totalVectors}
          {compressionCount > 0 && (
            <span className="text-violet-500 ml-0.5">·{compressionCount}</span>
          )}
        </span>
      </button>

      {/* Custom Tooltip */}
      {visible && (
        <div
          ref={tooltipRef}
          onMouseEnter={showTooltip}
          onMouseLeave={hideTooltip}
          className={cn(
            "absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50",
            "w-80 rounded-xl border bg-popover p-4 text-popover-foreground shadow-lg",
            "animate-in fade-in-0 zoom-in-95 slide-in-from-bottom-2 duration-200"
          )}
        >
          {/* Arrow */}
          <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 size-3 rotate-45 border-b border-r bg-popover" />

          {error ? (
            <div className="text-sm text-destructive">{error}</div>
          ) : !stats ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2Icon className="size-4 animate-spin" />
              加载中...
            </div>
          ) : (
            <div className="space-y-3">
              {/* Header */}
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold">上下文信息</span>
                <div className="flex gap-1.5">
                  <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                    Vector DB
                  </span>
                  {compressionCount > 0 && (
                    <span className="rounded-full bg-violet-500/10 px-2 py-0.5 text-xs font-medium text-violet-500">
                      已压缩 ×{compressionCount}
                    </span>
                  )}
                </div>
              </div>

              {/* ── Working Memory Section ─────────────── */}
              {memStats && (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground flex items-center gap-1">
                      <ArchiveIcon className="size-3" />
                      工作记忆
                    </span>
                    <span
                      className={cn(
                        "font-medium",
                        getUsageColor(memUsagePercent)
                      )}
                    >
                      {memUsagePercent}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-muted">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-500",
                        compressionCount > 0 ? "bg-violet-500" : getUsageBarColor(memUsagePercent)
                      )}
                      style={{
                        width: `${Math.min(memUsagePercent, 100)}%`,
                      }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                      {formatTokens(memStats.total_tokens)} tokens
                    </span>
                    <span>/ {formatTokens(memStats.max_tokens)}</span>
                  </div>

                  {/* Compression details */}
                  {compressionCount > 0 && (
                    <div className="flex items-center gap-3 text-xs">
                      <div className="flex items-center gap-1 text-violet-500">
                        <ArchiveIcon className="size-3" />
                        <span>
                          压缩 {compressionCount} 次
                        </span>
                      </div>
                      {compressionEvent && (
                        <span className="text-muted-foreground">
                          比率: {Math.round((1 - compressionEvent.ratio) * 100)}% ↓
                        </span>
                      )}
                      {memStats.image_url_count > 0 && (
                        <span className="text-muted-foreground">
                          🖼 {memStats.image_url_count}
                        </span>
                      )}
                    </div>
                  )}

                  {/* Last compression event detail */}
                  {compressionEvent && (
                    <div className="rounded-lg bg-violet-500/5 border border-violet-500/10 p-2 text-xs text-violet-600 dark:text-violet-400">
                      <div className="font-medium mb-1">最近一次压缩</div>
                      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                        <span>压缩前</span>
                        <span className="text-right font-mono">
                          {formatTokens(compressionEvent.tokens_before)} tokens
                        </span>
                        <span>压缩后</span>
                        <span className="text-right font-mono">
                          {formatTokens(compressionEvent.tokens_after)} tokens
                        </span>
                        <span>消息数</span>
                        <span className="text-right font-mono">
                          {compressionEvent.original_count} → {compressionEvent.compressed_count}
                        </span>
                        <span>策略</span>
                        <span className="text-right font-mono">
                          {compressionEvent.strategy}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ── Vector DB Context Section ──────────── */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <DatabaseIcon className="size-3" />
                    向量上下文窗口
                  </span>
                  <span
                    className={cn(
                      "font-medium",
                      getUsageColor(usagePercent)
                    )}
                  >
                    {usagePercent}%
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-muted">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      getUsageBarColor(usagePercent)
                    )}
                    style={{
                      width: `${Math.min(usagePercent, 100)}%`,
                    }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {formatTokens(stats.total_tokens_estimate)} tokens
                  </span>
                  <span>/ {formatTokens(stats.max_context_tokens)}</span>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-3 gap-2">
                <div className="rounded-lg bg-muted/50 p-2">
                  <div className="text-lg font-bold tabular-nums">
                    {stats.total_vectors}
                  </div>
                  <div className="text-xs text-muted-foreground">向量数</div>
                </div>
                <div className="rounded-lg bg-muted/50 p-2">
                  <div className="text-lg font-bold tabular-nums">
                    {stats.user_message_count + stats.assistant_message_count}
                  </div>
                  <div className="text-xs text-muted-foreground">消息数</div>
                </div>
                <div className="rounded-lg bg-muted/50 p-2">
                  <div className="text-lg font-bold tabular-nums">
                    {memStats?.message_count ?? "–"}
                  </div>
                  <div className="text-xs text-muted-foreground">工作记忆</div>
                </div>
              </div>

              {/* Role Breakdown */}
              <div className="space-y-1">
                <div className="text-xs font-medium text-muted-foreground">
                  消息分布
                </div>
                <div className="flex gap-3 text-xs">
                  <div className="flex items-center gap-1">
                    <span className="size-2 rounded-full bg-blue-500" />
                    <span>用户: {stats.user_message_count}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="size-2 rounded-full bg-green-500" />
                    <span>助手: {stats.assistant_message_count}</span>
                  </div>
                  {stats.system_message_count > 0 && (
                    <div className="flex items-center gap-1">
                      <span className="size-2 rounded-full bg-orange-500" />
                      <span>系统: {stats.system_message_count}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Time Range */}
              {stats.oldest_message_time && (
                <div className="border-t pt-2 text-xs text-muted-foreground">
                  <div className="flex justify-between">
                    <span>最早消息</span>
                    <span>
                      {new Date(stats.oldest_message_time).toLocaleString(
                        "zh-CN"
                      )}
                    </span>
                  </div>
                  {stats.newest_message_time &&
                    stats.newest_message_time !==
                      stats.oldest_message_time && (
                      <div className="flex justify-between mt-0.5">
                        <span>最新消息</span>
                        <span>
                          {new Date(
                            stats.newest_message_time
                          ).toLocaleString("zh-CN")}
                        </span>
                      </div>
                    )}
                </div>
              )}

              {/* Collection Name */}
              <div className="border-t pt-2 text-xs text-muted-foreground">
                <div className="flex justify-between">
                  <span>ChromaDB 集合</span>
                  <span className="font-mono">{stats.collection_name}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ContextCounter;

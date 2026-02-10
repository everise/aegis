/**
 * Settings page with Prompt Management and Chat History tabs.
 */

import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Plus,
  Trash2,
  Save,
  Check,
  Edit2,
  X,
  MessageSquare,
  FileText,
  ChevronLeft,
  ChevronRight,
  Star,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { promptsApi, sessionsApi } from "@/services/api";
import type { Prompt, Session } from "@/types";

// ---------------------------------------------------------------------------
// Tab definitions
// ---------------------------------------------------------------------------

type Tab = "prompts" | "history";

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>("prompts");

  return (
    <div className="flex h-screen w-full flex-col bg-background">
      {/* Header */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b px-4">
        <Button
          variant="ghost"
          size="icon"
          className="size-9"
          onClick={() => navigate("/chat")}
        >
          <ArrowLeft className="size-4" />
          <span className="sr-only">返回</span>
        </Button>
        <h1 className="text-lg font-semibold">设置</h1>
      </header>

      {/* Tabs */}
      <div className="flex shrink-0 border-b px-4">
        <button
          type="button"
          onClick={() => setActiveTab("prompts")}
          className={cn(
            "flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors",
            activeTab === "prompts"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground",
          )}
        >
          <FileText className="size-4" />
          Prompt 管理
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("history")}
          className={cn(
            "flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors",
            activeTab === "history"
              ? "border-primary text-primary"
              : "border-transparent text-muted-foreground hover:text-foreground",
          )}
        >
          <MessageSquare className="size-4" />
          对话记录
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === "prompts" ? <PromptManagement /> : <ChatHistory />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prompt Management
// ---------------------------------------------------------------------------

function PromptManagement() {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const pageSize = 10;

  // Editing state
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [editContent, setEditContent] = useState("");

  // New prompt state
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newContent, setNewContent] = useState("");

  const fetchPrompts = useCallback(async () => {
    setLoading(true);
    try {
      const data = await promptsApi.list(page, pageSize);
      setPrompts(data.prompts);
      setTotal(data.total);
    } catch (err) {
      console.error("Failed to load prompts:", err);
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    fetchPrompts();
  }, [fetchPrompts]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const handleCreate = async () => {
    if (!newName.trim() || !newContent.trim()) return;
    try {
      await promptsApi.create(newName.trim(), newContent.trim());
      setCreating(false);
      setNewName("");
      setNewContent("");
      fetchPrompts();
    } catch (err) {
      console.error("Failed to create prompt:", err);
    }
  };

  const handleUpdate = async (id: number) => {
    try {
      await promptsApi.update(id, { name: editName, content: editContent });
      setEditingId(null);
      fetchPrompts();
    } catch (err) {
      console.error("Failed to update prompt:", err);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      await promptsApi.delete(id);
      fetchPrompts();
    } catch (err) {
      console.error("Failed to delete prompt:", err);
    }
  };

  const handleActivate = async (id: number) => {
    try {
      await promptsApi.activate(id);
      fetchPrompts();
    } catch (err) {
      console.error("Failed to activate prompt:", err);
    }
  };

  const startEdit = (prompt: Prompt) => {
    setEditingId(prompt.id);
    setEditName(prompt.name);
    setEditContent(prompt.content);
  };

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-base font-semibold">Prompt 模板</h2>
          <p className="text-sm text-muted-foreground">
            管理系统 prompt 模板，激活后的 prompt 将作为规划模型的系统指令
          </p>
        </div>
        <Button
          size="sm"
          onClick={() => {
            setCreating(true);
            setNewName("");
            setNewContent("");
          }}
          disabled={creating}
        >
          <Plus className="mr-1 size-4" />
          新建
        </Button>
      </div>

      {/* Create form */}
      {creating && (
        <div className="space-y-3 rounded-lg border p-4">
          <input
            className="w-full rounded-md border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/50"
            placeholder="Prompt 名称"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <textarea
            className="min-h-[120px] w-full rounded-md border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/50 resize-y"
            placeholder="Prompt 内容..."
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
          />
          <div className="flex gap-2 justify-end">
            <Button size="sm" variant="ghost" onClick={() => setCreating(false)}>
              <X className="mr-1 size-3" />
              取消
            </Button>
            <Button size="sm" onClick={handleCreate} disabled={!newName.trim() || !newContent.trim()}>
              <Save className="mr-1 size-3" />
              保存
            </Button>
          </div>
        </div>
      )}

      {/* Prompts list */}
      {loading ? (
        <div className="py-12 text-center text-sm text-muted-foreground animate-pulse">
          加载中…
        </div>
      ) : prompts.length === 0 && !creating ? (
        <div className="py-12 text-center text-sm text-muted-foreground">
          暂无 Prompt 模板，点击「新建」创建第一个
        </div>
      ) : (
        <div className="space-y-3">
          {prompts.map((prompt) => (
            <div
              key={prompt.id}
              className={cn(
                "rounded-lg border p-4 transition-colors",
                prompt.is_active && "border-primary/50 bg-primary/5",
              )}
            >
              {editingId === prompt.id ? (
                /* Edit mode */
                <div className="space-y-3">
                  <input
                    className="w-full rounded-md border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/50"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                  />
                  <textarea
                    className="min-h-[120px] w-full rounded-md border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/50 resize-y"
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                  />
                  <div className="flex gap-2 justify-end">
                    <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}>
                      <X className="mr-1 size-3" />
                      取消
                    </Button>
                    <Button size="sm" onClick={() => handleUpdate(prompt.id)}>
                      <Save className="mr-1 size-3" />
                      保存
                    </Button>
                  </div>
                </div>
              ) : (
                /* Display mode */
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{prompt.name}</span>
                      {prompt.is_active && (
                        <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                          <Check className="size-3" />
                          已激活
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1">
                      {!prompt.is_active && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 px-2 text-xs"
                          onClick={() => handleActivate(prompt.id)}
                          title="激活此 Prompt"
                        >
                          <Star className="mr-1 size-3" />
                          激活
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2"
                        onClick={() => startEdit(prompt)}
                      >
                        <Edit2 className="size-3" />
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-destructive hover:text-destructive"
                        onClick={() => handleDelete(prompt.id)}
                      >
                        <Trash2 className="size-3" />
                      </Button>
                    </div>
                  </div>
                  <p className="whitespace-pre-wrap text-sm text-muted-foreground line-clamp-4">
                    {prompt.content}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    更新于 {new Date(prompt.updated_at).toLocaleString("zh-CN")}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button
            size="sm"
            variant="ghost"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="size-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            {page} / {totalPages}
          </span>
          <Button
            size="sm"
            variant="ghost"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat History
// ---------------------------------------------------------------------------

function ChatHistory() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const pageSize = 15;

  const fetchSessions = useCallback(async () => {
    setLoading(true);
    try {
      const data = await sessionsApi.list(page, pageSize);
      setSessions(data.sessions);
      setTotal(data.total);
    } catch (err) {
      console.error("Failed to load sessions:", err);
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const handleDelete = async (id: number) => {
    try {
      await sessionsApi.delete(id);
      fetchSessions();
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  const statusBadge = (status: string) => {
    const map: Record<string, { label: string; cls: string }> = {
      active: {
        label: "进行中",
        cls: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
      },
      completed: {
        label: "已完成",
        cls: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
      },
      failed: {
        label: "失败",
        cls: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
      },
    };
    const info = map[status] ?? { label: status, cls: "bg-muted text-muted-foreground" };
    return (
      <span className={cn("rounded-full px-2 py-0.5 text-xs font-medium", info.cls)}>
        {info.label}
      </span>
    );
  };

  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6">
      <div>
        <h2 className="text-base font-semibold">对话记录</h2>
        <p className="text-sm text-muted-foreground">
          浏览和管理历史对话会话
        </p>
      </div>

      {loading ? (
        <div className="py-12 text-center text-sm text-muted-foreground animate-pulse">
          加载中…
        </div>
      ) : sessions.length === 0 ? (
        <div className="py-12 text-center text-sm text-muted-foreground">
          暂无对话记录
        </div>
      ) : (
        <div className="space-y-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              className="group flex items-center gap-3 rounded-lg border p-3 transition-colors hover:bg-muted/50 cursor-pointer"
              onClick={() => navigate(`/chat/${session.id}`)}
            >
              <MessageSquare className="size-5 shrink-0 text-muted-foreground" />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">
                    会话 #{session.id}
                  </span>
                  {statusBadge(session.status)}
                  {session.task_type && (
                    <span className="text-xs text-muted-foreground">
                      {session.task_type}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
                  <span>{session.message_count} 条消息</span>
                  <span>
                    {new Date(session.created_at).toLocaleString("zh-CN")}
                  </span>
                </div>
              </div>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 px-2 opacity-0 group-hover:opacity-100 text-destructive hover:text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(session.id);
                }}
              >
                <Trash2 className="size-3" />
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button
            size="sm"
            variant="ghost"
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="size-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            {page} / {totalPages}
          </span>
          <Button
            size="sm"
            variant="ghost"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight className="size-4" />
          </Button>
        </div>
      )}
    </div>
  );
}

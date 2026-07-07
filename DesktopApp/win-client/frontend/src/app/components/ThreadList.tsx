"use client";

import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import {
  BrainCircuit,
  Clock,
  Download,
  Loader2,
  MessageSquare,
  Pencil,
  Pin,
  PinOff,
  Puzzle,
  Search,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";
import { useQueryState } from "nuqs";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectLabel,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { formatTime, formatFullTime } from "@/lib/time";
import type { ThreadItem } from "@/app/hooks/useThreads";
import {
  useThreads,
  deleteThread,
  renameThread,
  pinThread,
  exportThread,
} from "@/app/hooks/useThreads";
import { useMemoryActivity } from "@/app/hooks/useMemoryActivity";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { useI18n } from "@/providers/I18nProvider";
import type { MessageKey } from "@/i18n/messages";

type StatusFilter = "all" | "idle" | "busy" | "interrupted" | "error";

const GROUP_LABELS = {
  interrupted: "thread.group.interrupted",
  today: "thread.group.today",
  yesterday: "thread.group.yesterday",
  week: "thread.group.week",
  older: "thread.group.older",
} as const;

const STATUS_COLORS: Record<ThreadItem["status"], string> = {
  idle: "bg-green-500",
  busy: "bg-blue-500",
  interrupted: "bg-orange-500",
  error: "bg-red-600",
};

const STATUS_LABELS = {
  idle: "thread.status.idle",
  busy: "thread.status.busy",
  interrupted: "thread.status.interrupted",
  error: "thread.status.error",
} as const satisfies Record<ThreadItem["status"], MessageKey>;

function getThreadColor(status: ThreadItem["status"]): string {
  return STATUS_COLORS[status] ?? "bg-gray-400";
}

function StatusFilterItem({
  status,
  label,
  badge,
}: {
  status: ThreadItem["status"];
  label: string;
  badge?: number;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <span
        className={cn(
          "inline-block size-2 rounded-full",
          getThreadColor(status)
        )}
      />
      {label}
      {badge !== undefined && badge > 0 && (
        <span className="ml-1 inline-flex items-center justify-center rounded-full bg-red-600 px-1.5 py-0.5 text-xs font-bold leading-none text-white">
          {badge}
        </span>
      )}
    </span>
  );
}

function ErrorState({ message, title }: { message: string; title: string }) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center">
      <p className="text-sm text-red-600">{title}</p>
      <p className="mt-1 text-xs text-muted-foreground">{message}</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="space-y-2 p-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton
          key={i}
          className="h-16 w-full"
        />
      ))}
    </div>
  );
}

function EmptyState({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center">
      <MessageSquare className="mb-2 h-12 w-12 text-gray-300" />
      <p className="text-sm text-muted-foreground">{label}</p>
    </div>
  );
}

interface ThreadListProps {
  onThreadSelect: (id: string) => void;
  onClose?: () => void;
  onNewChat?: () => void;
  onMutateReady?: (mutate: () => void) => void;
  onInterruptCountChange?: (count: number) => void;
}

export function ThreadList({
  onThreadSelect,
  onClose,
  onNewChat,
  onMutateReady,
  onInterruptCountChange,
}: ThreadListProps) {
  const { t } = useI18n();
  const [currentThreadId, setThreadId] = useQueryState("threadId");
  const [view, setView] = useQueryState("view");
  // Badge the Memory nav when long-term memory changed since last viewed.
  const { unseenCount: memoryUnseen, markSeen: markMemorySeen } =
    useMemoryActivity();
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [search, setSearch] = useState("");
  const [renameTarget, setRenameTarget] = useState<ThreadItem | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ThreadItem | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [now, setNow] = useState(() => new Date());
  const [pinBusyIds, setPinBusyIds] = useState<Set<string>>(() => new Set());
  const [exportBusyIds, setExportBusyIds] = useState<Set<string>>(
    () => new Set()
  );

  const threads = useThreads({
    status: statusFilter === "all" ? undefined : statusFilter,
    limit: 20,
  });

  // Dedupe by id, keeping the first occurrence — page 0 wins, so the freshest
  // `updated_at` survives. Without this, a thread whose `updated_at` advances
  // mid-run (every agent step persists) appears in page 0 with the new value
  // AND in a stale page 1+ with the old value, getting bucketed into both
  // "Today" and "This Week" simultaneously. `revalidateFirstPage: true` in
  // useThreads.ts keeps page 0 fresh but leaves later pages cached.
  const flattened = useMemo(() => {
    const seen = new Set<string>();
    const out: ThreadItem[] = [];
    for (const t of threads.data?.flat() ?? []) {
      if (seen.has(t.id)) continue;
      seen.add(t.id);
      out.push(t);
    }
    return out;
  }, [threads.data]);

  // Client-side filter of the loaded threads by title.
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return flattened;
    return flattened.filter((t) => t.title.toLowerCase().includes(q));
  }, [flattened, search]);

  useEffect(() => {
    const tick = () => setNow(new Date());
    const interval = window.setInterval(tick, 60_000);
    window.addEventListener("focus", tick);
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", tick);
      document.removeEventListener("visibilitychange", tick);
    };
  }, []);

  const isLoadingMore =
    threads.size > 0 && threads.data?.[threads.size - 1] == null;
  const isEmpty = threads.data?.at(0)?.length === 0;
  const isReachingEnd = isEmpty || (threads.data?.at(-1)?.length ?? 0) < 20;

  // Pinned threads float to a dedicated "Research" section at the top, sorted
  // newest-first (same order as the time groups below). `filtered` already
  // arrives sorted by updated_at desc from the backend.
  const pinned = useMemo(() => filtered.filter((t) => t.pinned), [filtered]);

  // Group threads by time and status
  const grouped = useMemo(() => {
    const groups: Record<keyof typeof GROUP_LABELS, ThreadItem[]> = {
      interrupted: [],
      today: [],
      yesterday: [],
      week: [],
      older: [],
    };

    filtered.forEach((thread) => {
      // Pinned threads live in the "Research" section only, not the time groups.
      if (thread.pinned) return;

      if (thread.status === "interrupted") {
        groups.interrupted.push(thread);
        return;
      }

      const diff = now.getTime() - thread.updatedAt.getTime();
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));

      if (days === 0) {
        groups.today.push(thread);
      } else if (days === 1) {
        groups.yesterday.push(thread);
      } else if (days < 7) {
        groups.week.push(thread);
      } else {
        groups.older.push(thread);
      }
    });

    return groups;
  }, [filtered, now]);

  const interruptedCount = useMemo(() => {
    return flattened.filter((t) => t.status === "interrupted").length;
  }, [flattened]);

  // Expose thread list revalidation to parent component
  // Use refs to create a stable callback that always calls the latest mutate function
  const onMutateReadyRef = useRef(onMutateReady);
  const mutateRef = useRef(threads.mutate);

  useEffect(() => {
    onMutateReadyRef.current = onMutateReady;
  }, [onMutateReady]);

  useEffect(() => {
    mutateRef.current = threads.mutate;
  }, [threads.mutate]);

  const mutateFn = useCallback(() => {
    mutateRef.current();
  }, []);

  useEffect(() => {
    onMutateReadyRef.current?.(mutateFn);
    // Only run once on mount to avoid infinite loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Notify parent of interrupt count changes
  useEffect(() => {
    onInterruptCountChange?.(interruptedCount);
  }, [interruptedCount, onInterruptCountChange]);

  // Synchronous re-entry lock: `actionBusy` state only blocks after a re-render,
  // so a fast second Enter could fire a mutation twice. The ref guards instantly.
  const actionBusyRef = useRef(false);
  const pinBusyIdsRef = useRef<Set<string>>(new Set());
  const exportBusyIdsRef = useRef<Set<string>>(new Set());

  // After deleting a thread its row (and the trigger button) is gone, so move
  // keyboard focus to a stable target (New Chat) instead of dropping to <body>.
  const newChatRef = useRef<HTMLButtonElement>(null);
  const pendingDeleteFocusRef = useRef(false);

  const submitRename = async () => {
    if (!renameTarget || actionBusyRef.current) return;
    const title = renameValue.trim();
    if (!title || title === renameTarget.title) {
      setRenameTarget(null);
      return;
    }
    actionBusyRef.current = true;
    setActionBusy(true);
    try {
      await renameThread(renameTarget.id, title);
      setRenameTarget(null);
      mutateFn();
    } catch {
      toast.error(t("thread.toast.renameFailed"));
    } finally {
      actionBusyRef.current = false;
      setActionBusy(false);
    }
  };

  const confirmDelete = async () => {
    if (!deleteTarget || actionBusyRef.current) return;
    actionBusyRef.current = true;
    setActionBusy(true);
    try {
      await deleteThread(deleteTarget.id);
      // If the open thread was deleted, take the SAME reset path as New Chat
      // (which also remounts the chat session) instead of only clearing URL state.
      if (currentThreadId === deleteTarget.id) {
        if (onNewChat) {
          onNewChat();
        } else {
          setThreadId(null);
          setView(null);
        }
      }
      // Hand focus to New Chat once the dialog closes (the trigger is gone).
      pendingDeleteFocusRef.current = true;
      setDeleteTarget(null);
      mutateFn();
    } catch {
      toast.error(t("thread.toast.deleteFailed"));
    } finally {
      actionBusyRef.current = false;
      setActionBusy(false);
    }
  };

  const togglePin = async (thread: ThreadItem) => {
    if (pinBusyIdsRef.current.has(thread.id)) return;
    pinBusyIdsRef.current.add(thread.id);
    setPinBusyIds((current) => {
      const next = new Set(current);
      next.add(thread.id);
      return next;
    });
    try {
      await pinThread(thread.id, !thread.pinned);
      mutateFn();
    } catch {
      toast.error(
        thread.pinned
          ? t("thread.toast.unpinFailed")
          : t("thread.toast.pinFailed")
      );
    } finally {
      pinBusyIdsRef.current.delete(thread.id);
      setPinBusyIds((current) => {
        const next = new Set(current);
        next.delete(thread.id);
        return next;
      });
    }
  };

  const runExport = async (thread: ThreadItem) => {
    if (exportBusyIdsRef.current.has(thread.id)) return;
    exportBusyIdsRef.current.add(thread.id);
    setExportBusyIds((current) => {
      const next = new Set(current);
      next.add(thread.id);
      return next;
    });
    try {
      await exportThread(thread.id, thread.title);
    } catch {
      toast.error(t("thread.toast.exportFailed"));
    } finally {
      exportBusyIdsRef.current.delete(thread.id);
      setExportBusyIds((current) => {
        const next = new Set(current);
        next.delete(thread.id);
        return next;
      });
    }
  };

  // A single thread row (select button + per-thread actions). Used by both the
  // pinned "Research" section and the time-grouped "Recents" sections; the only
  // difference is the Pin ↔ Unpin action, driven by `thread.pinned`.
  const renderThreadCard = (thread: ThreadItem) => {
    const pinBusy = pinBusyIds.has(thread.id);
    const exportBusy = exportBusyIds.has(thread.id);
    const statusLabel = t(STATUS_LABELS[thread.status]);

    return (
      <div
        key={thread.id}
        className="group relative"
      >
        {/* Selectable row — a native button so Enter/Space and role come for
          free. Action buttons are SIBLINGS (below), never nested inside. */}
        <button
          type="button"
          onClick={() => onThreadSelect(thread.id)}
          className={cn(
            "grid w-full cursor-pointer items-center gap-2 rounded-md py-2 pl-2.5 pr-28 text-left transition-colors duration-200 md:pr-2.5 md:group-focus-within:pr-28 md:group-hover:pr-28",
            "hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            currentThreadId === thread.id
              ? "border border-primary bg-accent hover:bg-accent"
              : "border border-transparent bg-transparent"
          )}
          aria-current={currentThreadId === thread.id}
        >
          <div className="min-w-0 flex-1">
            {/* Title + Timestamp Row */}
            <div className="mb-0.5 flex items-center justify-between gap-2">
              <h3 className="flex min-w-0 items-center gap-1.5 text-sm font-semibold">
                {thread.pinned && (
                  <Pin
                    className="size-3 flex-shrink-0 text-[var(--brand)]"
                    aria-hidden="true"
                  />
                )}
                <span className="truncate">{thread.title}</span>
              </h3>
              <span className="ml-2 flex-shrink-0 text-xs tabular-nums text-muted-foreground">
                <time
                  dateTime={thread.updatedAt.toISOString()}
                  title={formatFullTime(thread.updatedAt)}
                >
                  {formatTime(thread.updatedAt, now)}
                </time>
              </span>
            </div>
            {/* Description + Status Row */}
            <div className="flex items-center justify-between">
              <p className="flex-1 truncate text-[13px] text-muted-foreground">
                {thread.description}
              </p>
              <div className="ml-2 flex-shrink-0">
                <span
                  role="img"
                  aria-label={t("thread.statusAria", { status: statusLabel })}
                  title={t("thread.statusAria", { status: statusLabel })}
                  className={cn(
                    "h-2 w-2 rounded-full",
                    getThreadColor(thread.status)
                  )}
                />
              </div>
            </div>
          </div>
        </button>
        {/* Per-thread actions — siblings of the select button (not nested);
          shown on touch, reveal on hover/focus on desktop. */}
        <div className="absolute right-1.5 top-1.5 flex items-center gap-0.5 rounded-md bg-accent/95 p-0.5 opacity-100 shadow-sm backdrop-blur-sm transition-opacity md:opacity-0 md:group-focus-within:opacity-100 md:group-hover:opacity-100">
          <button
            type="button"
            aria-label={
              thread.pinned
                ? t("thread.unpinAria", { title: thread.title })
                : t("thread.pinAria", { title: thread.title })
            }
            title={thread.pinned ? t("thread.unpin") : t("thread.pin")}
            onClick={() => togglePin(thread)}
            disabled={pinBusy}
            className={cn(
              "rounded p-1 transition-colors hover:bg-background focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
              thread.pinned
                ? "text-[var(--brand)] hover:text-[var(--brand)]"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {thread.pinned ? (
              <PinOff
                className="size-3.5"
                aria-hidden="true"
              />
            ) : (
              <Pin
                className="size-3.5"
                aria-hidden="true"
              />
            )}
          </button>
          <button
            type="button"
            aria-label={t("thread.renameAria", { title: thread.title })}
            title={t("thread.rename")}
            onClick={() => {
              setRenameTarget(thread);
              setRenameValue(thread.title);
            }}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Pencil
              className="size-3.5"
              aria-hidden="true"
            />
          </button>
          <button
            type="button"
            aria-label={t("thread.exportAria", { title: thread.title })}
            title={t("thread.exportJson")}
            onClick={() => runExport(thread)}
            disabled={exportBusy}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          >
            {exportBusy ? (
              <Loader2
                className="size-3.5 animate-spin"
                aria-hidden="true"
              />
            ) : (
              <Download
                className="size-3.5"
                aria-hidden="true"
              />
            )}
          </button>
          <button
            type="button"
            aria-label={t("thread.deleteAria", { title: thread.title })}
            title={t("thread.delete")}
            onClick={() => setDeleteTarget(thread)}
            className="rounded p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring"
          >
            <Trash2
              className="size-3.5"
              aria-hidden="true"
            />
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="absolute inset-0 flex flex-col">
      <button
        ref={newChatRef}
        type="button"
        onClick={() => {
          if (onNewChat) {
            onNewChat();
          } else {
            setThreadId(null);
            setView(null);
          }
          onClose?.();
        }}
        className="flex flex-shrink-0 items-center gap-2.5 border-b border-border px-3 py-3 text-left text-sm font-medium transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
      >
        <SquarePen
          className="size-4"
          aria-hidden="true"
        />
        {t("thread.newChat")}
      </button>
      <button
        type="button"
        onClick={() => {
          if (view === "skills") {
            setView(null);
            onClose?.();
            return;
          }
          setView("skills");
          onClose?.();
        }}
        className={cn(
          "flex flex-shrink-0 items-center gap-2.5 border-b border-border px-3 py-3 text-left text-sm font-medium transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
          view === "skills" && "bg-accent"
        )}
      >
        <Puzzle
          className="size-4"
          aria-hidden="true"
        />
        {t("thread.researchSkills")}
      </button>
      <button
        type="button"
        onClick={() => {
          if (view === "memory") {
            setView(null);
            onClose?.();
            return;
          }
          markMemorySeen();
          setView("memory");
          onClose?.();
        }}
        className={cn(
          "flex flex-shrink-0 items-center gap-2.5 border-b border-border px-3 py-3 text-left text-sm font-medium transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
          view === "memory" && "bg-accent"
        )}
      >
        <BrainCircuit
          className="size-4"
          aria-hidden="true"
        />
        EvoMemory
        {view !== "memory" && memoryUnseen > 0 && (
          <span
            className="ml-auto inline-flex min-w-4 items-center justify-center rounded-full bg-[var(--brand-solid)] px-1 text-[10px] font-bold leading-none text-[var(--brand-foreground)]"
            title={t("thread.memoryUpdatesTitle", { count: memoryUnseen })}
            aria-label={t("thread.memoryUpdatesAria", { count: memoryUnseen })}
          >
            {memoryUnseen}
          </span>
        )}
      </button>
      <button
        type="button"
        onClick={() => {
          if (view === "schedule") {
            setView(null);
            onClose?.();
            return;
          }
          setView("schedule");
          onClose?.();
        }}
        className={cn(
          "flex flex-shrink-0 items-center gap-2.5 border-b border-border px-3 py-3 text-left text-sm font-medium transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
          view === "schedule" && "bg-accent"
        )}
      >
        <Clock
          className="size-4"
          aria-hidden="true"
        />
        {t("thread.scheduled")}
      </button>
      <div className="flex-shrink-0 border-b border-border p-2.5">
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <input
            type="search"
            name="research-search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t("thread.searchPlaceholder")}
            aria-label={t("thread.searchAria")}
            autoComplete="off"
            spellCheck={false}
            className="w-full rounded-md border border-border bg-background py-1.5 pl-8 pr-8 text-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring"
          />
          {search && (
            <button
              type="button"
              aria-label={t("thread.clearSearchAria")}
              onClick={() => setSearch("")}
              className="absolute right-1 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
            >
              <X
                className="size-3.5"
                aria-hidden="true"
              />
            </button>
          )}
        </div>
      </div>
      <div className="grid flex-shrink-0 grid-cols-[1fr_auto] items-center gap-2 border-b border-border px-3 py-2.5">
        <h2 className="text-base font-semibold tracking-tight">{t("thread.recents")}</h2>
        <div className="flex items-center gap-2">
          <Select
            value={statusFilter}
            onValueChange={(v) => setStatusFilter(v as StatusFilter)}
          >
            <SelectTrigger className="w-fit">
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="all">{t("thread.filter.all")}</SelectItem>
              <SelectSeparator />
              <SelectGroup>
                <SelectLabel>{t("thread.filter.active")}</SelectLabel>
                <SelectItem value="idle">
                  <StatusFilterItem
                    status="idle"
                    label={t("thread.status.idle")}
                  />
                </SelectItem>
                <SelectItem value="busy">
                  <StatusFilterItem
                    status="busy"
                    label={t("thread.status.busy")}
                  />
                </SelectItem>
              </SelectGroup>
              <SelectSeparator />
              <SelectGroup>
                <SelectLabel>{t("thread.filter.attention")}</SelectLabel>
                <SelectItem value="interrupted">
                  <StatusFilterItem
                    status="interrupted"
                    label={t("thread.status.interrupted")}
                    badge={interruptedCount}
                  />
                </SelectItem>
                <SelectItem value="error">
                  <StatusFilterItem
                    status="error"
                    label={t("thread.status.error")}
                  />
                </SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          {onClose && (
            <button
              type="button"
              aria-label={view ? t("thread.closeNavigation") : t("thread.closeResearch")}
              onClick={onClose}
              className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
            >
              <X
                className="size-4"
                aria-hidden="true"
              />
            </button>
          )}
        </div>
      </div>

      <ScrollArea className="h-0 flex-1">
        {threads.error && <ErrorState message={threads.error.message} title={t("thread.error.loadResearch")} />}

        {!threads.error && !threads.data && threads.isLoading && (
          <div aria-live="polite">
            <LoadingState />
          </div>
        )}

        {!threads.error && !threads.isLoading && isEmpty && <EmptyState label={t("thread.empty")} />}

        {!threads.error &&
          !isEmpty &&
          search.trim() &&
          filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center p-8 text-center">
              <p className="text-sm text-muted-foreground">
                {t("thread.noSearchResults")}
              </p>
            </div>
          )}

        {!threads.error && !isEmpty && filtered.length > 0 && (
          <div className="box-border w-full max-w-full overflow-hidden p-1.5">
            {/* Pinned threads — shown only when at least one thread is pinned. */}
            {pinned.length > 0 && (
              <div className="mb-3">
                <h4 className="m-0 px-2.5 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {t("thread.pinnedResearch")}
                </h4>
                <div className="flex flex-col gap-1">
                  {pinned.map((thread) => renderThreadCard(thread))}
                </div>
              </div>
            )}

            {(
              Object.keys(GROUP_LABELS) as Array<keyof typeof GROUP_LABELS>
            ).map((group) => {
              const groupThreads = grouped[group];
              if (groupThreads.length === 0) return null;

              return (
                <div
                  key={group}
                  className="mb-3"
                >
                  <h4 className="m-0 px-2.5 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    {t(GROUP_LABELS[group])}
                  </h4>
                  <div className="flex flex-col gap-1">
                    {groupThreads.map((thread) => renderThreadCard(thread))}
                  </div>
                </div>
              );
            })}

            {!isReachingEnd && (
              <div className="flex justify-center py-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => threads.setSize(threads.size + 1)}
                  disabled={isLoadingMore}
                >
                  {isLoadingMore ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {t("thread.loading")}
                    </>
                  ) : (
                    t("thread.loadMore")
                  )}
                </Button>
              </div>
            )}
          </div>
        )}
      </ScrollArea>

      {/* Rename dialog */}
      <Dialog
        open={renameTarget !== null}
        onOpenChange={(open) => {
          // Don't let Escape / backdrop close the dialog mid-save.
          if (!open && !actionBusy) setRenameTarget(null);
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("thread.renameTitle")}</DialogTitle>
            <DialogDescription>
              {t("thread.renameDescription")}
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submitRename();
              }
            }}
            placeholder={t("thread.renamePlaceholder")}
            maxLength={100}
            disabled={actionBusy}
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setRenameTarget(null)}
              disabled={actionBusy}
            >
              {t("thread.cancel")}
            </Button>
            <Button
              onClick={submitRename}
              disabled={actionBusy || !renameValue.trim()}
            >
              {actionBusy ? t("thread.saving") : t("thread.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete confirmation */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          // Don't let Escape / backdrop close the dialog mid-delete.
          if (!open && !actionBusy) setDeleteTarget(null);
        }}
      >
        <DialogContent
          className="sm:max-w-md"
          onCloseAutoFocus={(e) => {
            // After a delete the trigger row is gone — send focus to New Chat
            // instead of letting it fall to <body>.
            if (pendingDeleteFocusRef.current) {
              e.preventDefault();
              pendingDeleteFocusRef.current = false;
              newChatRef.current?.focus();
            }
          }}
        >
          <DialogHeader>
            <DialogTitle>{t("thread.deleteTitle")}</DialogTitle>
            <DialogDescription>
              {t("thread.deleteDescription", { title: deleteTarget?.title ?? "" })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteTarget(null)}
              disabled={actionBusy}
            >
              {t("thread.cancel")}
            </Button>
            <Button
              onClick={confirmDelete}
              disabled={actionBusy}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {actionBusy ? t("thread.deleting") : t("thread.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

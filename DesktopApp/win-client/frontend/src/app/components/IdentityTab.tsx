"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  type LucideIcon,
  Compass,
  FolderOpen,
  Loader2,
  Pencil,
  Sparkles,
  User,
} from "lucide-react";
import { MarkdownContent } from "@/app/components/MarkdownContent";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { cn } from "@/lib/utils";
import { useI18n } from "@/providers/I18nProvider";

interface IdentityTabProps {
  listing: { entries: Array<{ path: string }> } | null;
  listingLoading: boolean;
}

interface ProfileEntry {
  path: string;
  label: string;
  description: string;
  Icon: LucideIcon;
}

const CORE_ENTRY_DEFS = [
  {
    path: "profile/USER_PROFILE.md",
    labelKey: "memory.identity.user",
    descriptionKey: "memory.identity.userDescription",
    Icon: User,
  },
  {
    path: "profile/RESEARCH_TASTE.md",
    labelKey: "memory.identity.taste",
    descriptionKey: "memory.identity.tasteDescription",
    Icon: Compass,
  },
  {
    path: "profile/SOUL.md",
    labelKey: "memory.identity.soul",
    descriptionKey: "memory.identity.soulDescription",
    Icon: Sparkles,
  },
] as const;

function ProfileRow({
  entry,
  selected,
  onSelect,
}: {
  entry: ProfileEntry;
  selected: boolean;
  onSelect: (e: ProfileEntry) => void;
}) {
  const { Icon } = entry;

  return (
    <button
      type="button"
      onClick={() => onSelect(entry)}
      className={cn(
        "group flex w-full min-w-0 items-center gap-2 rounded-md border px-3 py-2.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected
          ? "border-[var(--brand)]/40 bg-accent"
          : "hover:border-[var(--brand)]/40 border-border bg-[var(--color-surface)] hover:bg-accent/60"
      )}
    >
      <span
        className={cn(
          "flex size-7 shrink-0 items-center justify-center rounded-md transition-colors",
          selected
            ? "bg-[var(--brand)] text-white"
            : "bg-muted text-[var(--brand)] group-hover:bg-background"
        )}
      >
        <Icon
          className="size-4"
          aria-hidden="true"
        />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">
          {entry.label}
        </span>
        <span className="block truncate text-[11px] text-muted-foreground">
          {entry.description}
        </span>
      </span>
    </button>
  );
}

function ContentPanel({ entry }: { entry: ProfileEntry | null }) {
  const { t } = useI18n();
  const tRef = useRef(t);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  const [content, setContent] = useState("");
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reqRef = useRef(0);

  useEffect(() => {
    if (!entry) {
      setContent("");
      setDraft("");
      return;
    }
    const reqId = ++reqRef.current;
    setLoading(true);
    setEditing(false);
    setError(null);
    fetch(`/api/memory?path=${encodeURIComponent(entry.path)}`)
      .then(async (r) => {
        const d = (await r.json().catch(() => ({}))) as {
          content?: string;
          error?: string;
        };
        if (!r.ok) {
          throw new Error(d.error || tRef.current("memory.identity.error.load"));
        }
        return d;
      })
      .then((d) => {
        if (reqRef.current !== reqId) return;
        setContent(d.content ?? "");
        setDraft(d.content ?? "");
      })
      .catch((e: unknown) => {
        if (reqRef.current !== reqId) return;
        setContent("");
        setDraft("");
        setError(e instanceof Error ? e.message : tRef.current("memory.identity.error.load"));
      })
      .finally(() => {
        if (reqRef.current === reqId) setLoading(false);
      });
  }, [entry]);

  const save = async () => {
    if (!entry) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/memory", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: entry.path, content: draft }),
      });
      const d = (await res.json()) as { error?: string };
      if (!res.ok) throw new Error(d.error ?? t("memory.identity.error.save"));
      setContent(draft);
      setEditing(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("memory.identity.error.save"));
    } finally {
      setSaving(false);
    }
  };

  if (!entry) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        {t("memory.identity.selectProfile")}
      </div>
    );
  }

  const body = content.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "").trim();

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex flex-shrink-0 items-center justify-between border-b border-border px-4 py-2.5">
        <span className="text-sm font-medium">{entry.label}</span>
        <div className="flex items-center gap-1.5">
          {editing ? (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setEditing(false);
                  setDraft(content);
                }}
              >
                {t("memory.identity.cancel")}
              </Button>
              <Button
                size="sm"
                onClick={save}
                disabled={saving || draft === content}
              >
                {saving ? (
                  <>
                    <Loader2
                      className="size-3.5 animate-spin"
                      aria-hidden="true"
                    />
                    {t("memory.identity.saving")}
                  </>
                ) : (
                  t("memory.identity.save")
                )}
              </Button>
            </>
          ) : (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setEditing(true)}
              disabled={loading}
            >
              <Pencil
                className="size-3.5"
                aria-hidden="true"
              />
              {t("memory.identity.editRaw")}
            </Button>
          )}
        </div>
      </div>

      {error && (
        <p className="flex-shrink-0 px-4 py-2 text-xs text-destructive">
          {error}
        </p>
      )}

      <div className="min-h-0 flex-1 overflow-hidden">
        {loading ? (
          <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2
              className="size-4 animate-spin"
              aria-hidden="true"
            />
            {t("memory.identity.loading")}
          </div>
        ) : editing ? (
          <textarea
            aria-label={t("memory.identity.editAria", { label: entry.label })}
            className="size-full resize-none bg-background p-4 font-mono text-sm text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            spellCheck={false}
          />
        ) : (
          <ScrollArea className="h-full">
            <div className="px-4 py-4">
              <MarkdownContent content={body} />
            </div>
          </ScrollArea>
        )}
      </div>
    </div>
  );
}

export function IdentityTab({ listing, listingLoading }: IdentityTabProps) {
  const { t } = useI18n();
  const [selected, setSelected] = useState<ProfileEntry | null>(null);
  const [isDesktop, setIsDesktop] = useState(false);

  const coreEntries: ProfileEntry[] = useMemo(
    () =>
      CORE_ENTRY_DEFS.map((entry) => ({
        path: entry.path,
        label: t(entry.labelKey),
        description: t(entry.descriptionKey),
        Icon: entry.Icon,
      })),
    [t]
  );
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)");
    const update = () => setIsDesktop(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  const projectEntries: ProfileEntry[] = useMemo(
    () =>
      (listing?.entries ?? [])
        .filter(
          (e) =>
            e.path.startsWith("profile/projects/") && e.path.endsWith(".md")
        )
        .map((e) => {
          const segments = e.path.split("/");
          const id =
            segments.length >= 4
              ? `${segments[2]}/${segments.slice(3).join("/")}`
              : e.path;
          return {
            path: e.path,
            label: id.replace(/\/PROJECT_PROFILE\.md$/, ""),
            description: t("memory.identity.projectProfile"),
            Icon: FolderOpen,
          };
        }),
    [listing?.entries, t]
  );

  const entries = useMemo(
    () => [...coreEntries, ...projectEntries],
    [coreEntries, projectEntries]
  );

  useEffect(() => {
    if (listingLoading || entries.length === 0) return;
    if (!selected || !entries.some((entry) => entry.path === selected.path)) {
      setSelected(entries[0]);
    }
  }, [entries, listingLoading, selected]);

  const sidebar = (
    <ScrollArea className="h-full">
      <div className="p-3">
        <p className="mb-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
          {t("memory.identity.profile")}
        </p>
        <div className="space-y-1.5">
          {coreEntries.map((entry) => (
            <ProfileRow
              key={entry.path}
              entry={entry}
              selected={selected?.path === entry.path}
              onSelect={setSelected}
            />
          ))}
        </div>

        <p className="mb-2 mt-5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
          {t("memory.identity.projectProfiles")}
        </p>
        {listingLoading ? (
          <div className="flex h-10 items-center gap-2 text-xs text-muted-foreground">
            <Loader2
              className="size-3.5 animate-spin"
              aria-hidden="true"
            />
            {t("memory.identity.loadingProjects")}
          </div>
        ) : projectEntries.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            {t("memory.identity.noProjectProfiles")}
          </p>
        ) : (
          <div className="space-y-1.5">
            {projectEntries.map((entry) => (
              <ProfileRow
                key={entry.path}
                entry={entry}
                selected={selected?.path === entry.path}
                onSelect={setSelected}
              />
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  );

  if (!isDesktop) {
    return (
      <div className="flex min-h-0 w-full flex-1 flex-col">
        <div className="max-h-[42svh] flex-shrink-0 overflow-hidden border-b border-border">
          {sidebar}
        </div>
        <div className="min-h-0 flex-1">
          <ContentPanel entry={selected} />
        </div>
      </div>
    );
  }

  return (
    <ResizablePanelGroup
      direction="horizontal"
      autoSaveId="evomemory-identity"
      className="h-full"
    >
      <ResizablePanel
        defaultSize={35}
        minSize={22}
        maxSize={55}
      >
        {sidebar}
      </ResizablePanel>

      <ResizableHandle />

      <ResizablePanel
        defaultSize={65}
        minSize={45}
      >
        <ContentPanel entry={selected} />
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}

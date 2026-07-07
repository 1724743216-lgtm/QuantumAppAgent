"use client";

import React, { useEffect, useRef, useState } from "react";
import { Puzzle, Loader2, CheckCircle2 } from "lucide-react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MarkdownContent } from "@/app/components/MarkdownContent";
import { useI18n } from "@/providers/I18nProvider";

export interface SkillDetailTarget {
  name: string;
  title: string;
  description: string;
  version?: string;
  fileCount?: number;
  installed: boolean;
}

export const SkillDetailDialog = React.memo<{
  skill: SkillDetailTarget | null;
  onClose: () => void;
}>(({ skill, onClose }) => {
  const { t } = useI18n();
  const tRef = useRef(t);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  interface FetchedDetail {
    title?: string;
    description?: string;
    version?: string;
    body?: string;
    installed?: boolean;
  }
  const [detail, setDetail] = useState<FetchedDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!skill) return;
    let cancelled = false;
    setDetail(null);
    setError(null);
    setLoading(true);
    fetch(`/api/skills/detail?name=${encodeURIComponent(skill.name)}`)
      .then(async (res) => {
        const d = await res.json().catch(() => null);
        if (!res.ok)
          throw new Error(d?.error || tRef.current("skills.detail.error.loadWithStatus", { status: res.status }));
        return d as FetchedDetail;
      })
      .then((d) => {
        if (!cancelled) setDetail(d);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message ?? tRef.current("skills.detail.error.load"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [skill]);

  if (!skill) return null;

  // Prefer the fetched detail once it resolves (it reflects the actual installed
  // copy); fall back to the opener data as an optimistic placeholder.
  const title = detail?.title ?? skill.title;
  const version = detail?.version ?? skill.version;
  const description = detail?.description ?? skill.description;
  const installed = detail?.installed ?? skill.installed;
  const body = detail?.body;

  return (
    <Dialog
      open={true}
      onOpenChange={onClose}
    >
      <DialogContent
        aria-describedby={undefined}
        className="flex h-[80vh] max-h-[80vh] min-w-[60vw] flex-col p-6"
      >
        <DialogTitle className="sr-only">{title}</DialogTitle>
        <div className="mb-3 flex items-start gap-3 border-b border-border pb-4">
          <Puzzle
            className="mt-1 size-6 shrink-0 text-[var(--brand)]"
            aria-hidden="true"
          />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
              <h2 className="break-words text-lg font-semibold">{title}</h2>
              {version && (
                <span className="font-mono text-xs text-muted-foreground">
                  v{version}
                </span>
              )}
              {skill.fileCount != null && (
                <span className="text-xs text-muted-foreground">
                  · {t("skills.fileCount", { count: skill.fileCount })}
                </span>
              )}
              {installed && (
                <span className="bg-[var(--brand)]/10 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium text-[var(--brand)]">
                  <CheckCircle2
                    className="size-3"
                    aria-hidden="true"
                  />
                  {t("skills.installed")}
                </span>
              )}
            </div>
          </div>
        </div>

        <ScrollArea className="min-h-0 flex-1">
          <div className="pr-3">
            {/* Full description — always shown, un-truncated. */}
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
              {description || t("skills.noDescription")}
            </p>

            {/* SKILL.md body — what the skill actually does. */}
            <div className="mt-4 border-t border-border pt-4">
              {loading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2
                    className="size-4 animate-spin"
                    aria-hidden="true"
                  />
                  {t("skills.detail.loadingContents")}
                </div>
              ) : error ? (
                <p className="text-sm text-muted-foreground">{error}</p>
              ) : body ? (
                <MarkdownContent content={body} />
              ) : (
                <p className="text-sm text-muted-foreground">
                  {t("skills.detail.noAdditionalContents")}
                </p>
              )}
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
});

SkillDetailDialog.displayName = "SkillDetailDialog";

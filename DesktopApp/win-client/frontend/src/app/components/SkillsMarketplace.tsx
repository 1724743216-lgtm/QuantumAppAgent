"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Puzzle, RotateCw, Trash2 } from "lucide-react";
import {
  SkillDetailDialog,
  type SkillDetailTarget,
} from "@/app/components/SkillDetailDialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useI18n } from "@/providers/I18nProvider";

interface SkillCard {
  name: string;
  title: string;
  description: string;
  dir: string;
  source: "workspace" | "global" | "builtin";
  writable: boolean;
}

export function SkillsMarketplace() {
  const { t } = useI18n();
  const tRef = useRef(t);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  const [skills, setSkills] = useState<SkillCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<Record<string, "uninstall">>({});
  const [detail, setDetail] = useState<SkillDetailTarget | null>(null);
  const [uninstallTarget, setUninstallTarget] = useState<{
    name: string;
    title: string;
  } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const installed = await fetch("/api/skills").then(async (r) => {
        const d = await r.json();
        if (!r.ok) {
          throw new Error(d.error || tRef.current("skills.error.loadSkills"));
        }
        return (d.skills ?? []) as SkillCard[];
      });
      setSkills(installed);
    } catch (e) {
      setSkills([]);
      setError(
        e instanceof Error
          ? e.message
          : tRef.current("skills.error.loadInstalledSkills")
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const uninstall = async (name: string) => {
    setBusy((b) => ({ ...b, [name]: "uninstall" }));
    setError(null);
    try {
      const res = await fetch(`/api/skills?name=${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.error || t("skills.error.uninstall"));
      }
      setSkills((prev) => prev.filter((s) => s.name !== name));
    } catch (e) {
      setError(e instanceof Error ? e.message : t("skills.error.uninstall"));
    } finally {
      setBusy((b) => {
        const next = { ...b };
        delete next[name];
        return next;
      });
    }
  };

  const confirmUninstall = async () => {
    if (!uninstallTarget) return;
    const target = uninstallTarget;
    setUninstallTarget(null);
    await uninstall(target.name);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[960px] px-4 py-5 sm:px-5 sm:py-6">
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold sm:text-2xl">
              {t("skills.title")}
            </h2>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">
              {t("skills.localDescription")}
            </p>
          </div>
          <button
            type="button"
            onClick={() => load()}
            disabled={loading}
            aria-label={t("skills.refresh")}
            className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
          >
            <RotateCw
              className={loading ? "size-4 animate-spin" : "size-4"}
              aria-hidden="true"
            />
          </button>
        </header>

        {error && (
          <p
            role="alert"
            className="mb-4 text-sm text-destructive"
          >
            {error}
          </p>
        )}

        {loading ? (
          <div
            className="flex items-center gap-2 text-sm text-muted-foreground"
            aria-live="polite"
          >
            <Loader2
              className="size-4 animate-spin"
              aria-hidden="true"
            />
            {t("skills.loading")}
          </div>
        ) : (
          <section>
            <h3 className="mb-2.5 text-xs font-semibold uppercase tracking-wider text-tertiary">
              {t("skills.localInstalled")}
            </h3>
            {skills.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                {t("skills.emptyInstalled")}
              </p>
            ) : (
              <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                {skills.map((s) => (
                  <SkillTile
                    key={`${s.source}:${s.name}`}
                    title={s.title}
                    description={s.description}
                    source={s.source}
                    writable={s.writable}
                    busy={busy[s.name]}
                    onOpen={() =>
                      setDetail({
                        name: s.name,
                        title: s.title,
                        description: s.description,
                        installed: true,
                      })
                    }
                    onUninstall={
                      s.writable
                        ? () =>
                            setUninstallTarget({
                              name: s.name,
                              title: s.title,
                            })
                        : undefined
                    }
                  />
                ))}
              </div>
            )}
          </section>
        )}
      </div>

      <SkillDetailDialog
        skill={detail}
        onClose={() => setDetail(null)}
      />
      <Dialog
        open={uninstallTarget !== null}
        onOpenChange={(open) => {
          if (!open) setUninstallTarget(null);
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("skills.uninstallTitle")}</DialogTitle>
            <DialogDescription>
              {t("skills.uninstallDescription", {
                title: uninstallTarget?.title ?? uninstallTarget?.name ?? "",
              })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setUninstallTarget(null)}
            >
              {t("skills.cancel")}
            </Button>
            <Button
              onClick={confirmUninstall}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t("skills.uninstall")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function SkillTile({
  title,
  description,
  source,
  writable,
  busy,
  onOpen,
  onUninstall,
}: {
  title: string;
  description: string;
  source: SkillCard["source"];
  writable: boolean;
  busy?: "uninstall";
  onOpen?: () => void;
  onUninstall?: () => void;
}) {
  const { t } = useI18n();
  const sourceLabel =
    source === "workspace"
      ? t("skills.source.workspace")
      : source === "global"
      ? t("skills.source.global")
      : t("skills.source.builtin");

  return (
    <div className="flex flex-col rounded-lg border border-border bg-card p-3">
      <button
        type="button"
        onClick={onOpen}
        className="-m-1 flex items-start gap-2.5 rounded-md p-1 text-left transition-colors hover:bg-muted/50 focus-visible:ring-2 focus-visible:ring-ring"
        title={t("skills.viewDetails")}
      >
        <Puzzle
          className="mt-0.5 size-5 shrink-0 text-[var(--brand)]"
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <h3 className="break-words text-lg font-medium leading-tight">
              {title}
            </h3>
            <span className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground">
              {sourceLabel}
            </span>
          </div>
          <p className="mt-1 line-clamp-2 text-sm leading-6 text-muted-foreground">
            {description || t("skills.noDescription")}
          </p>
        </div>
      </button>
      <div className="mt-2.5 flex items-center justify-end gap-2">
        {writable ? (
          <button
            type="button"
            onClick={onUninstall}
            disabled={!!busy}
            className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:border-destructive/40 hover:bg-destructive/10 hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
          >
            {busy === "uninstall" ? (
              <Loader2
                className="size-3.5 animate-spin"
                aria-hidden="true"
              />
            ) : (
              <Trash2
                className="size-3.5"
                aria-hidden="true"
              />
            )}
            {busy === "uninstall"
              ? t("skills.removing")
              : t("skills.uninstall")}
          </button>
        ) : (
          <span className="rounded-md border border-border px-2.5 py-1 text-xs font-medium text-muted-foreground">
            {t("skills.readOnly")}
          </span>
        )}
      </div>
    </div>
  );
}

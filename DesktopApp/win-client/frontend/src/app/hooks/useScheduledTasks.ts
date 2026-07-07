"use client";

import { useCallback, useEffect, useState } from "react";
import type { Cron } from "@langchain/langgraph-sdk";
import { Client } from "@langchain/langgraph-sdk";
import { getConfig } from "@/lib/config";
import { useI18n } from "@/providers/I18nProvider";

const SCHEDULED_RUN_KIND = "scheduled_task";
const SCHEDULER_GRAPH_ID = "scheduler";
export const NO_DEPLOYMENT_CONFIGURED_ERROR = "NO_DEPLOYMENT_CONFIGURED";

function makeClient(): Client | null {
  const config = getConfig();
  if (!config) return null;
  const apiKey =
    config.langsmithApiKey || process.env.NEXT_PUBLIC_LANGSMITH_API_KEY || "";
  return new Client({
    apiUrl: config.deploymentUrl,
    defaultHeaders: apiKey ? { "X-Api-Key": apiKey } : {},
  });
}

export interface ScheduledTask {
  cron_id: string;
  name: string;
  prompt: string;
  schedule: string;
  next_run_date: string | null;
  created_at: string;
  updated_at: string;
}

function parseCron(cron: Cron, unnamedTaskLabel: string): ScheduledTask {
  const meta = (cron.metadata ?? {}) as Record<string, unknown>;
  return {
    cron_id: cron.cron_id,
    name:
      typeof meta.name === "string" && meta.name ? meta.name : unnamedTaskLabel,
    prompt: typeof meta.prompt === "string" ? meta.prompt : "",
    schedule: cron.schedule,
    next_run_date: cron.next_run_date ?? null,
    created_at: cron.created_at,
    updated_at: cron.updated_at,
  };
}

export async function listScheduledTasks(
  unnamedTaskLabel: string
): Promise<ScheduledTask[]> {
  const client = makeClient();
  if (!client) return [];
  // The TS SDK's crons.search doesn't support metadata filtering — fetch all
  // and filter client-side. Only EvoScientist crons carry run_kind in metadata.
  const crons = await client.crons.search({ limit: 200 });
  return crons
    .filter(
      (c) =>
        (c.metadata as Record<string, unknown>)?.run_kind === SCHEDULED_RUN_KIND
    )
    .map((cron) => parseCron(cron, unnamedTaskLabel));
}

export async function createScheduledTask(params: {
  name: string;
  prompt: string;
  schedule: string;
}): Promise<ScheduledTask> {
  const client = makeClient();
  if (!client) throw new Error(NO_DEPLOYMENT_CONFIGURED_ERROR);
  const cron = await client.crons.create(SCHEDULER_GRAPH_ID, {
    input: { messages: [{ role: "user", content: params.prompt }] },
    schedule: params.schedule,
    metadata: {
      run_kind: SCHEDULED_RUN_KIND,
      name: params.name,
      prompt: params.prompt,
    },
  });
  return parseCron(cron as unknown as Cron, params.name);
}

export async function deleteScheduledTask(cronId: string): Promise<void> {
  const client = makeClient();
  if (!client) throw new Error(NO_DEPLOYMENT_CONFIGURED_ERROR);
  await client.crons.delete(cronId);
}

export async function updateScheduledTask(params: {
  cronId: string;
  name: string;
  prompt: string;
  schedule: string;
}): Promise<{ task: ScheduledTask; oldTaskDeleted: boolean }> {
  const task = await createScheduledTask({
    name: params.name,
    prompt: params.prompt,
    schedule: params.schedule,
  });

  try {
    await deleteScheduledTask(params.cronId);
    return { task, oldTaskDeleted: true };
  } catch {
    return { task, oldTaskDeleted: false };
  }
}

export async function runScheduledTaskNow(prompt: string): Promise<void> {
  const client = makeClient();
  if (!client) throw new Error(NO_DEPLOYMENT_CONFIGURED_ERROR);
  const thread = await client.threads.create({ graphId: SCHEDULER_GRAPH_ID });
  await client.runs.create(thread.thread_id, SCHEDULER_GRAPH_ID, {
    input: { messages: [{ role: "user", content: prompt }] },
    metadata: {
      run_kind: SCHEDULED_RUN_KIND,
      name: "manual-run",
      prompt,
    },
  });
}

export function useScheduledTasks(): {
  tasks: ScheduledTask[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
} {
  const { t } = useI18n();
  const [tasks, setTasks] = useState<ScheduledTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rev, setRev] = useState(0);

  const refresh = useCallback(() => setRev((r) => r + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listScheduledTasks(t("schedule.unnamedTask"))
      .then((result) => {
        if (!cancelled) {
          setTasks(result);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : t("schedule.error.loadTasks")
          );
          setTasks([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [rev, t]);

  return { tasks, loading, error, refresh };
}

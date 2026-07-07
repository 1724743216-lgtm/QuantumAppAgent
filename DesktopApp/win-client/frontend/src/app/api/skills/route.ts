import { NextRequest, NextResponse } from "next/server";
import { join, resolve, sep } from "path";
import { promises as fs } from "fs";
import {
  getSkillDirs,
  recordUninstall,
  isValidSkillName,
  type SkillSource,
} from "@/lib/server/skills";

interface SkillCard {
  /** Directory name - the install/uninstall identity. */
  name: string;
  /** Frontmatter name for display; falls back to the directory name. */
  title: string;
  description: string;
  dir: string;
  source: SkillSource;
  writable: boolean;
}

// Minimal frontmatter parse - we only need name + description. Avoids pulling
// in a YAML dependency.
function parseFrontmatter(md: string): { name?: string; description?: string } {
  const match = md.match(/^---\s*\n([\s\S]*?)\n---/);
  if (!match) return {};
  const fm = match[1];
  const get = (key: string) => {
    const m = fm.match(new RegExp(`^${key}\\s*:\\s*(.+?)\\s*$`, "m"));
    if (!m) return undefined;
    return m[1].replace(/^["']|["']$/g, "").trim();
  };
  return { name: get("name"), description: get("description") };
}

async function safeSkillDir(
  root: string,
  entry: string
): Promise<string | undefined> {
  const realRoot = await fs.realpath(root);
  const candidate = join(root, entry);
  const realDir = await fs.realpath(candidate);
  if (realDir !== realRoot && !realDir.startsWith(realRoot + sep)) {
    return undefined;
  }
  const stat = await fs.stat(realDir);
  return stat.isDirectory() ? realDir : undefined;
}

async function readSkills(): Promise<SkillCard[]> {
  const skills: SkillCard[] = [];
  const seen = new Set<string>();

  for (const skillRoot of await getSkillDirs()) {
    let entries: string[] = [];
    try {
      entries = await fs.readdir(skillRoot.dir);
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.startsWith(".")) continue;
      try {
        const realDir = await safeSkillDir(skillRoot.dir, entry);
        if (!realDir) continue;
        const md = await fs.readFile(join(realDir, "SKILL.md"), "utf-8");
        const { name, description } = parseFrontmatter(md);
        const identity = name || entry;
        if (seen.has(identity)) continue;
        seen.add(identity);
        skills.push({
          name: entry,
          title: identity,
          description: description || "",
          dir: realDir,
          source: skillRoot.source,
          writable: skillRoot.writable,
        });
      } catch {
        // no SKILL.md or unreadable - skip
      }
    }
  }

  return skills.sort((a, b) => a.title.localeCompare(b.title));
}

export async function GET() {
  try {
    const skills = await readSkills();
    return NextResponse.json({ skills });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Failed to read skills" },
      { status: 500 }
    );
  }
}

async function findWritableSkill(
  root: string,
  name: string
): Promise<{ path: string; entry: string } | null> {
  const realRoot = await fs.realpath(root);
  const direct = resolve(join(root, name));
  if (direct === resolve(root) || !direct.startsWith(resolve(root) + sep)) {
    return null;
  }

  try {
    const realDir = await fs.realpath(direct);
    if (realDir === realRoot || !realDir.startsWith(realRoot + sep)) {
      return null;
    }
    const stat = await fs.stat(realDir);
    if (stat.isDirectory()) return { path: realDir, entry: name };
  } catch {
    // Fall through to frontmatter-name lookup.
  }

  const entries = await fs.readdir(root);
  for (const entry of entries) {
    if (entry.startsWith(".")) continue;
    try {
      const realDir = await safeSkillDir(root, entry);
      if (!realDir) continue;
      const md = await fs.readFile(join(realDir, "SKILL.md"), "utf-8");
      if (parseFrontmatter(md).name === name) {
        return { path: realDir, entry };
      }
    } catch {
      // not a valid skill
    }
  }
  return null;
}

// Uninstall = remove a user-manageable skill directory. Built-in backend skills
// are read-only and are intentionally excluded from this search.
export async function DELETE(req: NextRequest) {
  const name = req.nextUrl.searchParams.get("name");
  if (!name || !isValidSkillName(name)) {
    return NextResponse.json({ error: "Invalid skill name" }, { status: 400 });
  }

  for (const skillRoot of await getSkillDirs({
    includeBuiltin: false,
    writableOnly: true,
  })) {
    try {
      const target = await findWritableSkill(skillRoot.dir, name);
      if (!target) continue;
      await fs.rm(target.path, { recursive: true, force: true });
      if (skillRoot.source === "global") {
        await recordUninstall(target.entry).catch(() => {});
      }
      return NextResponse.json({ ok: true });
    } catch {
      // Try the next writable tier.
    }
  }

  return NextResponse.json({ error: "Skill not found" }, { status: 404 });
}

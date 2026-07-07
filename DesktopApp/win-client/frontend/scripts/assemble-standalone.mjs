import { cp, rm, readdir } from "fs/promises";
import { existsSync, readdirSync } from "fs";
import { join, dirname } from "path";

let STANDALONE = ".next/standalone";

// Auto-discover the directory containing server.js inside .next/standalone
function findServerJs(dir) {
  const entries = readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isDirectory()) {
      const res = findServerJs(join(dir, entry.name));
      if (res) return res;
    } else if (entry.name === "server.js") {
      return dir;
    }
  }
  return null;
}

if (!existsSync(join(STANDALONE, "server.js"))) {
  const found = findServerJs(STANDALONE);
  if (found) {
    STANDALONE = found;
  }
}

const STATIC = ".next/static";
const PUBLIC = "public";
const OUT = "dist";

if (!existsSync(STANDALONE) || !existsSync(join(STANDALONE, "server.js"))) {
  console.error(
    `✗ server.js not found in ${STANDALONE}. Did "next build" run with output:"standalone"?`
  );
  process.exit(1);
}

await rm(OUT, { recursive: true, force: true });
await cp(STANDALONE, OUT, { recursive: true });
await cp(STATIC, join(OUT, ".next/static"), { recursive: true });
if (existsSync(PUBLIC)) {
  await cp(PUBLIC, join(OUT, "public"), { recursive: true });
}

const KEEP = new Set([
  "server.js",
  ".next",
  "node_modules",
  "package.json",
  "public",
]);
for (const entry of await readdir(OUT)) {
  if (!KEEP.has(entry)) {
    await rm(join(OUT, entry), { recursive: true, force: true });
  }
}

let strippedMaps = 0;
for (const entry of await readdir(OUT, { recursive: true })) {
  if (entry.endsWith(".map")) {
    await rm(join(OUT, entry), { force: true });
    strippedMaps++;
  }
}

console.log(
  `✓ Assembled standalone server into ${OUT}/ ` +
    `(${strippedMaps} source map(s) stripped; run: node ${OUT}/server.js)`
);

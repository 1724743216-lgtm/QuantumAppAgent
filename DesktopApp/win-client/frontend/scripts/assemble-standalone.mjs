import { cp, rm, readdir } from "fs/promises";
import { existsSync } from "fs";

let STANDALONE = ".next/standalone";
if (!existsSync(`${STANDALONE}/server.js`)) {
  if (existsSync(`${STANDALONE}/frontend/server.js`)) {
    STANDALONE = `${STANDALONE}/frontend`;
  } else if (existsSync(`${STANDALONE}/DesktopApp/win-client/frontend/server.js`)) {
    STANDALONE = `${STANDALONE}/DesktopApp/win-client/frontend`;
  }
}

const STATIC = ".next/static";
const PUBLIC = "public";
const OUT = "dist";

if (!existsSync(STANDALONE)) {
  console.error(
    `✗ ${STANDALONE} not found. Did "next build" run with output:"standalone"?`
  );
  process.exit(1);
}

await rm(OUT, { recursive: true, force: true });
await cp(STANDALONE, OUT, { recursive: true });
await cp(STATIC, `${OUT}/.next/static`, { recursive: true });
if (existsSync(PUBLIC)) {
  await cp(PUBLIC, `${OUT}/public`, { recursive: true });
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
    await rm(`${OUT}/${entry}`, { recursive: true, force: true });
  }
}

let strippedMaps = 0;
for (const entry of await readdir(OUT, { recursive: true })) {
  if (entry.endsWith(".map")) {
    await rm(`${OUT}/${entry}`, { force: true });
    strippedMaps++;
  }
}

console.log(
  `✓ Assembled standalone server into ${OUT}/ ` +
    `(${strippedMaps} source map(s) stripped; run: node ${OUT}/server.js)`
);

// Bundles static/editor/editor.js (CodeMirror 6 with Python) into one
// minified script. Usage: node scripts/editor.mjs OUTFILE
import { build } from "esbuild";

const outfile = process.argv[2];
if (!outfile) {
  console.error("usage: node scripts/editor.mjs OUTFILE");
  process.exit(2);
}
await build({
  entryPoints: [new URL("../static/editor/editor.js", import.meta.url).pathname],
  bundle: true,
  minify: true,
  format: "iife",
  target: "es2020",
  outfile,
  legalComments: "none",
  logLevel: "warning",
});

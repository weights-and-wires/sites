// Renders TeX to HTML at build time. Reads a JSON array of
// {tex, display, where} from stdin and writes a JSON array of HTML strings.
// Any formula KaTeX can't parse fails the build.
import katex from "katex";

let input = "";
for await (const chunk of process.stdin) input += chunk;

const items = JSON.parse(input);
const failures = [];
const html = items.map((item) => {
  try {
    return katex.renderToString(item.tex, {
      displayMode: item.display,
      output: "htmlAndMathml",
      throwOnError: true,
      strict: "warn",
    });
  } catch (err) {
    failures.push(`${item.where}: ${err.message}\n    ${item.tex}`);
    return "";
  }
});

if (failures.length > 0) {
  console.error(`KaTeX failed on ${failures.length} formula(s):\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
process.stdout.write(JSON.stringify(html));

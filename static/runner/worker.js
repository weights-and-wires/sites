// Runs exercise tests in Pyodide, off the main thread. The page creates this
// worker on the first "Run tests" click, so Pyodide downloads only then.
//
// Messages in:  {id, harnessURL, submission, tests} to run an exercise's tests,
//               or {id, harnessURL, code} to run a deep-dive cell.
// Messages out: {id, type: "status", text} while loading,
//               {id, type: "result", results, stdout} for tests,
//               {id, type: "output", stdout, result} for a cell, where result
//               is the JSON of the cell's global `result`, or null,
//               or {id, type: "error", text}
let pyodide = null;
let booting = null;

async function boot(harnessURL, say) {
  say("Loading Python (first run only, about 15 MB)…");
  importScripts("/pyodide/pyodide.js");
  const py = await loadPyodide({ indexURL: "/pyodide/", fullStdLib: false });
  say("Loading NumPy…");
  await py.loadPackage("numpy", { messageCallback: () => {}, errorCallback: () => {} });
  const harness = await (await fetch(harnessURL)).text();
  py.FS.mkdirTree("/work");
  py.FS.writeFile("/work/harness.py", harness);
  py.runPython("import sys; sys.path.insert(0, '/work')");
  return py;
}

// runCell runs code in a fresh namespace and returns what it printed, with a
// traceback when it raises, plus its global `result` as JSON.
function runCell(py, code) {
  let stdout = "";
  py.setStdout({ batched: (s) => { stdout += s + "\n"; } });
  py.setStderr({ batched: (s) => { stdout += s + "\n"; } });
  py.globals.set("__cell_code", code);
  const json = py.runPython(`
import json, traceback
__ns = {"__name__": "__main__"}
try:
    exec(compile(__cell_code, "cell.py", "exec"), __ns)
except Exception:
    traceback.print_exc()
__r = __ns.get("result")
try:
    __out = json.dumps(__r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
except Exception:
    __out = "null"
__out`);
  return { stdout: stdout.slice(0, 20000), result: JSON.parse(json) };
}

self.onmessage = async (event) => {
  const { id, harnessURL, submission, tests, code } = event.data;
  const say = (text) => self.postMessage({ id, type: "status", text });
  try {
    if (!pyodide) {
      booting = booting || boot(harnessURL, say);
      pyodide = await booting;
    }
    if (typeof code === "string") {
      say("Running…");
      self.postMessage({ id, type: "output", ...runCell(pyodide, code) });
      return;
    }
    say("Running tests…");
    pyodide.FS.writeFile("/work/submission.py", submission);
    pyodide.FS.writeFile("/work/tests.py", tests);
    let stdout = "";
    pyodide.setStdout({ batched: (s) => { stdout += s + "\n"; } });
    pyodide.setStderr({ batched: (s) => { stdout += s + "\n"; } });
    const json = pyodide.runPython("import json, harness\njson.dumps(harness.run())");
    self.postMessage({ id, type: "result", results: JSON.parse(json), stdout: stdout.slice(0, 20000) });
  } catch (err) {
    self.postMessage({ id, type: "error", text: String(err && err.message ? err.message : err) });
  }
};

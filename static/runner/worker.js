// Runs exercise tests in Pyodide, off the main thread. The page creates this
// worker on the first "Run tests" click, so Pyodide downloads only then.
//
// Messages in:  {id, harnessURL, submission, tests}
// Messages out: {id, type: "status", text} while loading,
//               {id, type: "result", results, stdout} or {id, type: "error", text}
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

self.onmessage = async (event) => {
  const { id, harnessURL, submission, tests } = event.data;
  const say = (text) => self.postMessage({ id, type: "status", text });
  try {
    if (!pyodide) {
      booting = booting || boot(harnessURL, say);
      pyodide = await booting;
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

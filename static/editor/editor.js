// Source for the CodeMirror 6 bundle. scripts/editor.mjs bundles this file
// with esbuild into one self-hosted script that defines window.PapersEditor.
import { EditorState } from "@codemirror/state";
import {
  EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection,
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { indentUnit, bracketMatching, syntaxHighlighting, HighlightStyle } from "@codemirror/language";
import { python } from "@codemirror/lang-python";
import { tags as t } from "@lezer/highlight";

// The code slab colours come from the --code-* and --hl-* design tokens.
const theme = EditorView.theme({
  "&": { color: "var(--code-ink)", backgroundColor: "var(--code-bg)", fontSize: "13px" },
  ".cm-content": { fontFamily: "var(--mono)", caretColor: "var(--code-ink)", padding: "12px 0" },
  ".cm-gutters": { backgroundColor: "var(--code-bg)", color: "var(--code-gutter)", border: "none" },
  ".cm-activeLine": { backgroundColor: "var(--code-active-line)" },
  ".cm-activeLineGutter": { backgroundColor: "var(--code-active-line)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": { backgroundColor: "var(--code-selection)" },
  "&.cm-focused": { outline: "2px solid var(--accent)", outlineOffset: "-2px" },
  ".cm-cursor": { borderLeftColor: "var(--code-ink)" },
  ".cm-scroller": { lineHeight: "1.65" },
}, { dark: true });

const highlight = HighlightStyle.define([
  { tag: [t.keyword, t.operatorKeyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword], color: "var(--hl-keyword)" },
  { tag: [t.string, t.special(t.string)], color: "var(--hl-string)" },
  { tag: [t.number, t.bool, t.null], color: "var(--hl-number)" },
  { tag: t.comment, color: "var(--hl-comment)", fontStyle: "italic" },
  { tag: [t.function(t.variableName), t.function(t.propertyName), t.definition(t.function(t.variableName))], color: "var(--hl-title)" },
  { tag: [t.className, t.typeName], color: "var(--hl-type)" },
  { tag: [t.standard(t.variableName), t.self], color: "var(--hl-builtin)" },
  { tag: [t.punctuation, t.operator, t.bracket], color: "var(--hl-punct)" },
  { tag: t.meta, color: "var(--hl-meta)" },
]);

// mount replaces el's contents with an editor holding doc. onChange gets the
// new text after every edit. It returns {get, set, view}.
function mount(el, doc, { onChange, label, readOnly } = {}) {
  const view = new EditorView({
    parent: el,
    state: EditorState.create({
      doc,
      extensions: [
        lineNumbers(), highlightActiveLineGutter(), highlightActiveLine(), drawSelection(), history(),
        bracketMatching(), indentUnit.of("    "), python(), theme, syntaxHighlighting(highlight),
        keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
        EditorState.readOnly.of(!!readOnly),
        EditorView.contentAttributes.of({ "aria-label": label || "Code editor" }),
        EditorView.updateListener.of((u) => { if (u.docChanged && onChange) onChange(u.state.doc.toString()); }),
      ],
    }),
  });
  return {
    view,
    get: () => view.state.doc.toString(),
    set: (text) => view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } }),
  };
}

window.PapersEditor = { mount };

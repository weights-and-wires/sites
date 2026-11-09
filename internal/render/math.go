// Package render turns markdown with TeX math into HTML. Math is rendered
// once, at build time, by KaTeX in Node, so pages ship no math JavaScript.
package render

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
)

// Math is one formula pulled out of the markdown.
type Math struct {
	TeX     string `json:"tex"`
	Display bool   `json:"display"`
	// Where says which page the formula came from, for error messages.
	Where string `json:"where"`
}

const (
	inlineToken  = "KATEXINLINEQZ%dQZ"
	displayToken = "KATEXDISPLAYQZ%dQZ"
)

var (
	fenceLine  = regexp.MustCompile("^\\s*(```|~~~)")
	displayRe  = regexp.MustCompile(`(?s)\$\$(.+?)\$\$`)
	tokenRe    = regexp.MustCompile(`KATEX(INLINE|DISPLAY)QZ(\d+)QZ`)
	displayPar = regexp.MustCompile(`<p>(KATEXDISPLAYQZ\d+QZ)</p>`)
)

// ExtractMath replaces $...$ and $$...$$ outside code with placeholder tokens
// that goldmark passes through untouched. It appends each formula to items
// and returns the rewritten markdown.
func ExtractMath(src, where string, items *[]Math) string {
	var out strings.Builder
	var prose strings.Builder
	flush := func() {
		out.WriteString(extractProse(prose.String(), where, items))
		prose.Reset()
	}
	inFence := false
	for _, line := range strings.SplitAfter(src, "\n") {
		if fenceLine.MatchString(line) {
			if !inFence {
				flush()
			}
			inFence = !inFence
			out.WriteString(line)
			continue
		}
		if inFence {
			out.WriteString(line)
		} else {
			prose.WriteString(line)
		}
	}
	flush()
	return out.String()
}

func extractProse(src, where string, items *[]Math) string {
	src = displayRe.ReplaceAllStringFunc(src, func(m string) string {
		*items = append(*items, Math{TeX: strings.TrimSpace(m[2 : len(m)-2]), Display: true, Where: where})
		return "\n\n" + fmt.Sprintf(displayToken, len(*items)-1) + "\n\n"
	})
	var b strings.Builder
	runes := []rune(src)
	for i := 0; i < len(runes); i++ {
		c := runes[i]
		switch {
		case c == '\\' && i+1 < len(runes) && runes[i+1] == '$':
			b.WriteString(`\$`)
			i++
		case c == '`':
			// Copy an inline code span verbatim.
			j := i
			for j < len(runes) && runes[j] == '`' {
				j++
			}
			ticks := string(runes[i:j])
			end := strings.Index(string(runes[j:]), ticks)
			if end < 0 {
				b.WriteString(ticks)
				i = j - 1
				continue
			}
			endRunes := len([]rune(string(runes[j:])[:end]))
			b.WriteString(string(runes[i : j+endRunes+len(ticks)]))
			i = j + endRunes + len(ticks) - 1
		case c == '$' && i+1 < len(runes) && runes[i+1] != ' ' && runes[i+1] != '$':
			end := -1
			for k := i + 1; k < len(runes); k++ {
				if runes[k] == '\n' {
					break
				}
				if runes[k] == '$' && runes[k-1] != '\\' && runes[k-1] != ' ' {
					end = k
					break
				}
			}
			if end < 0 {
				b.WriteRune(c)
				continue
			}
			*items = append(*items, Math{TeX: string(runes[i+1 : end]), Where: where})
			b.WriteString(fmt.Sprintf(inlineToken, len(*items)-1))
			i = end
		default:
			b.WriteRune(c)
		}
	}
	return b.String()
}

// RestoreMath swaps placeholder tokens in rendered HTML for KaTeX output.
func RestoreMath(html string, rendered []string) string {
	html = displayPar.ReplaceAllString(html, "$1")
	return tokenRe.ReplaceAllStringFunc(html, func(tok string) string {
		m := tokenRe.FindStringSubmatch(tok)
		var n int
		fmt.Sscanf(m[2], "%d", &n)
		if n >= len(rendered) {
			return tok
		}
		if m[1] == "DISPLAY" {
			return `<div class="math-display">` + rendered[n] + `</div>`
		}
		return rendered[n]
	})
}

// KaTeX renders every formula in one Node process. A formula KaTeX can't
// parse fails the build with the page it came from.
func KaTeX(script string, items []Math) ([]string, error) {
	if len(items) == 0 {
		return nil, nil
	}
	in, err := json.Marshal(items)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command("node", script)
	cmd.Stdin = bytes.NewReader(in)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("katex: %v\n%s", err, stderr.String())
	}
	var out []string
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		return nil, fmt.Errorf("katex output: %w", err)
	}
	if len(out) != len(items) {
		return nil, fmt.Errorf("katex returned %d formulas, want %d", len(out), len(items))
	}
	return out, nil
}

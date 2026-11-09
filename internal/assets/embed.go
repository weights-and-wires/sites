// Package assets embeds the built sites. Run `go run ./cmd/build` first; it
// writes one directory per site under public/.
package assets

import (
	"embed"
	"io/fs"
)

//go:embed all:public
var files embed.FS

// Public returns the built sites, rooted at public/.
func Public() fs.FS {
	sub, err := fs.Sub(files, "public")
	if err != nil {
		panic(err)
	}
	return sub
}

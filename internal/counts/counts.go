// Package counts reads lesson counts from each course's GitHub content tree,
// so the hub doesn't go stale when a course adds lessons.
package counts

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path"
	"time"
)

// API is the GitHub REST endpoint. Tests point it at a local server.
var API = "https://api.github.com"

// Count returns the number of files in repo's default branch whose path
// matches glob (path.Match syntax, for example "content/lessons/*.md").
// token can be empty for public repositories.
func Count(ctx context.Context, client *http.Client, repo, glob, token string) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, API+"/repos/"+repo+"/git/trees/HEAD?recursive=1", nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "weights-and-wires-sites-build")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("%s: GitHub returned %s", repo, resp.Status)
	}
	var tree struct {
		Truncated bool `json:"truncated"`
		Tree      []struct {
			Path string `json:"path"`
			Type string `json:"type"`
		} `json:"tree"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tree); err != nil {
		return 0, fmt.Errorf("%s: %w", repo, err)
	}
	if tree.Truncated {
		return 0, fmt.Errorf("%s: tree listing is truncated", repo)
	}
	n := 0
	for _, e := range tree.Tree {
		if e.Type != "blob" {
			continue
		}
		if ok, err := path.Match(glob, e.Path); err != nil {
			return 0, err
		} else if ok {
			n++
		}
	}
	if n == 0 {
		return 0, fmt.Errorf("%s: no files match %s", repo, glob)
	}
	return n, nil
}

package counts

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCount(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/o/public/git/trees/HEAD":
			if r.Header.Get("Authorization") != "Bearer tok" {
				t.Errorf("missing token")
			}
			w.Write([]byte(`{"tree":[
				{"path":"content/lessons/001-a.md","type":"blob"},
				{"path":"content/lessons/002-b.md","type":"blob"},
				{"path":"content/lessons/README","type":"blob"},
				{"path":"content/lessons/sub","type":"tree"},
				{"path":"content/lessons/sub/x.md","type":"blob"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	API = srv.URL

	n, err := Count(context.Background(), srv.Client(), "o/public", "content/lessons/*.md", "tok")
	if err != nil || n != 2 {
		t.Fatalf("Count = %d, %v; want 2", n, err)
	}
	if _, err := Count(context.Background(), srv.Client(), "o/private", "content/lessons/*.md", "tok"); err == nil {
		t.Fatal("want an error for a repo the API can't read")
	}
}

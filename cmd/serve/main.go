// Command serve serves papers.ifkash.dev and courses.ifkash.dev from the
// sites embedded at build time. It routes by the Host header.
package main

import (
	"flag"
	"io/fs"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/weights-and-wires/sites/internal/assets"
	"github.com/weights-and-wires/sites/internal/server"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	dir := flag.String("dir", "", "serve from this directory instead of the embedded sites (development)")
	health := flag.Bool("healthcheck", false, "request /health on -addr and exit 0 if it answers 200 (for Docker HEALTHCHECK)")
	flag.Parse()

	if *health {
		os.Exit(healthcheck(*addr))
	}

	var root fs.FS = assets.Public()
	if *dir != "" {
		root = os.DirFS(*dir)
	}
	hosts := map[string]string{
		"papers.ifkash.dev":  "papers",
		"courses.ifkash.dev": "courses",
	}
	// Local previews: papers.localhost and courses.localhost resolve to
	// 127.0.0.1 in browsers.
	hosts["papers.localhost"] = "papers"
	hosts["courses.localhost"] = "courses"

	log.Printf("serve %s listening on %s", version, *addr)
	log.Fatal(server.Server(*addr, server.New(root, hosts)).ListenAndServe())
}

// version is set at build time with -ldflags "-X main.version=SHA".
var version = "dev"

func healthcheck(addr string) int {
	host := addr
	if len(host) > 0 && host[0] == ':' {
		host = "127.0.0.1" + host
	}
	client := http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://" + host + "/health")
	if err != nil {
		return 1
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 1
	}
	return 0
}

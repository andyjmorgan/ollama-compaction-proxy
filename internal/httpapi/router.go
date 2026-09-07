// Package httpapi provides routing and per-dialect error envelopes.
package httpapi

import (
	"net/http"
)

// Routes are the handlers the router mounts.
type Routes struct {
	Messages         http.HandlerFunc // POST /v1/messages
	CountTokens      http.HandlerFunc // POST /v1/messages/count_tokens
	Responses        http.HandlerFunc // POST /v1/responses
	ResponsesCompact http.HandlerFunc // POST /v1/responses/compact
	ChatCompletions  http.HandlerFunc // POST /v1/chat/completions
	Health           http.HandlerFunc // GET /health
}

// NewRouter assembles the mux with body-size limits applied.
func NewRouter(r Routes, maxBodyBytes int64) http.Handler {
	mux := http.NewServeMux()

	limit := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, req *http.Request) {
			req.Body = http.MaxBytesReader(w, req.Body, maxBodyBytes)
			next(w, req)
		}
	}

	mux.HandleFunc("POST /v1/messages", limit(r.Messages))
	mux.HandleFunc("POST /v1/messages/count_tokens", limit(r.CountTokens))
	mux.HandleFunc("POST /v1/responses", limit(r.Responses))
	mux.HandleFunc("POST /v1/responses/compact", limit(r.ResponsesCompact))
	mux.HandleFunc("POST /v1/chat/completions", limit(r.ChatCompletions))
	mux.HandleFunc("GET /health", r.Health)

	return mux
}

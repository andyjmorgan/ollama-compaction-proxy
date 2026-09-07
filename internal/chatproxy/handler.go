// Package chatproxy passes /v1/chat/completions through to Ollama untouched,
// adding only the stream-truncation watchdog. No compaction on this surface.
package chatproxy

import (
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// Handler serves /v1/chat/completions.
type Handler struct {
	upstream    *upstream.Client
	modelPrefix string
	log         *slog.Logger
}

// New wires the chat passthrough. modelPrefix, when non-empty, is stripped
// from incoming model names before forwarding (gateway "agent/" routing).
func New(up *upstream.Client, modelPrefix string, log *slog.Logger) *Handler {
	return &Handler{upstream: up, modelPrefix: modelPrefix, log: log}
}

// ChatCompletions handles POST /v1/chat/completions.
func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, "failed to read request body")
		return
	}

	var peek struct {
		Stream bool   `json:"stream"`
		Model  string `json:"model"`
	}
	_ = json.Unmarshal(body, &peek)
	log := h.log.With("dialect", "openai", "path", "/v1/chat/completions", "stream", peek.Stream)

	// Strip a gateway routing prefix ("agent/<model>") before forwarding.
	if h.modelPrefix != "" && strings.HasPrefix(peek.Model, h.modelPrefix) && len(peek.Model) > len(h.modelPrefix) {
		var outer map[string]json.RawMessage
		if err := json.Unmarshal(body, &outer); err == nil {
			bare, _ := json.Marshal(peek.Model[len(h.modelPrefix):])
			outer["model"] = bare
			if rebuilt, err := json.Marshal(outer); err == nil {
				body = rebuilt
			}
		}
	}

	if !peek.Stream {
		resp, err := h.upstream.PostJSON(r.Context(), "/v1/chat/completions", body)
		if err != nil {
			log.Error("upstream call failed", "error", err)
			httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadGateway, "upstream unavailable")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(resp.Body)
		log.Info("request served", "status", resp.StatusCode, "duration_ms", time.Since(start).Milliseconds())
		return
	}

	resp, err := h.upstream.PostStream(r.Context(), "/v1/chat/completions", body)
	if err != nil {
		log.Error("upstream call failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadGateway, "upstream unavailable")
		return
	}
	defer resp.Close()

	if resp.StatusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(resp.ErrorBody)
		return
	}

	fw := upstream.NewFrameWriter(w)
	sawDone := false
	for {
		frame, err := resp.Frames.Next()
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Error("upstream stream read failed", "error", err)
			}
			break
		}
		if err := fw.Write(frame); err != nil {
			return
		}
		if frame.IsDone() {
			sawDone = true
			break
		}
	}

	if !sawDone {
		// Ollama swallows mid-stream errors; surface one so SDKs don't hang.
		data, _ := json.Marshal(map[string]any{
			"error": map[string]any{
				"message": "upstream stream ended unexpectedly",
				"type":    "api_error",
				"param":   nil,
				"code":    nil,
			},
		})
		_ = fw.Write(upstream.Frame{Data: data})
		log.Warn("stream truncated; synthesized error chunk", "stream_truncated", true)
	}
	log.Info("request served", "status", 200, "duration_ms", time.Since(start).Milliseconds())
}

// Package httpapi serves the token-counting endpoint.
package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/extract"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/ollama"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/render"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/tokenizer"
	"github.com/ollama/ollama/anthropic"
)

// maxBodyBytes caps request bodies. Prompts get large, but not this large.
const maxBodyBytes = 64 << 20

// Provider supplies tokenizers for models.
type Provider interface {
	ForModel(ctx context.Context, id string) (*tokenizer.Model, bool, error)
}

// Handler serves /count-tokens and /health.
type Handler struct {
	provider Provider
	log      *slog.Logger
}

// New returns an http.Handler for the service.
func New(provider Provider, log *slog.Logger) http.Handler {
	h := &Handler{provider: provider, log: log}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /count-tokens", h.countTokens)
	mux.HandleFunc("GET /health", h.health)
	return mux
}

type countResponse struct {
	Model string `json:"model"`
	// Tokens is an estimate, not an authoritative inference token count.
	Tokens    int  `json:"tokens"`
	Estimated bool `json:"estimated"`
}

func (h *Handler) countTokens(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		h.log.Warn("read request body failed", "error", err)
		writeError(w, http.StatusBadRequest, "invalid JSON")
		return
	}

	// Use the same conversion as the inference endpoint. Flattening typed
	// Anthropic blocks loses tool names/schemas, call/result structure and thinking.
	if r.Header.Get("X-Tokenizer-Dialect") == "anthropic" {
		var wire anthropic.MessagesRequest
		if err := json.Unmarshal(body, &wire); err != nil {
			writeError(w, http.StatusBadRequest, "invalid Anthropic request")
			return
		}
		chat, err := anthropic.FromMessagesRequest(wire)
		if err != nil {
			writeError(w, http.StatusBadRequest, "unsupported Anthropic request")
			return
		}
		body, err = json.Marshal(chat)
		if err != nil {
			writeError(w, http.StatusBadRequest, "cannot normalize Anthropic request")
			return
		}
	}
	req, err := extract.Parse(body)
	if err != nil {
		// Both are client mistakes; neither is worth logging the body for.
		switch {
		case errors.Is(err, extract.ErrMissingModel):
			writeError(w, http.StatusBadRequest, "model is required")
		default:
			writeError(w, http.StatusBadRequest, "invalid JSON")
		}
		return
	}

	model, cached, err := h.provider.ForModel(r.Context(), req.Model)
	if err != nil {
		if errors.Is(err, ollama.ErrModelNotFound) {
			h.log.Info("model not found", "model", req.Model)
			writeError(w, http.StatusNotFound, "model not found")
			return
		}
		h.log.Error("load tokenizer failed", "model", req.Model, "error", err)
		writeError(w, http.StatusInternalServerError, "failed to load tokenizer")
		return
	}

	rendered := render.Render(req, render.Model{
		Renderer: model.Renderer,
		Template: model.Template,
		Thinking: model.Thinking,
		System:   model.System,
	})

	count, err := model.Count(rendered.Text)
	if err != nil {
		h.log.Error("tokenize failed", "model", req.Model, "error", err)
		writeError(w, http.StatusInternalServerError, "failed to tokenize request")
		return
	}

	// Prompt, message, and tool content are never logged: only shapes, counts,
	// and timings.
	attrs := []any{
		"model", req.Model,
		"tokens", count,
		"tokenizer_cache", hitMiss(cached),
		"algorithm", model.Algorithm,
		"render_tier", rendered.Tier,
		"think", rendered.Think,
		"duration", time.Since(start),
	}
	if rendered.Media > 0 {
		// Media tokens are model-specific and not accounted for; surfacing the
		// count keeps that omission visible rather than silent.
		attrs = append(attrs, "media_blocks", rendered.Media)
	}
	if rendered.Fallback != "" {
		attrs = append(attrs, "render_fallback", rendered.Fallback)
	}
	h.log.Info("counted tokens", attrs...)

	writeJSON(w, http.StatusOK, countResponse{Model: req.Model, Tokens: count, Estimated: true})
}

// health reports process liveness only. It deliberately does not load a
// tokenizer or reach out to Ollama.
func (h *Handler) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func hitMiss(cached bool) string {
	if cached {
		return "hit"
	}
	return "miss"
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

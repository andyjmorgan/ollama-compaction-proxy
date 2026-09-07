package summarize

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// The summarizer must always disable reasoning: on thinking models the
// chain-of-thought is generated at full cost and blows SUMMARIZE_TIMEOUT,
// which silently no-ops compaction (observed with qwen3.x self-summarizing).
func TestSummarizeDisablesReasoning(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &captured); err != nil {
			t.Fatalf("decode summarize request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "condensed"}}},
			"usage":   map[string]int{"prompt_tokens": 10, "completion_tokens": 3},
		})
	}))
	defer server.Close()

	s := New(upstream.New(server.URL, 30*time.Second), "", 30*time.Second)
	res, err := s.Summarize(context.Background(), "qwen3.8-27b-uncensored:q4_k_m", "", "hello world")
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if res.Summary != "condensed" {
		t.Errorf("summary = %q, want %q", res.Summary, "condensed")
	}
	if got := captured["reasoning_effort"]; got != "none" {
		t.Errorf("reasoning_effort = %v, want %q", got, "none")
	}
}

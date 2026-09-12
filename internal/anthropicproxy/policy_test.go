package anthropicproxy

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/config"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/tokencount"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

func TestNativeClaudeBudget(t *testing.T) {
	count := 109999
	calls := 0
	failCount := false
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/count-tokens" {
			if r.Header.Get("X-Tokenizer-Dialect") != "anthropic" {
				t.Error("missing Anthropic tokenizer dialect")
			}
			if failCount {
				w.WriteHeader(503)
				io.WriteString(w, `{"error":"offline"}`)
				return
			}
			json.NewEncoder(w).Encode(map[string]int{"tokens": count})
			return
		}
		calls++
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["max_tokens"].(float64) > 8192 {
			t.Error("output reserve not enforced")
		}
		io.WriteString(w, `{"id":"m","content":[],"usage":{"input_tokens":1,"output_tokens":0}}`)
	}))
	defer fake.Close()
	cfg := &config.Config{ModelPrefix: "agent/", ClaudeModels: map[string]config.ClaudeModel{
		"qwen":  {ContextWindow: 262144, MaxOutputTokens: 8192, CompactAt: 230000},
		"gemma": {ContextWindow: 131072, MaxOutputTokens: 8192, CompactAt: 110000},
	}}
	h := New(cfg, upstream.New(fake.URL, time.Second), tokencount.New(fake.URL), nil, slog.New(slog.DiscardHandler))
	run := func(model, content string, enabled bool) *httptest.ResponseRecorder {
		b, _ := json.Marshal(map[string]any{"model": model, "max_tokens": 10000, "messages": []map[string]string{{"role": "user", "content": content}}})
		r := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(string(b)))
		if enabled {
			r.Header.Set("X-Claude-Compaction", "native")
		}
		w := httptest.NewRecorder()
		h.Messages(w, r)
		return w
	}
	if w := run("agent/gemma", "hello", true); w.Code != 200 {
		t.Fatalf("below limit: %d %s", w.Code, w.Body)
	}
	count = 110000
	if w := run("agent/gemma", "hello", true); w.Code != 400 || !strings.Contains(w.Body.String(), "prompt is too long") {
		t.Fatalf("at limit: %d %s", w.Code, w.Body)
	}
	if calls != 1 {
		t.Fatal("over-budget request reached inference")
	}
	if w := run("agent/qwen", "hello", true); w.Code != 200 {
		t.Fatalf("parent limit inherited by child or vice versa: %d", w.Code)
	}
	summary := "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\nYour task is to create a detailed summary of the conversation so far"
	if w := run("agent/gemma", summary, true); w.Code != 200 {
		t.Fatalf("summary denied: %d", w.Code)
	}
	count = 131072 - 8192
	if w := run("agent/gemma", summary, true); w.Code != 400 {
		t.Fatal("summary bypassed physical window")
	}
	if w := run("agent/missing", "hello", true); w.Code != 400 {
		t.Fatal("unknown policy accepted")
	}
	failCount = true
	if w := run("agent/gemma", "hello", true); w.Code != 503 {
		t.Fatal("count failure allowed unchecked inference")
	}
}

func TestClaudeSummaryOnlyTerminalUserInstruction(t *testing.T) {
	marker := "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.\nYour task is to create a detailed summary of the conversation so far"
	for _, tc := range []struct {
		role string
		tail bool
		want bool
	}{{"user", false, true}, {"assistant", false, false}, {"user", true, false}} {
		ms := []map[string]any{{"role": tc.role, "content": []map[string]string{{"type": "text", "text": marker}}}}
		if tc.tail {
			ms = append(ms, map[string]any{"role": "user", "content": "Continue the task"})
		}
		b, _ := json.Marshal(map[string]any{"messages": ms})
		if got := claudeSummary(b); got != tc.want {
			t.Errorf("%+v got %v", tc, got)
		}
	}
}

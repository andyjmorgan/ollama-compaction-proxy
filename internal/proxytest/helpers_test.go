// Package proxytest exercises the assembled proxy against a scripted fake
// Ollama + tokenizer, per dialect.
package proxytest

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/anthropicproxy"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/chatproxy"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/config"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/openaiproxy"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/testsupport/fakeollama"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/tokencount"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

var testKey = []byte("test-hmac-key-0123456789abcdef00")

// newProxy assembles the full proxy wired to a fresh fake upstream.
func newProxy(t *testing.T) (http.Handler, *fakeollama.Fake) {
	t.Helper()
	fake := fakeollama.New(t)

	cfg := &config.Config{
		OllamaURL:               fake.URL,
		TokenizerURL:            fake.URL,
		HMACKey:                 testKey,
		DefaultTriggerAnthropic: 150000,
		DefaultThresholdOpenAI:  200000,
		MinTrigger:              10,
		ModelPrefix:             "agent/",
		MaxBodyBytes:            8 << 20,
		SummarizeTimeout:        30 * time.Second,
		UpstreamTimeout:         30 * time.Second,
	}

	up := upstream.New(cfg.OllamaURL, cfg.UpstreamTimeout)
	counter := tokencount.New(cfg.TokenizerURL)
	summarizer := summarize.New(up, cfg.CompactModel, cfg.SummarizeTimeout)
	log := slog.New(slog.DiscardHandler)

	anthropic := anthropicproxy.New(cfg, up, counter, summarizer, log)
	openai := openaiproxy.New(cfg, up, counter, summarizer, log)
	chat := chatproxy.New(up, cfg.ModelPrefix, log)

	return httpapi.NewRouter(httpapi.Routes{
		Messages:         anthropic.Messages,
		CountTokens:      anthropic.CountTokens,
		Responses:        openai.Responses,
		ResponsesCompact: openai.Compact,
		ChatCompletions:  chat.ChatCompletions,
		Health:           func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{}`)) },
	}, cfg.MaxBodyBytes), fake
}

// post drives one request through the proxy.
func post(t *testing.T, h http.Handler, path, body string) (int, []byte) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	data, _ := io.ReadAll(rec.Body)
	return rec.Code, data
}

// mustJSON parses JSON or fails the test.
func mustJSON(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("response is not JSON: %v\n%s", err, data)
	}
	return m
}

// sseFrames splits an SSE body into (event, data) pairs.
type sseFrame struct {
	event string
	data  string
}

// fakeollamaFrames is sugar for scripting SSE fixtures.
type fakeollamaFrames struct {
	event string
	data  string
}

type frameList []fakeollamaFrames

func (l frameList) frames() []fakeollama.SSEFrame {
	out := make([]fakeollama.SSEFrame, len(l))
	for i, f := range l {
		out[i] = fakeollama.SSEFrame{Event: f.event, Data: f.data}
	}
	return out
}

func parseSSE(t *testing.T, body []byte) []sseFrame {
	t.Helper()
	var frames []sseFrame
	for _, chunk := range strings.Split(string(body), "\n\n") {
		var f sseFrame
		seen := false
		for _, line := range strings.Split(chunk, "\n") {
			if v, ok := strings.CutPrefix(line, "event: "); ok {
				f.event = v
				seen = true
			}
			if v, ok := strings.CutPrefix(line, "data: "); ok {
				f.data = v
				seen = true
			}
		}
		if seen {
			frames = append(frames, f)
		}
	}
	return frames
}

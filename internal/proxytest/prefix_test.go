package proxytest

import (
	"encoding/json"
	"strings"
	"testing"
)

// A gateway routes "agent/<model>" here by prefix; the proxy must resolve the
// bare model on every endpoint so Ollama never sees the routing prefix.
func TestModelPrefixStripped(t *testing.T) {
	t.Run("messages", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/messages",
			`{"model":"agent/m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		var fwd struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(fake.MessagesReqs[0], &fwd)
		if fwd.Model != "m" {
			t.Errorf("forwarded model = %q, want bare %q", fwd.Model, "m")
		}
	})

	t.Run("messages compaction uses bare model", func(t *testing.T) {
		h, fake := newProxy(t)
		fake.TokenCount = 100
		body := strings.Replace(compactEditMessages, `"model":"m"`, `"model":"agent/m"`, 1)
		status, resp := post(t, h, "/v1/messages", body)
		if status != 200 {
			t.Fatalf("status = %d: %s", status, resp)
		}
		if !strings.Contains(string(resp), `"compaction"`) {
			t.Error("compaction did not run for prefixed model")
		}
		// Summarizer call must use the bare model too.
		var sum struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(fake.ChatReqs[0], &sum)
		if sum.Model != "m" {
			t.Errorf("summarizer model = %q, want bare %q", sum.Model, "m")
		}
	})

	t.Run("count_tokens", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/messages/count_tokens",
			`{"model":"agent/m","messages":[{"role":"user","content":"hi"}]}`)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		var counted struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(fake.CountReqs[0], &counted)
		if counted.Model != "m" {
			t.Errorf("counted model = %q, want bare %q", counted.Model, "m")
		}
	})

	t.Run("responses", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/responses",
			`{"model":"agent/m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		var fwd struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(fake.ResponsesReqs[0], &fwd)
		if fwd.Model != "m" {
			t.Errorf("forwarded model = %q, want bare %q", fwd.Model, "m")
		}
	})

	t.Run("responses string input", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/responses", `{"model":"agent/m","input":"hi"}`)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		var fwd struct {
			Model string `json:"model"`
			Input string `json:"input"`
		}
		_ = json.Unmarshal(fake.ResponsesReqs[0], &fwd)
		if fwd.Model != "m" || fwd.Input != "hi" {
			t.Errorf("forwarded = %+v, want bare model with input intact", fwd)
		}
	})

	t.Run("responses compact", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/responses/compact",
			`{"model":"agent/m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		var sum struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(fake.ChatReqs[0], &sum)
		if sum.Model != "m" {
			t.Errorf("summarizer model = %q, want bare %q", sum.Model, "m")
		}
	})

	t.Run("chat completions", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/chat/completions",
			`{"model":"agent/m","messages":[{"role":"user","content":"hi"}]}`)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		var fwd struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(fake.ChatReqs[0], &fwd)
		if fwd.Model != "m" {
			t.Errorf("forwarded model = %q, want bare %q", fwd.Model, "m")
		}
	})

	t.Run("unprefixed models untouched", func(t *testing.T) {
		h, fake := newProxy(t)
		status, _ := post(t, h, "/v1/messages", plainMessages)
		if status != 200 {
			t.Fatalf("status = %d", status)
		}
		if got := string(fake.MessagesReqs[0]); got != plainMessages {
			t.Errorf("unprefixed request was rewritten: %s", got)
		}
	})
}

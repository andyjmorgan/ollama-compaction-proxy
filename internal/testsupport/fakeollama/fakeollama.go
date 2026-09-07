// Package fakeollama is a scripted stand-in for Ollama's compat endpoints and
// the tokenizer service, so the proxy's tests run hermetically.
package fakeollama

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// SSEFrame scripts one server-sent event.
type SSEFrame struct {
	Event string
	Data  string
}

// Fake is a scripted upstream. Configure the exported fields, then read the
// captured requests after driving the proxy.
type Fake struct {
	URL string

	mu sync.Mutex

	// Captured request bodies per endpoint, in arrival order.
	MessagesReqs  [][]byte
	ResponsesReqs [][]byte
	ChatReqs      [][]byte
	CountReqs     [][]byte

	// MessagesJSON / ResponsesJSON are returned for non-streaming calls.
	MessagesJSON  string
	ResponsesJSON string

	// MessagesFrames / ResponsesFrames are emitted for streaming calls.
	MessagesFrames  []SSEFrame
	ResponsesFrames []SSEFrame

	// TruncateStream drops all frames after emitting the first N (simulating
	// Ollama's swallowed mid-stream errors). 0 means emit everything.
	TruncateStream int

	// SummaryText is returned by /v1/chat/completions (the summarizer path)
	// along with SummaryUsage.
	SummaryText             string
	SummaryPromptTokens     int
	SummaryCompletionTokens int
	SummarizeFails          bool

	// TokenCount is returned by the fake tokenizer; TokenizerFails simulates
	// an outage.
	TokenCount     int
	TokenizerFails bool
}

// New starts a Fake and registers cleanup.
func New(t *testing.T) *Fake {
	t.Helper()
	f := &Fake{
		MessagesJSON: `{"id":"msg_up1","type":"message","role":"assistant","model":"m",` +
			`"content":[{"type":"text","text":"upstream says hi"}],"stop_reason":"end_turn",` +
			`"usage":{"input_tokens":10,"output_tokens":5}}`,
		ResponsesJSON: `{"id":"resp_up1","object":"response","created_at":1,"status":"completed","model":"m",` +
			`"output":[{"id":"msg_up1","type":"message","status":"completed","role":"assistant",` +
			`"content":[{"type":"output_text","text":"upstream says hi"}]}],` +
			`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`,
		SummaryText:             "the conversation so far, condensed",
		SummaryPromptTokens:     100,
		SummaryCompletionTokens: 20,
		TokenCount:              10,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", func(w http.ResponseWriter, r *http.Request) {
		body := readBody(r)
		f.mu.Lock()
		f.MessagesReqs = append(f.MessagesReqs, body)
		f.mu.Unlock()
		f.respond(w, body, f.MessagesJSON, f.MessagesFrames)
	})
	mux.HandleFunc("POST /v1/responses", func(w http.ResponseWriter, r *http.Request) {
		body := readBody(r)

		// Mirror Ollama's strictness: unknown input item types are a 400.
		if msg, bad := unknownItemType(body); bad {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprintf(w, `{"error":{"message":%q,"type":"invalid_request_error","param":null,"code":null}}`, msg)
			return
		}

		f.mu.Lock()
		f.ResponsesReqs = append(f.ResponsesReqs, body)
		f.mu.Unlock()
		f.respond(w, body, f.ResponsesJSON, f.ResponsesFrames)
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		body := readBody(r)
		f.mu.Lock()
		f.ChatReqs = append(f.ChatReqs, body)
		f.mu.Unlock()

		if f.SummarizeFails {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":{"message":"summarizer down","type":"api_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": f.SummaryText}}},
			"usage": map[string]int{
				"prompt_tokens":     f.SummaryPromptTokens,
				"completion_tokens": f.SummaryCompletionTokens,
			},
		})
	})
	mux.HandleFunc("POST /count-tokens", func(w http.ResponseWriter, r *http.Request) {
		body := readBody(r)
		f.mu.Lock()
		f.CountReqs = append(f.CountReqs, body)
		f.mu.Unlock()

		if f.TokenizerFails {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, `{"error":"tokenizer down"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"model":"m","tokens":%d,"estimated":true}`, f.TokenCount)
	})
	mux.HandleFunc("GET /api/version", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"version":"0.32.8-fake"}`)
	})
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `{"status":"ok"}`)
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	f.URL = srv.URL
	return f
}

// respond serves either the scripted JSON or the scripted SSE frames,
// depending on the request's stream flag.
func (f *Fake) respond(w http.ResponseWriter, reqBody []byte, jsonBody string, frames []SSEFrame) {
	var peek struct {
		Stream bool `json:"stream"`
	}
	_ = json.Unmarshal(reqBody, &peek)

	if !peek.Stream {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, jsonBody)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	flusher, _ := w.(http.Flusher)
	for i, fr := range frames {
		if f.TruncateStream > 0 && i >= f.TruncateStream {
			return // connection closes with no terminal event
		}
		if fr.Event != "" {
			fmt.Fprintf(w, "event: %s\n", fr.Event)
		}
		fmt.Fprintf(w, "data: %s\n\n", fr.Data)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// unknownItemType mirrors Ollama's input-item strictness on /v1/responses.
func unknownItemType(body []byte) (string, bool) {
	var req struct {
		Input json.RawMessage `json:"input"`
	}
	if json.Unmarshal(body, &req) != nil || len(req.Input) == 0 {
		return "", false
	}
	var items []json.RawMessage
	if json.Unmarshal(req.Input, &items) != nil {
		return "", false // string input
	}
	for i, raw := range items {
		var peek struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		_ = json.Unmarshal(raw, &peek)
		switch peek.Type {
		case "message", "function_call", "function_call_output", "reasoning":
		case "":
			if peek.Role == "" {
				return fmt.Sprintf("input[%d]: missing type", i), true
			}
		default:
			return fmt.Sprintf("input[%d]: unknown input item type: %q", i, peek.Type), true
		}
	}
	return "", false
}

func readBody(r *http.Request) []byte {
	defer r.Body.Close()
	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 4096)
	for {
		n, err := r.Body.Read(tmp)
		buf = append(buf, tmp[:n]...)
		if err != nil {
			return buf
		}
	}
}

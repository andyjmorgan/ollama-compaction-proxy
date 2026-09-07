//go:build parity

// Package parity characterizes drift between this service's estimate and the
// prompt token count Ollama actually reports.
//
// This is a validation harness, not part of the API contract. It never fails on
// non-zero drift: its job is to tell us how large the drift is, per model and
// per request shape, so a tolerance can be chosen from evidence.
//
// It needs a real Ollama with the models installed, and it loads them, so it is
// behind a build tag:
//
//	OLLAMA_URL=http://192.168.69.28:11434 go test -tags parity ./internal/parity/ -v
package parity

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strings"
	"testing"
	"text/tabwriter"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/ollama"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/tokenizer"
)

// defaultModels are the models this service was built for, plus a template-based
// model to cover the non-renderer path. Override with PARITY_MODELS.
var defaultModels = []string{
	"gemma4:e4b",
	"muse-glimmer:latest",
	"qwen3:4b-instruct-2507-q4_K_M",
}

// shape is one representative request, per spec section 18.
type shape struct {
	name     string
	messages []map[string]any
	tools    []map[string]any
}

var shapes = []shape{
	{
		name:     "user",
		messages: []map[string]any{{"role": "user", "content": "Explain Kubernetes."}},
	},
	{
		name: "system+user",
		messages: []map[string]any{
			{"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
			{"role": "user", "content": "Explain Kubernetes."},
		},
	},
	{
		name: "multi-turn",
		messages: []map[string]any{
			{"role": "system", "content": "You are a terse assistant."},
			{"role": "user", "content": "Explain Kubernetes."},
			{"role": "assistant", "content": "It orchestrates containers across a cluster."},
			{"role": "user", "content": "How does scheduling work?"},
			{"role": "assistant", "content": "The scheduler binds pods to nodes by fit and priority."},
			{"role": "user", "content": "And rescheduling?"},
		},
	},
	{
		name:     "tools",
		messages: []map[string]any{{"role": "user", "content": "What is the weather in Dublin?"}},
		tools: []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "get_weather",
				"description": "Get the current weather for a location",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string", "description": "City and country"},
						"unit":     map[string]any{"type": "string", "enum": []string{"celsius", "fahrenheit"}},
					},
					"required": []string{"location"},
				},
			},
		}},
	},
	{
		name: "tool call+result",
		messages: []map[string]any{
			{"role": "user", "content": "What is the weather in Dublin?"},
			{"role": "assistant", "tool_calls": []map[string]any{{
				"function": map[string]any{"name": "get_weather", "arguments": map[string]any{"location": "Dublin"}},
			}}},
			{"role": "tool", "tool_name": "get_weather", "content": "12C, raining"},
			{"role": "user", "content": "Should I bring an umbrella?"},
		},
		tools: []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "get_weather",
				"description": "Get the current weather for a location",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"location": map[string]any{"type": "string"}},
					"required":   []string{"location"},
				},
			},
		}},
	},
}

type row struct {
	model     string
	shape     string
	estimated int
	actual    int
}

func (r row) diff() int { return r.estimated - r.actual }

func (r row) diffPct() float64 {
	if r.actual == 0 {
		return 0
	}
	return float64(r.diff()) / float64(r.actual) * 100
}

func TestDriftAgainstOllama(t *testing.T) {
	baseURL := os.Getenv("OLLAMA_URL")
	if baseURL == "" {
		t.Skip("set OLLAMA_URL to run the parity harness")
	}

	models := defaultModels
	if env := os.Getenv("PARITY_MODELS"); env != "" {
		models = strings.Split(env, ",")
	}

	// Loading a model can be slow on a cold cache.
	hc := &http.Client{Timeout: 10 * time.Minute}
	client := ollama.New(baseURL, hc)
	svc := httpapi.New(tokenizer.NewProvider(client), slog.New(slog.DiscardHandler))

	var rows []row
	for _, model := range models {
		model = strings.TrimSpace(model)
		t.Run(model, func(t *testing.T) {
			for _, s := range shapes {
				t.Run(s.name, func(t *testing.T) {
					body := map[string]any{"model": model, "messages": s.messages}
					if s.tools != nil {
						body["tools"] = s.tools
					}

					estimated, err := estimate(svc, body)
					if err != nil {
						t.Fatalf("estimate: %v", err)
					}
					actual, err := promptEvalCount(t.Context(), hc, baseURL, body)
					if err != nil {
						t.Fatalf("measure: %v", err)
					}

					r := row{model: model, shape: s.name, estimated: estimated, actual: actual}
					rows = append(rows, r)
					t.Logf("estimated %d, actual %d, diff %+d (%+.1f%%)", r.estimated, r.actual, r.diff(), r.diffPct())
				})
			}
		})
	}

	report(t, rows)
}

// estimate calls the service in process, exactly as a consumer would.
func estimate(svc http.Handler, body map[string]any) (int, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return 0, err
	}

	rec := httptest.NewRecorder()
	svc.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/count-tokens", bytes.NewReader(raw)))
	if rec.Code != http.StatusOK {
		return 0, fmt.Errorf("service returned %d: %s", rec.Code, strings.TrimSpace(rec.Body.String()))
	}

	var resp struct {
		Tokens int `json:"tokens"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		return 0, err
	}
	return resp.Tokens, nil
}

// promptEvalCount asks Ollama how many prompt tokens the request really costs.
//
// num_predict is 0, so this evaluates the prompt without generating. This is a
// measurement for the harness only; the service itself must never obtain a
// count this way.
func promptEvalCount(ctx context.Context, hc *http.Client, baseURL string, body map[string]any) (int, error) {
	req := make(map[string]any, len(body)+2)
	for k, v := range body {
		req[k] = v
	}
	req["stream"] = false
	req["options"] = map[string]any{"num_predict": 0}

	raw, err := json.Marshal(req)
	if err != nil {
		return 0, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(baseURL, "/")+"/api/chat", bytes.NewReader(raw))
	if err != nil {
		return 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := hc.Do(httpReq)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var decoded struct {
		PromptEvalCount int    `json:"prompt_eval_count"`
		Error           string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return 0, err
	}
	if decoded.Error != "" {
		return 0, fmt.Errorf("ollama: %s", decoded.Error)
	}
	if decoded.PromptEvalCount == 0 {
		return 0, fmt.Errorf("ollama reported no prompt_eval_count")
	}
	return decoded.PromptEvalCount, nil
}

func report(t *testing.T, rows []row) {
	t.Helper()
	if len(rows) == 0 {
		return
	}

	var b strings.Builder
	b.WriteString("\ndrift against ollama prompt_eval_count\n\n")
	w := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "MODEL\tSHAPE\tESTIMATED\tACTUAL\tDIFF\tDIFF%")
	for _, r := range rows {
		fmt.Fprintf(w, "%s\t%s\t%d\t%d\t%+d\t%+.1f%%\n", r.model, r.shape, r.estimated, r.actual, r.diff(), r.diffPct())
	}

	// Worst absolute drift per model is what a consumer would size a margin on.
	worst := map[string]float64{}
	for _, r := range rows {
		if abs(r.diffPct()) > abs(worst[r.model]) {
			worst[r.model] = r.diffPct()
		}
	}
	models := make([]string, 0, len(worst))
	for m := range worst {
		models = append(models, m)
	}
	sort.Strings(models)

	fmt.Fprintln(w, "\tSUMMARY\t\t\t\t")
	for _, m := range models {
		fmt.Fprintf(w, "%s\tworst drift\t\t\t\t%+.1f%%\n", m, worst[m])
	}
	w.Flush()

	t.Log(b.String())
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

package httpapi_test

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/ollama"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/testsupport"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/tokenizer"
)

const testModel = "fake-model:1b"

// newService wires the real handler, provider, and client against a fake
// Ollama, so these tests exercise the whole request path.
func newService(t *testing.T, models map[string]testsupport.Model) (http.Handler, *testsupport.Ollama) {
	t.Helper()

	fake := testsupport.NewOllama(t, models)
	client := ollama.New(fake.URL, nil)
	log := slog.New(slog.DiscardHandler)
	return httpapi.New(tokenizer.NewProvider(client), log), fake
}

func defaultService(t *testing.T) (http.Handler, *testsupport.Ollama) {
	t.Helper()
	return newService(t, map[string]testsupport.Model{
		testModel: {Template: testsupport.ChatTemplate},
	})
}

// count posts body and returns the status and decoded token count.
func count(t *testing.T, h http.Handler, body string) (int, countBody) {
	t.Helper()

	req := httptest.NewRequest(http.MethodPost, "/count-tokens", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	var decoded countBody
	raw, _ := io.ReadAll(rec.Body)
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("response is not JSON: %v (body %q)", err, raw)
	}
	return rec.Code, decoded
}

type countBody struct {
	Model     string `json:"model"`
	Tokens    int    `json:"tokens"`
	Estimated bool   `json:"estimated"`
	Error     string `json:"error"`
}

// mustCount requires a 200 and returns the token count.
func mustCount(t *testing.T, h http.Handler, body string) int {
	t.Helper()

	status, got := count(t, h, body)
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200 (error %q) for body %s", status, got.Error, body)
	}
	if !got.Estimated {
		t.Error("estimated = false, want true: counts must never claim to be authoritative")
	}
	if got.Model != testModel {
		t.Errorf("model = %q, want %q", got.Model, testModel)
	}
	return got.Tokens
}

func TestMinimalRequest(t *testing.T) {
	h, _ := defaultService(t)

	if got := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello"}`); got <= 0 {
		t.Errorf("tokens = %d, want > 0", got)
	}
}

// Unknown fields must be ignored rather than rejected, so a new field in some
// provider's API never breaks this service.
func TestUnknownFieldsIgnored(t *testing.T) {
	h, _ := defaultService(t)

	base := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello"}`)
	withUnknown := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello",`+
		`"banana":"this service has never heard of this field"}`)

	if withUnknown != base {
		t.Errorf("unknown field changed count: %d, want %d", withUnknown, base)
	}
}

// Generation and sampling options influence inference but are not prompt text.
func TestGenerationFieldsNotCounted(t *testing.T) {
	h, _ := defaultService(t)

	base := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello"}`)
	withOptions := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello",`+
		`"temperature":0.7,"top_p":0.9,"stream":true,"max_tokens":4096,`+
		`"max_output_tokens":4096,"metadata":{"user":"someone"}}`)

	if withOptions != base {
		t.Errorf("generation fields changed count: %d, want %d", withOptions, base)
	}
}

func TestPromptBearingFieldsIncreaseCount(t *testing.T) {
	h, _ := defaultService(t)

	base := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"Explain Kubernetes."}]}`)

	cases := []struct {
		name string
		body string
	}{
		{"system", `{"model":"` + testModel + `","system":"You are a useful assistant.",` +
			`"messages":[{"role":"user","content":"Explain Kubernetes."}]}`},
		{"instructions", `{"model":"` + testModel + `","instructions":"Answer only in haiku.",` +
			`"messages":[{"role":"user","content":"Explain Kubernetes."}]}`},
		{"tools", `{"model":"` + testModel + `","messages":[{"role":"user","content":"Explain Kubernetes."}],` +
			`"tools":[{"type":"function","function":{"name":"get_weather",` +
			`"description":"Get the current weather for a location",` +
			`"parameters":{"type":"object","properties":{"location":{"type":"string",` +
			`"description":"City and country"}},"required":["location"]}}}]}`},
		{"extra message", `{"model":"` + testModel + `","messages":[` +
			`{"role":"user","content":"Explain Kubernetes."},` +
			`{"role":"assistant","content":"It orchestrates containers."},` +
			`{"role":"user","content":"Go on."}]}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := mustCount(t, h, tc.body); got <= base {
				t.Errorf("tokens = %d, want > %d (base)", got, base)
			}
		})
	}
}

// A template that never references .Tools does not send them to the model, so
// they cost nothing. This is deliberate: the count reflects what the model
// actually receives, not what the caller supplied.
func TestToolsIgnoredByToollessTemplate(t *testing.T) {
	h, _ := newService(t, map[string]testsupport.Model{
		testModel: {Template: testsupport.NoToolsTemplate},
	})

	base := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}]}`)
	withTools := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}],`+
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather"}}]}`)

	if withTools != base {
		t.Errorf("tools changed count to %d (base %d), but this template never renders them", withTools, base)
	}
}

// Message roles must reach the tokenizer, not just message content.
func TestMessageRoleContributes(t *testing.T) {
	h, _ := defaultService(t)

	short := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}]}`)
	long := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"assistant","content":"hi"}]}`)

	if short == long {
		t.Errorf("role did not affect count: both %d", short)
	}
}

// Anthropic- and OpenAI-Responses-shaped requests send content as typed blocks
// rather than a string; that text must still be counted.
func TestBlockStructuredContent(t *testing.T) {
	h, _ := defaultService(t)

	plain := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"Explain Kubernetes."}]}`)
	blocks := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":[`+
		`{"type":"text","text":"Explain Kubernetes."}]}]}`)

	if blocks != plain {
		t.Errorf("block content = %d tokens, want %d (same text)", blocks, plain)
	}
}

// Media has a model-specific cost this service does not estimate, but its
// presence must not fail the request.
func TestMultimodalContentDoesNotFail(t *testing.T) {
	h, _ := defaultService(t)

	got := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":[`+
		`{"type":"text","text":"What is in this image?"},`+
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}]}]}`)
	if got <= 0 {
		t.Errorf("tokens = %d, want > 0", got)
	}
}

func TestLongerTextCountsMore(t *testing.T) {
	h, _ := defaultService(t)

	short := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello"}`)
	long := mustCount(t, h, `{"model":"`+testModel+`","prompt":"`+strings.Repeat("hello world ", 50)+`"}`)

	if long <= short {
		t.Errorf("long text = %d tokens, want > %d", long, short)
	}
}

func TestErrors(t *testing.T) {
	h, _ := defaultService(t)

	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantError  string
	}{
		{"missing model", `{"prompt":"hello"}`, http.StatusBadRequest, "model is required"},
		{"empty model", `{"model":"","prompt":"hello"}`, http.StatusBadRequest, "model is required"},
		{"non-string model", `{"model":42,"prompt":"hello"}`, http.StatusBadRequest, "model is required"},
		{"invalid JSON", `{"model":"x",`, http.StatusBadRequest, "invalid JSON"},
		{"not an object", `["nope"]`, http.StatusBadRequest, "invalid JSON"},
		{"unknown model", `{"model":"nope:1b","prompt":"hello"}`, http.StatusNotFound, "model not found"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, got := count(t, h, tc.body)
			if status != tc.wantStatus {
				t.Errorf("status = %d, want %d", status, tc.wantStatus)
			}
			if got.Error != tc.wantError {
				t.Errorf("error = %q, want %q", got.Error, tc.wantError)
			}
		})
	}
}

// A model with no prompt-bearing fields is valid: it counts whatever the
// template emits on its own.
func TestRequestWithoutPromptContent(t *testing.T) {
	h, _ := defaultService(t)

	status, got := count(t, h, `{"model":"`+testModel+`","temperature":0.5}`)
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200 (error %q)", status, got.Error)
	}
	if got.Tokens < 0 {
		t.Errorf("tokens = %d, want >= 0", got.Tokens)
	}
}

func TestTokenizerCachedAcrossRequests(t *testing.T) {
	h, fake := defaultService(t)

	for range 3 {
		mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello"}`)
	}

	if got := fake.Shows.Load(); got != 1 {
		t.Errorf("/api/show called %d times, want 1: tokenizer should be cached", got)
	}
}

// Concurrent first requests for one model must be safe and must not each
// fetch a vocabulary. Run with -race.
func TestConcurrentFirstRequestsShareOneLoad(t *testing.T) {
	h, fake := defaultService(t)

	const n = 20
	var wg sync.WaitGroup
	wg.Add(n)
	for range n {
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodPost, "/count-tokens",
				strings.NewReader(`{"model":"`+testModel+`","prompt":"hello"}`))
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Errorf("status = %d, want 200", rec.Code)
			}
		}()
	}
	wg.Wait()

	if got := fake.Shows.Load(); got != 1 {
		t.Errorf("/api/show called %d times, want 1: concurrent loads should coalesce", got)
	}
}

// A failed load must not be cached, so a model pulled later starts working.
func TestFailedLoadNotCached(t *testing.T) {
	h, fake := defaultService(t)

	for range 2 {
		if status, _ := count(t, h, `{"model":"nope:1b","prompt":"hello"}`); status != http.StatusNotFound {
			t.Fatalf("status = %d, want 404", status)
		}
	}
	if got := fake.Shows.Load(); got != 0 {
		t.Errorf("successful shows = %d, want 0", got)
	}
}

// Renderer-based models pair a real renderer with a stub template. The
// renderer must be used, not the stub.
func TestRendererTierUsedOverStubTemplate(t *testing.T) {
	h, _ := newService(t, map[string]testsupport.Model{
		testModel: {Renderer: "gemma4", Template: testsupport.StubTemplate},
	})

	got := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}]}`)

	// The stub template would emit only the bare prompt. Gemma 4's renderer
	// wraps it in turn markers, so the count must exceed the raw text.
	if raw := testsupport.CountBytes("hi"); got <= raw {
		t.Errorf("tokens = %d, want > %d: renderer markup appears to be missing", got, raw)
	}
}

// An unknown renderer must degrade to the template rather than fail.
func TestUnknownRendererFallsBack(t *testing.T) {
	h, _ := newService(t, map[string]testsupport.Model{
		testModel: {Renderer: "no-such-renderer", Template: testsupport.ChatTemplate},
	})

	if got := mustCount(t, h, `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}]}`); got <= 0 {
		t.Errorf("tokens = %d, want > 0", got)
	}
}

// A model with neither renderer nor template still counts its text.
func TestNoRendererNoTemplate(t *testing.T) {
	h, _ := newService(t, map[string]testsupport.Model{
		testModel: {},
	})

	if got := mustCount(t, h, `{"model":"`+testModel+`","prompt":"hello"}`); got <= 0 {
		t.Errorf("tokens = %d, want > 0", got)
	}
}

func TestHealth(t *testing.T) {
	h, fake := defaultService(t)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	if got := rec.Body.String(); !strings.Contains(got, `"status":"ok"`) {
		t.Errorf("body = %q, want status ok", got)
	}
	// Health must not depend on loading tokenizers.
	if got := fake.Shows.Load(); got != 0 {
		t.Errorf("/api/show called %d times, want 0", got)
	}
}

// Ollama prepends the model's own system prompt when a request has none, and
// it is substantial for models like qwen2.5-coder. Ignoring it undercounts
// every request that omits a system message.
func TestModelDefaultSystemPromptCounted(t *testing.T) {
	const defaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

	plain, _ := newService(t, map[string]testsupport.Model{
		testModel: {Template: testsupport.ChatTemplate},
	})
	withDefault, _ := newService(t, map[string]testsupport.Model{
		testModel: {Template: testsupport.ChatTemplate, System: defaultSystem},
	})

	body := `{"model":"` + testModel + `","messages":[{"role":"user","content":"hi"}]}`
	base := mustCount(t, plain, body)
	got := mustCount(t, withDefault, body)

	if want := base + testsupport.CountBytes(defaultSystem); got < want {
		t.Errorf("tokens = %d, want at least %d: the default system prompt is missing", got, want)
	}
}

// A request's own system message replaces the model default rather than
// stacking on top of it.
func TestRequestSystemReplacesModelDefault(t *testing.T) {
	h, _ := newService(t, map[string]testsupport.Model{
		testModel: {Template: testsupport.ChatTemplate, System: strings.Repeat("default system. ", 20)},
	})

	got := mustCount(t, h, `{"model":"`+testModel+`","system":"be brief",`+
		`"messages":[{"role":"user","content":"hi"}]}`)

	if floor := testsupport.CountBytes(strings.Repeat("default system. ", 20)); got >= floor {
		t.Errorf("tokens = %d, want well under %d: the model default should have been replaced", got, floor)
	}
}

// Ollama turns thinking on by default for thinking-capable models, which adds
// a block to the rendered prompt.
func TestThinkingCapabilityChangesCount(t *testing.T) {
	plain, _ := newService(t, map[string]testsupport.Model{
		testModel: {Renderer: "gemma4", Template: testsupport.StubTemplate},
	})
	thinking, _ := newService(t, map[string]testsupport.Model{
		testModel: {Renderer: "gemma4", Template: testsupport.StubTemplate, Capabilities: []string{"completion", "thinking"}},
	})

	body := `{"model":"` + testModel + `","messages":[{"role":"user","content":"hi"}]}`
	base := mustCount(t, plain, body)
	got := mustCount(t, thinking, body)

	if got <= base {
		t.Errorf("thinking-capable count = %d, want > %d", got, base)
	}

	// An explicit think:false must override the capability default.
	off := mustCount(t, thinking, `{"model":"`+testModel+`","messages":[{"role":"user","content":"hi"}],"think":false}`)
	if off != base {
		t.Errorf("think:false count = %d, want %d (same as a non-thinking model)", off, base)
	}
}

// Equivalent native and Anthropic tool-bearing requests must render identically.
// In particular, an Anthropic tool schema must not become an empty native tool.
func TestAnthropicDialectPreservesTools(t *testing.T) {
	h, _ := defaultService(t)
	native := `{"model":"fake-model:1b","messages":[{"role":"user","content":"weather"}],"tools":[{"type":"function","function":{"name":"weather","description":"look up a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}`
	wire := `{"model":"fake-model:1b","messages":[{"role":"user","content":[{"type":"text","text":"weather"}]}],"tools":[{"name":"weather","description":"look up a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}`
	want := mustCount(t, h, native)
	req := httptest.NewRequest("POST", "/count-tokens", strings.NewReader(wire))
	req.Header.Set("X-Tokenizer-Dialect", "anthropic")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	var got countBody
	json.Unmarshal(rec.Body.Bytes(), &got)
	if rec.Code != 200 || got.Tokens != want {
		t.Fatalf("Anthropic count %d (%s), native %d", got.Tokens, rec.Body, want)
	}
}

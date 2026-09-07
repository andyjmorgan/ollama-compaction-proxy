package render_test

import (
	"strings"
	"testing"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/extract"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/render"
)

func parse(t *testing.T, body string) *extract.Request {
	t.Helper()
	req, err := extract.Parse([]byte(body))
	if err != nil {
		t.Fatalf("extract.Parse: %v", err)
	}
	return req
}

const chatTemplate = `{{- range .Messages }}<|{{ .Role }}|>{{ .Content }}{{ end }}`

func TestTierSelection(t *testing.T) {
	cases := []struct {
		name  string
		model render.Model
		want  render.Tier
	}{
		{"renderer wins over template", render.Model{Renderer: "gemma4", Template: "{{ .Prompt }}"}, render.TierRenderer},
		{"template when no renderer", render.Model{Template: chatTemplate}, render.TierTemplate},
		{"concat when neither", render.Model{}, render.TierConcat},
		{"unknown renderer degrades to template", render.Model{Renderer: "nope", Template: chatTemplate}, render.TierTemplate},
		{"unknown renderer and no template degrades to concat", render.Model{Renderer: "nope"}, render.TierConcat},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := render.Render(parse(t, `{"model":"m","messages":[{"role":"user","content":"hello"}]}`), tc.model)
			if got.Tier != tc.want {
				t.Errorf("Tier = %q, want %q (fallback %q)", got.Tier, tc.want, got.Fallback)
			}
			if !strings.Contains(got.Text, "hello") {
				t.Errorf("rendered text lost the prompt: %q", got.Text)
			}
		})
	}
}

// Every tier must preserve prompt text; a renderer failure must never silently
// drop content.
func TestUnknownRendererKeepsContentAndRecordsReason(t *testing.T) {
	got := render.Render(parse(t, `{"model":"m","prompt":"hello"}`), render.Model{Renderer: "no-such-renderer"})

	if !strings.Contains(got.Text, "hello") {
		t.Errorf("text = %q, want it to contain the prompt", got.Text)
	}
	if got.Fallback == "" {
		t.Error("Fallback is empty, want the reason the renderer was skipped")
	}
}

func TestContentBlockNormalization(t *testing.T) {
	cases := []struct {
		name string
		body string
		want []string
	}{
		{
			"anthropic text blocks",
			`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"alpha"},{"type":"text","text":"beta"}]}]}`,
			[]string{"alpha", "beta"},
		},
		{
			"openai responses input list",
			`{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"gamma"}]}]}`,
			[]string{"gamma"},
		},
		{
			"input as bare string",
			`{"model":"m","input":"delta"}`,
			[]string{"delta"},
		},
		{
			"plain string array content",
			`{"model":"m","messages":[{"role":"user","content":["epsilon","zeta"]}]}`,
			[]string{"epsilon", "zeta"},
		},
		{
			"system and instructions both kept",
			`{"model":"m","system":"eta","instructions":"theta","prompt":"iota"}`,
			[]string{"eta", "theta", "iota"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := render.Render(parse(t, tc.body), render.Model{Template: chatTemplate})
			for _, want := range tc.want {
				if !strings.Contains(got.Text, want) {
					t.Errorf("text %q missing %q", got.Text, want)
				}
			}
		})
	}
}

func TestMediaBlocksCountedButNotPricedAsText(t *testing.T) {
	got := render.Render(parse(t, `{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"describe"},
		{"type":"image","source":{"data":"AAAA"}},
		{"type":"input_audio","audio":{"data":"BBBB"}}]}]}`), render.Model{Template: chatTemplate})

	if got.Media != 2 {
		t.Errorf("Media = %d, want 2", got.Media)
	}
	if strings.Contains(got.Text, "AAAA") || strings.Contains(got.Text, "BBBB") {
		t.Errorf("media payload leaked into rendered text: %q", got.Text)
	}
	if !strings.Contains(got.Text, "describe") {
		t.Errorf("text lost the accompanying prompt: %q", got.Text)
	}
}

// Malformed tools still cost tokens, so they must reach the rendered text
// rather than being dropped.
func TestUnparseableToolsStillCounted(t *testing.T) {
	got := render.Render(parse(t, `{"model":"m","prompt":"hi","tools":["not-a-tool-object"]}`),
		render.Model{Template: chatTemplate})

	if !strings.Contains(got.Text, "not-a-tool-object") {
		t.Errorf("text = %q, want the raw tool JSON retained", got.Text)
	}
}

func TestToolCallsAndResultsCounted(t *testing.T) {
	got := render.Render(parse(t, `{"model":"m","messages":[
		{"role":"assistant","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"Dublin"}}}]},
		{"role":"tool","tool_name":"get_weather","content":"12C and raining"}]}`),
		render.Model{Renderer: "gemma4", Template: "{{ .Prompt }}"})

	for _, want := range []string{"get_weather", "Dublin", "12C and raining"} {
		if !strings.Contains(got.Text, want) {
			t.Errorf("rendered text %q missing %q", got.Text, want)
		}
	}
}

// The renderer must produce real chat markup, not just the bare text.
func TestRendererEmitsChatMarkup(t *testing.T) {
	got := render.Render(parse(t, `{"model":"m","messages":[{"role":"user","content":"hi"}]}`),
		render.Model{Renderer: "gemma4", Template: "{{ .Prompt }}"})

	if got.Tier != render.TierRenderer {
		t.Fatalf("Tier = %q, want %q", got.Tier, render.TierRenderer)
	}
	if len(got.Text) <= len("hi") {
		t.Errorf("text = %q, want chat markup around the prompt", got.Text)
	}
}

// OpenAI Responses items carry payloads outside "content": function_call in
// "arguments", function_call_output in "output". Both cost tokens.
func TestResponsesToolItemsCounted(t *testing.T) {
	got := render.Render(parse(t, `{"model":"m","input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"look it up"}]},
		{"type":"function_call","name":"lookup","arguments":"{\"city\":\"Dublin\"}","call_id":"c1"},
		{"type":"function_call_output","call_id":"c1","output":"a very large tool result payload"}]}`),
		render.Model{Template: chatTemplate})

	for _, want := range []string{"look it up", "Dublin", "a very large tool result payload"} {
		if !strings.Contains(got.Text, want) {
			t.Errorf("rendered text %q missing %q", got.Text, want)
		}
	}
}

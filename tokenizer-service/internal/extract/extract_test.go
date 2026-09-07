package extract_test

import (
	"errors"
	"testing"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/extract"
)

func TestParseKeepsOnlyPromptBearingFields(t *testing.T) {
	body := []byte(`{
		"model": "gemma4:e4b",
		"messages": [{"role":"user","content":"hello"}],
		"system": "be brief",
		"instructions": "haiku only",
		"prompt": "hi",
		"input": "there",
		"tools": [{"type":"function"}],
		"temperature": 0.7,
		"stream": true,
		"max_tokens": 100,
		"metadata": {"user":"someone"},
		"banana": "unrecognized"
	}`)

	req, err := extract.Parse(body)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if req.Model != "gemma4:e4b" {
		t.Errorf("Model = %q, want %q", req.Model, "gemma4:e4b")
	}
	for name, got := range map[string]any{
		"Messages":     req.Messages,
		"Input":        req.Input,
		"Instructions": req.Instructions,
		"System":       req.System,
		"Prompt":       req.Prompt,
		"Tools":        req.Tools,
	} {
		if got == nil {
			t.Errorf("%s was dropped, want kept", name)
		}
	}
}

func TestParseAbsentFieldsAreNotErrors(t *testing.T) {
	req, err := extract.Parse([]byte(`{"model":"m"}`))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if req.Any() {
		t.Error("Any() = true, want false for a request with no prompt content")
	}
}

func TestParseErrors(t *testing.T) {
	cases := []struct {
		name string
		body string
		want error
	}{
		{"missing model", `{"prompt":"hi"}`, extract.ErrMissingModel},
		{"empty model", `{"model":""}`, extract.ErrMissingModel},
		{"model wrong type", `{"model":["a"]}`, extract.ErrMissingModel},
		{"truncated", `{"model":`, extract.ErrInvalidJSON},
		{"array body", `[1,2,3]`, extract.ErrInvalidJSON},
		{"string body", `"hello"`, extract.ErrInvalidJSON},
		{"empty body", ``, extract.ErrInvalidJSON},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := extract.Parse([]byte(tc.body)); !errors.Is(err, tc.want) {
				t.Errorf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

// Package testsupport provides a fake Ollama instance for tests, so the suite
// runs without a real Ollama or any model weights.
package testsupport

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// Model describes a fake installed model.
type Model struct {
	// Renderer is emitted as a RENDERER directive in the modelfile, matching
	// how Ollama 0.32.8 reports it (its ShowResponse.Renderer field is left
	// empty by the server).
	Renderer string
	Template string

	// System is the model's own default system prompt, which Ollama prepends
	// when a request supplies none.
	System string

	// Capabilities mirrors /api/show; "thinking" changes what is rendered.
	Capabilities []string
}

// Ollama is a fake Ollama HTTP server exposing /api/show.
type Ollama struct {
	URL string

	// Shows counts verbose /api/show calls, so tests can prove the tokenizer
	// cache prevents refetching.
	Shows atomic.Int64
}

// NewOllama starts a fake Ollama serving the given models, keyed by model ID.
// Any other model ID returns 404.
func NewOllama(t *testing.T, models map[string]Model) *Ollama {
	t.Helper()

	fake := &Ollama{}
	vocab := ByteVocabulary()

	mux := http.NewServeMux()
	mux.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		model, ok := models[req.Model]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "model '" + req.Model + "' not found"})
			return
		}
		fake.Shows.Add(1)

		modelfile := "FROM /fake/blob\nTEMPLATE " + model.Template + "\n"
		if model.Renderer != "" {
			modelfile += "RENDERER " + model.Renderer + "\n"
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"modelfile":    modelfile,
			"template":     model.Template,
			"system":       model.System,
			"capabilities": model.Capabilities,
			"model_info":   vocab,
		})
	})

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	fake.URL = srv.URL
	return fake
}

// ByteVocabulary returns GGUF tokenizer metadata for a byte-level BPE
// vocabulary: the 256 GPT-2 byte characters plus two special tokens, and no
// merges.
//
// Every input byte therefore costs exactly one token, which makes counts
// deterministic and monotonic in input length without pinning a real model
// fixture or asserting magic numbers.
func ByteVocabulary() map[string]any {
	const (
		bosToken = "<bos>"
		eosToken = "<eos>"
	)

	tokens := make([]string, 0, 258)
	types := make([]int32, 0, 258)
	for b := range 256 {
		tokens = append(tokens, string(byteRune(byte(b))))
		types = append(types, 1) // TOKEN_TYPE_NORMAL
	}
	bosID := len(tokens)
	tokens = append(tokens, bosToken)
	types = append(types, 3) // TOKEN_TYPE_CONTROL
	eosID := len(tokens)
	tokens = append(tokens, eosToken)
	types = append(types, 3)

	return map[string]any{
		"tokenizer.ggml.model":         "gpt2",
		"tokenizer.ggml.pre":           "default",
		"tokenizer.ggml.tokens":        tokens,
		"tokenizer.ggml.token_type":    types,
		"tokenizer.ggml.merges":        []string{},
		"tokenizer.ggml.bos_token_id":  bosID,
		"tokenizer.ggml.eos_token_id":  eosID,
		"tokenizer.ggml.add_bos_token": false,
		"tokenizer.ggml.add_eos_token": false,
	}
}

// byteRune maps a byte to its GPT-2 byte-level codepoint, mirroring the
// normalization in Ollama's BytePairEncoding.Encode.
func byteRune(b byte) rune {
	r := rune(b)
	switch {
	case r == 0x00ad:
		r = 0x0143
	case r <= 0x0020:
		r += 0x0100
	case r >= 0x007f && r <= 0x00a0:
		r += 0x00a2
	}
	return r
}

// CountBytes returns the token count the byte-level fixture vocabulary yields
// for text: one token per byte, ignoring the whitespace the pretokenizer
// never emits.
func CountBytes(text string) int {
	return len([]byte(text))
}

// ChatTemplate is a minimal but realistic chat template: it exercises the
// Messages branch of Ollama's template engine and renders tools, as every
// tool-capable model's template does.
const ChatTemplate = `{{- if .Tools }}<|tools|>
{{ .Tools }}
{{ end }}
{{- range .Messages }}<|{{ .Role }}|>
{{ .Content }}
{{ end }}`

// NoToolsTemplate renders messages but ignores tools, like a model without
// tool support.
const NoToolsTemplate = `{{- range .Messages }}<|{{ .Role }}|>
{{ .Content }}
{{ end }}`

// StubTemplate mirrors the placeholder template shipped by renderer-based
// models such as gemma4 and muse-glimmer.
const StubTemplate = `{{ .Prompt }}`

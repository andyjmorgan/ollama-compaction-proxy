// Package extract pulls the prompt-bearing parts out of an arbitrary request
// body. It deliberately does not model OpenAI, Anthropic, or Ollama request
// schemas: unknown fields are ignored rather than rejected, so a new field in
// some provider's API is never an error here.
package extract

import (
	"encoding/json"
	"errors"
)

// ErrInvalidJSON reports a body that is not a JSON object.
var ErrInvalidJSON = errors.New("invalid JSON")

// ErrMissingModel reports a body with no usable "model" field.
var ErrMissingModel = errors.New("model is required")

// Request is the extracted content of a count request. Every field except
// Model is optional, and absence is never an error.
type Request struct {
	Model string

	// Raw JSON values of the recognized prompt-bearing fields, nil when the
	// field was absent.
	Messages     json.RawMessage
	Input        json.RawMessage
	Instructions json.RawMessage
	System       json.RawMessage
	Prompt       json.RawMessage
	Tools        json.RawMessage

	// Think is not prompt text, but it decides whether a thinking block is
	// rendered into the prompt, so it changes the count. Ollama accepts a bool
	// or a level string.
	Think json.RawMessage
}

// Any reports whether any prompt-bearing field was present. A request with
// none is valid and counts as whatever the model's template emits on its own.
func (r *Request) Any() bool {
	return r.Messages != nil || r.Input != nil || r.Instructions != nil ||
		r.System != nil || r.Prompt != nil || r.Tools != nil
}

// Parse reads a request body. Fields that are not prompt-bearing —
// temperature, stream, max_tokens, metadata, and anything unrecognized — are
// dropped, so they can never contribute to a token count.
func Parse(body []byte) (*Request, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, ErrInvalidJSON
	}

	var model string
	if raw, ok := fields["model"]; ok {
		// A non-string model is as unusable as a missing one.
		_ = json.Unmarshal(raw, &model)
	}
	if model == "" {
		return nil, ErrMissingModel
	}

	return &Request{
		Model:        model,
		Messages:     fields["messages"],
		Input:        fields["input"],
		Instructions: fields["instructions"],
		System:       fields["system"],
		Prompt:       fields["prompt"],
		Tools:        fields["tools"],
		Think:        fields["think"],
	}, nil
}

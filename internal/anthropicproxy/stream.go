package anthropicproxy

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// terminalAnthropic reports whether an SSE frame ends a Messages stream.
func terminalAnthropic(f upstream.Frame) bool {
	return f.Event == "message_stop" || f.Event == "error"
}

// truncationErrorFrame is the synthesized terminal error for a stream that
// ended without a terminal event (Ollama swallows mid-stream errors).
func truncationErrorFrame() upstream.Frame {
	data, _ := json.Marshal(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "api_error",
			"message": "upstream stream ended unexpectedly",
		},
	})
	return upstream.Frame{Event: "error", Data: data}
}

// injector rewrites an upstream Messages stream on a compaction turn: the
// outer message id is replaced, the compaction block trio is emitted at index
// 0, every upstream content block index shifts by one, and message_delta's
// usage gains the iterations entries.
type injector struct {
	c        *compactionResult
	model    string
	mintedID string

	emittedBlock bool
}

func newInjector(c *compactionResult, model string) *injector {
	return &injector{c: c, model: model, mintedID: httpapi.MintID("msg")}
}

// rewrite maps one upstream frame to the frames to emit in its place.
func (in *injector) rewrite(f upstream.Frame) ([]upstream.Frame, error) {
	switch f.Event {
	case "message_start":
		patched, err := patchMessageStart(f.Data, in.mintedID)
		if err != nil {
			return nil, err
		}
		out := []upstream.Frame{{Event: "message_start", Data: patched}}
		out = append(out, in.compactionTrio()...)
		in.emittedBlock = true
		return out, nil

	case "content_block_start", "content_block_delta", "content_block_stop":
		patched, err := shiftIndex(f.Data, 1)
		if err != nil {
			return nil, err
		}
		return []upstream.Frame{{Event: f.Event, Data: patched}}, nil

	case "message_delta":
		patched, err := in.patchMessageDelta(f.Data)
		if err != nil {
			return nil, err
		}
		return []upstream.Frame{{Event: "message_delta", Data: patched}}, nil

	default:
		return []upstream.Frame{f}, nil
	}
}

// compactionTrio emits the block per the contract: start with null fields,
// one single-shot compaction_delta carrying everything, then stop.
func (in *injector) compactionTrio() []upstream.Frame {
	start, _ := json.Marshal(map[string]any{
		"type":  "content_block_start",
		"index": 0,
		"content_block": map[string]any{
			"type":              "compaction",
			"content":           nil,
			"encrypted_content": nil,
		},
	})
	delta, _ := json.Marshal(map[string]any{
		"type":  "content_block_delta",
		"index": 0,
		"delta": map[string]any{
			"type":              "compaction_delta",
			"content":           in.c.summary,
			"encrypted_content": in.c.blob,
		},
	})
	stop, _ := json.Marshal(map[string]any{
		"type":  "content_block_stop",
		"index": 0,
	})
	return []upstream.Frame{
		{Event: "content_block_start", Data: start},
		{Event: "content_block_delta", Data: delta},
		{Event: "content_block_stop", Data: stop},
	}
}

// patchMessageStart replaces message.id, leaving every other byte of the
// nested payload untouched.
func patchMessageStart(data []byte, id string) ([]byte, error) {
	var outer map[string]json.RawMessage
	if err := json.Unmarshal(data, &outer); err != nil {
		return nil, fmt.Errorf("parse message_start: %w", err)
	}
	var msg map[string]json.RawMessage
	if err := json.Unmarshal(outer["message"], &msg); err != nil {
		return nil, fmt.Errorf("parse message_start.message: %w", err)
	}
	idJSON, _ := json.Marshal(id)
	msg["id"] = idJSON
	patchedMsg, err := json.Marshal(msg)
	if err != nil {
		return nil, err
	}
	outer["message"] = patchedMsg
	return json.Marshal(outer)
}

// shiftIndex bumps the top-level "index" field by delta via raw-JSON surgery.
func shiftIndex(data []byte, delta int) ([]byte, error) {
	var outer map[string]json.RawMessage
	if err := json.Unmarshal(data, &outer); err != nil {
		return nil, fmt.Errorf("parse content_block event: %w", err)
	}
	raw, ok := outer["index"]
	if !ok {
		return data, nil
	}
	idx, err := strconv.Atoi(string(raw))
	if err != nil {
		return nil, fmt.Errorf("parse index %q: %w", raw, err)
	}
	outer["index"] = json.RawMessage(strconv.Itoa(idx + delta))
	return json.Marshal(outer)
}

// patchMessageDelta attaches usage.iterations, keeping upstream's top-level
// token counts (which exclude the compaction iteration by construction).
func (in *injector) patchMessageDelta(data []byte) ([]byte, error) {
	var outer map[string]json.RawMessage
	if err := json.Unmarshal(data, &outer); err != nil {
		return nil, fmt.Errorf("parse message_delta: %w", err)
	}

	var usage map[string]json.RawMessage
	if raw, ok := outer["usage"]; ok && string(raw) != "null" {
		if err := json.Unmarshal(raw, &usage); err != nil {
			return nil, fmt.Errorf("parse message_delta.usage: %w", err)
		}
	} else {
		usage = map[string]json.RawMessage{}
	}

	msgIteration := map[string]any{
		"type":  "message",
		"model": in.model,
	}
	if raw, ok := usage["input_tokens"]; ok {
		msgIteration["input_tokens"] = json.RawMessage(raw)
	}
	if raw, ok := usage["output_tokens"]; ok {
		msgIteration["output_tokens"] = json.RawMessage(raw)
	}

	iterationsJSON, err := json.Marshal([]any{
		map[string]any{
			"type":          "compaction",
			"input_tokens":  in.c.usage.InputTokens,
			"output_tokens": in.c.usage.OutputTokens,
		},
		msgIteration,
	})
	if err != nil {
		return nil, err
	}
	usage["iterations"] = iterationsJSON

	patchedUsage, err := json.Marshal(usage)
	if err != nil {
		return nil, err
	}
	outer["usage"] = patchedUsage
	return json.Marshal(outer)
}

// pauseStream synthesizes the whole stream for pause_after_compaction — no
// upstream call happens.
func pauseStream(model string, c *compactionResult) []upstream.Frame {
	in := newInjector(c, model)

	start, _ := json.Marshal(map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":      in.mintedID,
			"type":    "message",
			"role":    "assistant",
			"model":   model,
			"content": []any{},
			"usage":   map[string]int{"input_tokens": 0, "output_tokens": 0},
		},
	})

	delta, _ := json.Marshal(map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "compaction"},
		"usage": map[string]any{
			"input_tokens":  0,
			"output_tokens": 0,
			"iterations": []any{map[string]any{
				"type":          "compaction",
				"input_tokens":  c.usage.InputTokens,
				"output_tokens": c.usage.OutputTokens,
			}},
		},
	})
	stop, _ := json.Marshal(map[string]any{"type": "message_stop"})

	frames := []upstream.Frame{{Event: "message_start", Data: start}}
	frames = append(frames, in.compactionTrio()...)
	frames = append(frames,
		upstream.Frame{Event: "message_delta", Data: delta},
		upstream.Frame{Event: "message_stop", Data: stop},
	)
	return frames
}

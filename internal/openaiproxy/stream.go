package openaiproxy

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// eventType reads the semantic type of a Responses stream frame — from the
// payload, which is authoritative regardless of the SSE event-name line.
func eventType(f upstream.Frame) string {
	var peek struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(f.Data, &peek)
	return peek.Type
}

// terminalOpenAI reports whether a frame ends a Responses stream.
func terminalOpenAI(f upstream.Frame) bool {
	switch eventType(f) {
	case "response.completed", "response.failed", "response.incomplete":
		return true
	}
	return false
}

// failedFrame synthesizes the terminal error for a truncated stream.
func failedFrame(seq int) upstream.Frame {
	data, _ := json.Marshal(map[string]any{
		"type":            "response.failed",
		"sequence_number": seq,
		"response": map[string]any{
			"object": "response",
			"status": "failed",
			"error": map[string]any{
				"code":    "server_error",
				"message": "upstream stream ended unexpectedly",
			},
		},
	})
	return upstream.Frame{Event: "response.failed", Data: data}
}

// streamInjector rewrites an upstream Responses stream on a compaction turn.
// The proxy owns sequence numbering outright: every emitted frame is
// re-stamped from a monotonic counter.
type streamInjector struct {
	c        *compactionResult
	mintedID string
	seq      int

	injected bool
}

func newStreamInjector(c *compactionResult) *streamInjector {
	return &streamInjector{c: c, mintedID: httpapi.MintID("resp")}
}

func (in *streamInjector) nextSeq() int {
	s := in.seq
	in.seq++
	return s
}

// rewrite maps one upstream frame to the frames to emit in its place.
func (in *streamInjector) rewrite(f upstream.Frame) ([]upstream.Frame, error) {
	outer := map[string]json.RawMessage{}
	if err := json.Unmarshal(f.Data, &outer); err != nil {
		return nil, fmt.Errorf("parse stream event: %w", err)
	}
	typ := eventType(f)

	// Patch the nested response object where present.
	if raw, ok := outer["response"]; ok {
		patched, err := in.patchResponse(raw, typ == "response.completed")
		if err != nil {
			return nil, err
		}
		outer["response"] = patched
	}

	// Shift output_index for the injected item at position 0.
	if raw, ok := outer["output_index"]; ok {
		idx, err := strconv.Atoi(string(raw))
		if err != nil {
			return nil, fmt.Errorf("parse output_index %q: %w", raw, err)
		}
		outer["output_index"] = json.RawMessage(strconv.Itoa(idx + 1))
	}

	var out []upstream.Frame
	out = append(out, in.stamp(typ, outer))

	// Inject the compaction item right after response.created.
	if typ == "response.created" && !in.injected {
		in.injected = true
		out = append(out, in.compactionPair()...)
	}
	return out, nil
}

// stamp re-serializes an event with the next sequence number.
func (in *streamInjector) stamp(typ string, outer map[string]json.RawMessage) upstream.Frame {
	outer["sequence_number"] = json.RawMessage(strconv.Itoa(in.nextSeq()))
	data, _ := json.Marshal(outer)
	return upstream.Frame{Event: typ, Data: data}
}

// compactionPair emits the generic item events carrying the compaction item —
// there is no dedicated event type in the contract.
func (in *streamInjector) compactionPair() []upstream.Frame {
	added, _ := json.Marshal(map[string]any{
		"type":            "response.output_item.added",
		"sequence_number": in.nextSeq(),
		"output_index":    0,
		"item":            json.RawMessage(in.c.outputItemJSON(false)),
	})
	done, _ := json.Marshal(map[string]any{
		"type":            "response.output_item.done",
		"sequence_number": in.nextSeq(),
		"output_index":    0,
		"item":            json.RawMessage(in.c.outputItemJSON(true)),
	})
	return []upstream.Frame{
		{Event: "response.output_item.added", Data: added},
		{Event: "response.output_item.done", Data: done},
	}
}

// patchResponse rewrites the nested response object: minted id always; on the
// terminal event the compaction item is prepended to output and the
// compaction_usage extension attached.
func (in *streamInjector) patchResponse(raw json.RawMessage, terminal bool) (json.RawMessage, error) {
	var resp map[string]json.RawMessage
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("parse response object: %w", err)
	}

	idJSON, _ := json.Marshal(in.mintedID)
	resp["id"] = idJSON

	if terminal {
		var output []json.RawMessage
		if rawOut, ok := resp["output"]; ok && string(rawOut) != "null" {
			if err := json.Unmarshal(rawOut, &output); err != nil {
				return nil, fmt.Errorf("parse response.output: %w", err)
			}
		}
		output = append([]json.RawMessage{in.c.outputItemJSON(true)}, output...)
		outJSON, err := json.Marshal(output)
		if err != nil {
			return nil, err
		}
		resp["output"] = outJSON
		resp["compaction_usage"] = in.c.compactionUsage()
	}

	return json.Marshal(resp)
}

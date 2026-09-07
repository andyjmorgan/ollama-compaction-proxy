package openaiproxy

import (
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/compactstate"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
)

// compactRequest is the /v1/responses/compact request surface this proxy
// supports. Unknown fields are ignored, matching the dialect's leniency.
type compactRequest struct {
	Model              string          `json:"model"`
	Input              json.RawMessage `json:"input"`
	Instructions       *string         `json:"instructions"`
	PreviousResponseID *string         `json:"previous_response_id"`
}

// Compact handles POST /v1/responses/compact: standalone, client-driven
// compaction. The response's output is all user messages verbatim followed by
// a single compaction item — exactly the shape the OpenAI Agents SDK's
// OpenAIResponsesCompactionSession replaces its history with.
func (h *Handler) Compact(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req compactRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, "invalid JSON")
		return
	}
	if req.Model == "" {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, "model is required")
		return
	}
	if req.PreviousResponseID != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest,
			"previous_response_id is not supported: this endpoint is stateless")
		return
	}
	req.Model, _ = h.cfg.StripModelPrefix(req.Model)

	// Reuse the request-shaped item scanner on a minimal envelope.
	items, aerr := scanItems(req.Input)
	if aerr != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, aerr.Error())
		return
	}

	log := h.log.With("dialect", "openai", "path", "/v1/responses/compact", "model", req.Model)

	// Compact-of-compacted: reconstitute any round-tripped state first.
	for i := len(items) - 1; i >= 0; i-- {
		if items[i].typ != "compaction" {
			continue
		}
		var c struct {
			EncryptedContent string `json:"encrypted_content"`
		}
		_ = json.Unmarshal(items[i].raw, &c)
		blob, derr := compactstate.Decode(c.EncryptedContent, h.cfg.HMACKey)
		if derr != nil {
			log.Warn("rejecting invalid compaction state", "blob_invalid", true, "error", derr)
			httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest,
				"invalid or foreign compaction state in input")
			return
		}
		items = substituteIncoming(items, i, blob.Summary)
		break
	}
	items = strip(items)

	instructions := ""
	if req.Instructions != nil {
		instructions = *req.Instructions
	}
	res, err := h.summarizer.Summarize(r.Context(), req.Model, instructions, transcript(items))
	if err != nil {
		log.Error("summarization failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadGateway, "compaction failed")
		return
	}

	blob, err := compactstate.Encode(compactstate.Blob{
		Kind:      "openai",
		Summary:   res.Summary,
		Model:     req.Model,
		SumModel:  res.Model,
		CreatedAt: time.Now().Unix(),
		InTokens:  res.InputTokens,
		OutTokens: res.OutputTokens,
	}, h.cfg.HMACKey)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusInternalServerError, "failed to encode compaction state")
		return
	}
	c := newCompactionResult(res.Summary, blob, *res)

	// Output: all user messages verbatim, then the single compaction item.
	var output []json.RawMessage
	for _, it := range items {
		if it.isUserMessage() {
			output = append(output, it.raw)
		}
	}
	output = append(output, c.outputItemJSON(true))

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":         httpapi.MintID("respc"),
		"object":     "response.compaction",
		"created_at": time.Now().Unix(),
		"output":     output,
		"usage": map[string]int{
			"input_tokens":  res.InputTokens,
			"output_tokens": res.OutputTokens,
			"total_tokens":  res.InputTokens + res.OutputTokens,
		},
	})
	log.Info("request served", "status", 200, "compaction_triggered", true,
		"summarizer_model", res.Model, "duration_ms", time.Since(start).Milliseconds())
}

// scanItems parses an input value (string or item array) into scanned items.
func scanItems(input json.RawMessage) ([]item, error) {
	if len(input) == 0 {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(input, &s); err == nil {
		raw, _ := json.Marshal(map[string]any{
			"type": "message", "role": "user",
			"content": []map[string]string{{"type": "input_text", "text": s}},
		})
		return []item{{raw: raw, typ: "message", role: "user"}}, nil
	}

	var rawItems []json.RawMessage
	if err := json.Unmarshal(input, &rawItems); err != nil {
		return nil, err
	}
	items := make([]item, 0, len(rawItems))
	for _, raw := range rawItems {
		var peek struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		if err := json.Unmarshal(raw, &peek); err != nil {
			return nil, err
		}
		items = append(items, item{raw: raw, typ: peek.Type, role: peek.Role})
	}
	return items, nil
}

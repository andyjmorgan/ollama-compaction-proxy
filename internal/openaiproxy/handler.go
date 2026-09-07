package openaiproxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/andyjmorgan/slipspace-gateway/protocols/openai/responses"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/compactstate"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/config"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/tokencount"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// Handler serves /v1/responses and /v1/responses/compact.
type Handler struct {
	cfg        *config.Config
	upstream   *upstream.Client
	counter    *tokencount.Client
	summarizer *summarize.Summarizer
	log        *slog.Logger
}

// New wires the OpenAI-dialect handler.
func New(cfg *config.Config, up *upstream.Client, counter *tokencount.Client, sum *summarize.Summarizer, log *slog.Logger) *Handler {
	return &Handler{cfg: cfg, upstream: up, counter: counter, summarizer: sum, log: log}
}

// Responses handles POST /v1/responses.
func (h *Handler) Responses(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req responses.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, "invalid JSON")
		return
	}

	// Statelessness contract: this proxy stores nothing, loudly.
	if req.PreviousResponseID != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest,
			"previous_response_id is not supported: this endpoint is stateless; chain turns by appending output items to input")
		return
	}
	if _, ok := req.Extra["conversation"]; ok {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest,
			"conversation is not supported: this endpoint is stateless")
		return
	}

	var prefixed bool
	req.Model, prefixed = h.cfg.StripModelPrefix(req.Model)

	a, err := analyze(&req, h.cfg.DefaultThresholdOpenAI, h.cfg.MinTrigger)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest, err.Error())
		return
	}

	stream := req.Stream != nil && *req.Stream
	log := h.log.With(
		"dialect", "openai", "path", "/v1/responses", "model", req.Model,
		"stream", stream, "incoming_compaction", a.lastCompaction >= 0,
	)

	// Fast path: nothing to consume or rewrite — forward original bytes.
	if a.compactionEntry == nil && a.lastCompaction < 0 && !a.forced && !prefixed {
		if _, hasParam := req.Extra["context_management"]; !hasParam {
			h.forward(w, r.Context(), body, stream, nil, log, start)
			return
		}
	}

	items := a.items

	// Substitution: reconstitute history from the round-tripped item.
	if a.lastCompaction >= 0 {
		blob, derr := compactstate.Decode(a.incomingBlob, h.cfg.HMACKey)
		if derr != nil {
			// encrypted_content is the ONLY carrier of the compacted context;
			// silently dropping it would produce amnesiac responses. 400.
			log.Warn("rejecting invalid compaction state", "blob_invalid", true, "error", derr)
			httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadRequest,
				"invalid or foreign compaction state in input")
			return
		}
		items = substituteIncoming(items, a.lastCompaction, blob.Summary)
	}
	items = strip(items)

	// Trigger decision.
	var compaction *compactionResult
	if a.compactionEntry != nil || a.forced {
		triggered := a.forced
		if !triggered {
			count, cerr := h.countItems(r.Context(), &req, items)
			if cerr != nil {
				log.Error("token count unavailable; skipping compaction trigger", "error", cerr)
			} else {
				log = log.With("tokens_post_substitution", count, "trigger_value", a.threshold)
				triggered = count >= a.threshold
			}
		}
		if triggered {
			compaction, items = h.compact(r.Context(), req.Model, items, log)
		}
	}

	// Rebuild the forward request.
	if !a.stringInput {
		req.Input = marshalItems(items)
	}
	delete(req.Extra, "context_management")
	forwardBody, err := json.Marshal(&req)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusInternalServerError, "failed to rebuild request")
		return
	}

	h.forward(w, r.Context(), forwardBody, stream, compaction, log, start)
}

// compact runs the summarizer over the compactable region. Failure is a
// no-op: nil result and the items unchanged.
func (h *Handler) compact(ctx context.Context, model string, items []item, log *slog.Logger) (*compactionResult, []item) {
	keepUsers, tail, ok := splitForCompaction(items)
	if !ok {
		return nil, items
	}

	sumStart := time.Now()
	res, err := h.summarizer.Summarize(ctx, model, "", transcript(items))
	if err != nil {
		log.Error("summarization failed; compaction skipped", "error", err)
		return nil, items
	}

	blob, err := compactstate.Encode(compactstate.Blob{
		Kind:      "openai",
		Summary:   res.Summary,
		Model:     model,
		SumModel:  res.Model,
		CreatedAt: time.Now().Unix(),
		InTokens:  res.InputTokens,
		OutTokens: res.OutputTokens,
	}, h.cfg.HMACKey)
	if err != nil {
		log.Error("blob encode failed; compaction skipped", "error", err)
		return nil, items
	}

	log.Info("compaction ran",
		"summarizer_model", res.Model,
		"summarizer_ms", time.Since(sumStart).Milliseconds(),
		"summarizer_in_tokens", res.InputTokens,
		"summarizer_out_tokens", res.OutputTokens,
	)
	return newCompactionResult(res.Summary, blob, *res), rebuildInput(keepUsers, res.Summary, tail)
}

// forward relays the exchange, injecting compaction state when present.
func (h *Handler) forward(w http.ResponseWriter, ctx context.Context, body []byte, stream bool, c *compactionResult, log *slog.Logger, start time.Time) {
	if stream {
		h.forwardStream(w, ctx, body, c, log, start)
		return
	}

	resp, err := h.upstream.PostJSON(ctx, "/v1/responses", body)
	if err != nil {
		log.Error("upstream call failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadGateway, "upstream unavailable")
		return
	}

	out := resp.Body
	if resp.StatusCode == http.StatusOK && c != nil {
		out, err = synthesizeResponse(resp.Body, c)
		if err != nil {
			log.Error("response synthesis failed", "error", err)
			httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadGateway, "failed to process upstream response")
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(out)
	log.Info("request served", "status", resp.StatusCode, "compaction_triggered", c != nil,
		"upstream_status", resp.StatusCode, "duration_ms", time.Since(start).Milliseconds())
}

func (h *Handler) forwardStream(w http.ResponseWriter, ctx context.Context, body []byte, c *compactionResult, log *slog.Logger, start time.Time) {
	resp, err := h.upstream.PostStream(ctx, "/v1/responses", body)
	if err != nil {
		log.Error("upstream call failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectOpenAI, http.StatusBadGateway, "upstream unavailable")
		return
	}
	defer resp.Close()

	if resp.StatusCode != http.StatusOK {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(resp.ErrorBody)
		log.Info("request served", "status", resp.StatusCode, "upstream_status", resp.StatusCode,
			"duration_ms", time.Since(start).Milliseconds())
		return
	}

	fw := upstream.NewFrameWriter(w)
	var in *streamInjector
	if c != nil {
		in = newStreamInjector(c)
	}

	sawTerminal := false
	frames := 0
	for {
		frame, err := resp.Frames.Next()
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Error("upstream stream read failed", "error", err)
			}
			break
		}
		if frame.IsDone() {
			break
		}
		frames++

		out := []upstream.Frame{frame}
		if in != nil {
			out, err = in.rewrite(frame)
			if err != nil {
				log.Error("stream rewrite failed", "error", err)
				break
			}
		}
		for _, f := range out {
			if terminalOpenAI(f) {
				sawTerminal = true
			}
			if err := fw.Write(f); err != nil {
				return // client went away
			}
		}
		if sawTerminal {
			break
		}
	}

	if !sawTerminal {
		seq := frames + 16
		if in != nil {
			seq = in.seq
		}
		_ = fw.Write(failedFrame(seq))
		log.Warn("stream truncated; synthesized response.failed", "stream_truncated", true)
	}
	log.Info("request served", "status", 200, "compaction_triggered", c != nil,
		"duration_ms", time.Since(start).Milliseconds())
}

// countItems counts a would-be request whose input is items.
func (h *Handler) countItems(ctx context.Context, req *responses.ResponsesRequest, items []item) (int, error) {
	payload := map[string]any{
		"model": req.Model,
		"input": marshalItems(items),
	}
	if req.Instructions != nil {
		payload["instructions"] = *req.Instructions
	}
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return 0, err
	}
	return h.counter.Count(ctx, raw)
}

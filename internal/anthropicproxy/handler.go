package anthropicproxy

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/andyjmorgan/slipspace-gateway/protocols/anthropic/messages"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/compactstate"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/config"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/tokencount"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// Handler serves /v1/messages and /v1/messages/count_tokens.
type Handler struct {
	cfg        *config.Config
	upstream   *upstream.Client
	counter    *tokencount.Client
	summarizer *summarize.Summarizer
	log        *slog.Logger
}

// New wires the Anthropic-dialect handler.
func New(cfg *config.Config, up *upstream.Client, counter *tokencount.Client, sum *summarize.Summarizer, log *slog.Logger) *Handler {
	return &Handler{cfg: cfg, upstream: up, counter: counter, summarizer: sum, log: log}
}

// Messages handles POST /v1/messages.
func (h *Handler) Messages(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req messages.MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "invalid JSON")
		return
	}

	// A gateway-routed "agent/<model>" resolves to the bare model; the prefix
	// forces the rewrite path so the forwarded body carries the real name.
	var prefixed bool
	req.Model, prefixed = h.cfg.StripModelPrefix(req.Model)

	a := analyze(&req, h.cfg.DefaultTriggerAnthropic, h.cfg.MinTrigger)
	log := h.log.With(
		"dialect", "anthropic", "path", "/v1/messages", "model", req.Model,
		"stream", req.Stream, "incoming_compaction", a.incoming != nil,
	)

	// Fast path: nothing to rewrite, forward the original bytes untouched.
	if !a.needsRewrite() && !prefixed {
		h.forward(w, r.Context(), body, req.Stream, nil, req.Model, log, start)
		return
	}

	// Substitution: reconstitute the compacted history from the round-tripped
	// block before anything else sees the request.
	if a.incoming != nil {
		summary, blobInvalid, serr := resolveSummary(a.incoming, h.cfg.HMACKey)
		log = log.With("blob_invalid", blobInvalid)
		if serr != nil {
			// Failed-compaction no-op per contract: drop the block, keep
			// the full history.
			if err := dropBlock(&req, a.incoming); err != nil {
				httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "unprocessable compaction block")
				return
			}
		} else if err := substitute(&req, a.incoming, summary); err != nil {
			log.Error("substitution failed", "error", err)
			httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "unprocessable compaction block")
			return
		}
	}
	normalize(&req)

	// Trigger decision: exact count of the request as it now stands.
	var compaction *compactionResult
	if a.edit != nil {
		count, cerr := h.countRequest(r.Context(), &req)
		if cerr != nil {
			// Never fail the user's turn on a counting outage.
			log.Error("token count unavailable; skipping compaction trigger", "error", cerr)
		} else {
			log = log.With("tokens_post_substitution", count, "trigger_value", a.trigger)
			if count >= a.trigger {
				compaction = h.compact(r.Context(), &req, a, log)
			}
		}
	}

	if compaction != nil && a.edit.PauseAfterCompaction != nil && *a.edit.PauseAfterCompaction {
		h.respondPause(w, &req, compaction, log, start)
		return
	}

	forwardBody, err := json.Marshal(&req)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusInternalServerError, "failed to rebuild request")
		return
	}
	h.forward(w, r.Context(), forwardBody, req.Stream, compaction, req.Model, log, start)
}

// compact runs the summarizer and rebuilds req around the summary. A failure
// is a no-op (nil result, full history forwarded), per the failed-compaction
// contract.
func (h *Handler) compact(ctx context.Context, req *messages.MessagesRequest, a *analysis, log *slog.Logger) *compactionResult {
	head, tail, ok := splitForCompaction(req.Messages)
	if !ok {
		return nil
	}

	sumStart := time.Now()
	res, err := h.summarizer.Summarize(ctx, req.Model, a.edit.Instructions, transcript(head))
	if err != nil {
		log.Error("summarization failed; compaction skipped", "error", err)
		return nil
	}

	blob, err := compactstate.Encode(compactstate.Blob{
		Kind:      "anthropic",
		Summary:   res.Summary,
		Model:     req.Model,
		SumModel:  res.Model,
		CreatedAt: time.Now().Unix(),
		InTokens:  res.InputTokens,
		OutTokens: res.OutputTokens,
	}, h.cfg.HMACKey)
	if err != nil {
		log.Error("blob encode failed; compaction skipped", "error", err)
		return nil
	}

	req.Messages = append([]messages.Message{summaryTurn(res.Summary)}, tail...)
	log.Info("compaction ran",
		"summarizer_model", res.Model,
		"summarizer_ms", time.Since(sumStart).Milliseconds(),
		"summarizer_in_tokens", res.InputTokens,
		"summarizer_out_tokens", res.OutputTokens,
	)
	return &compactionResult{summary: res.Summary, blob: blob, usage: *res}
}

// respondPause serves the pause_after_compaction path — no generation.
func (h *Handler) respondPause(w http.ResponseWriter, req *messages.MessagesRequest, c *compactionResult, log *slog.Logger, start time.Time) {
	if req.Stream {
		fw := upstream.NewFrameWriter(w)
		for _, f := range pauseStream(req.Model, c) {
			if err := fw.Write(f); err != nil {
				return
			}
		}
	} else {
		body, err := pauseResponse(req.Model, c)
		if err != nil {
			httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusInternalServerError, "failed to build response")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}
	log.Info("request served", "status", 200, "pause", true, "compaction_triggered", true,
		"duration_ms", time.Since(start).Milliseconds())
}

// forward sends the request upstream and relays the response, injecting
// compaction state when present.
func (h *Handler) forward(w http.ResponseWriter, ctx context.Context, body []byte, stream bool, c *compactionResult, model string, log *slog.Logger, start time.Time) {
	if stream {
		h.forwardStream(w, ctx, body, c, model, log, start)
		return
	}

	resp, err := h.upstream.PostJSON(ctx, "/v1/messages", body)
	if err != nil {
		log.Error("upstream call failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadGateway, "upstream unavailable")
		return
	}

	out := resp.Body
	if resp.StatusCode == http.StatusOK && c != nil {
		out, err = synthesizeResponse(resp.Body, c)
		if err != nil {
			log.Error("response synthesis failed", "error", err)
			httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadGateway, "failed to process upstream response")
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(out)
	log.Info("request served", "status", resp.StatusCode, "compaction_triggered", c != nil,
		"upstream_status", resp.StatusCode, "duration_ms", time.Since(start).Milliseconds())
}

// forwardStream relays an SSE exchange, in passthrough or rewrite mode, with
// the truncation watchdog in both.
func (h *Handler) forwardStream(w http.ResponseWriter, ctx context.Context, body []byte, c *compactionResult, model string, log *slog.Logger, start time.Time) {
	resp, err := h.upstream.PostStream(ctx, "/v1/messages", body)
	if err != nil {
		log.Error("upstream call failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadGateway, "upstream unavailable")
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
	var in *injector
	if c != nil {
		in = newInjector(c, model)
	}

	sawTerminal := false
	for {
		frame, err := resp.Frames.Next()
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Error("upstream stream read failed", "error", err)
			}
			break
		}

		out := []upstream.Frame{frame}
		if in != nil {
			out, err = in.rewrite(frame)
			if err != nil {
				log.Error("stream rewrite failed", "error", err)
				break
			}
		}
		for _, f := range out {
			if terminalAnthropic(f) {
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
		_ = fw.Write(truncationErrorFrame())
		log.Warn("stream truncated; synthesized terminal error", "stream_truncated", true)
	}
	log.Info("request served", "status", 200, "compaction_triggered", c != nil,
		"duration_ms", time.Since(start).Milliseconds())
}

// CountTokens handles POST /v1/messages/count_tokens. The same substitution
// as a real turn is applied first so a client holding a compaction block sees
// the true post-compaction count.
func (h *Handler) CountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "failed to read request body")
		return
	}

	var req messages.MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "invalid JSON")
		return
	}
	if req.Model == "" {
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "model is required")
		return
	}
	req.Model, _ = h.cfg.StripModelPrefix(req.Model)

	a := analyze(&req, h.cfg.DefaultTriggerAnthropic, h.cfg.MinTrigger)
	if a.incoming != nil {
		if summary, _, serr := resolveSummary(a.incoming, h.cfg.HMACKey); serr == nil {
			if err := substitute(&req, a.incoming, summary); err != nil {
				httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, "unprocessable compaction block")
				return
			}
		} else {
			_ = dropBlock(&req, a.incoming)
		}
	}
	normalize(&req)

	count, err := h.countRequest(r.Context(), &req)
	if err != nil {
		h.log.Error("count_tokens failed", "error", err)
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadGateway, "token counting unavailable")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]int{"input_tokens": count})
}

func (h *Handler) countRequest(ctx context.Context, req *messages.MessagesRequest) (int, error) {
	payload, err := json.Marshal(req)
	if err != nil {
		return 0, err
	}
	return h.counter.Count(ctx, payload)
}

package proxytest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/andyjmorgan/slipspace-gateway/protocols/anthropic/messages"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/compactstate"
)

const plainMessages = `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hello"}]}`

const compactEditMessages = `{"model":"m","max_tokens":100,` +
	`"context_management":{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":50}}]},` +
	`"messages":[` +
	`{"role":"user","content":"first question"},` +
	`{"role":"assistant","content":"first answer"},` +
	`{"role":"user","content":"second question"}]}`

// mintBlob creates a valid round-trippable blob under the test key.
func mintBlob(t *testing.T, summary string) string {
	t.Helper()
	blob, err := compactstate.Encode(compactstate.Blob{Kind: "anthropic", Summary: summary}, testKey)
	if err != nil {
		t.Fatal(err)
	}
	return blob
}

func TestAnthropicPassthroughIsByteIdentical(t *testing.T) {
	h, fake := newProxy(t)

	status, body := post(t, h, "/v1/messages", plainMessages)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	if got := string(fake.MessagesReqs[0]); got != plainMessages {
		t.Errorf("forwarded body was rewritten:\n got %s\nwant %s", got, plainMessages)
	}
	// Upstream response relayed untouched.
	if !strings.Contains(string(body), `"id":"msg_up1"`) {
		t.Errorf("response not passed through: %s", body)
	}
	if len(fake.ChatReqs) != 0 {
		t.Error("summarizer was called on a passthrough turn")
	}
}

func TestAnthropicTriggerCompacts(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 100 // over the trigger of 50

	status, body := post(t, h, "/v1/messages", compactEditMessages)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}

	// Response: compaction block first, upstream content after, minted id.
	var resp messages.MessagesResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("response does not parse as MessagesResponse: %v", err)
	}
	if resp.ID == "msg_up1" {
		t.Error("outer id was not re-minted")
	}
	if len(resp.Content) < 2 || resp.Content[0].BlockType() != "compaction" {
		t.Fatalf("compaction block not first in content: %s", body)
	}
	block := resp.Content[0].(*messages.UnknownBlock)
	var summary string
	_ = json.Unmarshal(block.Extra["content"], &summary)
	if summary != fake.SummaryText {
		t.Errorf("block content = %q, want the summary", summary)
	}
	var encrypted string
	_ = json.Unmarshal(block.Extra["encrypted_content"], &encrypted)
	if _, err := compactstate.Decode(encrypted, testKey); err != nil {
		t.Errorf("emitted blob does not verify: %v", err)
	}

	// Usage iterations: compaction first, message second; top-level excludes
	// the compaction spend.
	if len(resp.Usage.Iterations) != 2 ||
		resp.Usage.Iterations[0].Type != "compaction" ||
		resp.Usage.Iterations[1].Type != "message" {
		t.Errorf("iterations = %+v", resp.Usage.Iterations)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("top-level input_tokens = %d, want upstream's 10", resp.Usage.InputTokens)
	}

	// Forwarded request: summary turn + final user message, no context_management.
	var fwd messages.MessagesRequest
	if err := json.Unmarshal(fake.MessagesReqs[0], &fwd); err != nil {
		t.Fatal(err)
	}
	if fwd.ContextManagement != nil {
		t.Error("context_management leaked upstream")
	}
	if len(fwd.Messages) != 2 {
		t.Fatalf("forwarded %d messages, want 2 (summary + live turn)", len(fwd.Messages))
	}
	text, _ := fwd.Messages[0].ContentAsBlocks()
	if !strings.Contains(text[0].(*messages.TextBlock).Text, fake.SummaryText) {
		t.Error("summary turn does not carry the summary")
	}
	if s, _ := fwd.Messages[1].ContentAsString(); s != "second question" {
		t.Errorf("live turn = %q, want kept verbatim", s)
	}
}

func TestAnthropicBelowTriggerForwardsFullHistory(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 10 // under the trigger of 50

	status, body := post(t, h, "/v1/messages", compactEditMessages)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	if strings.Contains(string(body), `"compaction"`) {
		t.Error("compaction block emitted below trigger")
	}
	var fwd messages.MessagesRequest
	_ = json.Unmarshal(fake.MessagesReqs[0], &fwd)
	if len(fwd.Messages) != 3 {
		t.Errorf("forwarded %d messages, want all 3", len(fwd.Messages))
	}
	if len(fake.ChatReqs) != 0 {
		t.Error("summarizer called below trigger")
	}
}

func TestAnthropicIncomingBlockSubstitutes(t *testing.T) {
	h, fake := newProxy(t)
	blob := mintBlob(t, "we discussed kubernetes at length")

	body := `{"model":"m","max_tokens":100,"messages":[` +
		`{"role":"user","content":"old question"},` +
		`{"role":"assistant","content":[` +
		`{"type":"compaction","content":"we discussed kubernetes at length","encrypted_content":"` + blob + `"},` +
		`{"type":"text","text":"continuation after compaction"}]},` +
		`{"role":"user","content":"and now?"}]}`

	status, respBody := post(t, h, "/v1/messages", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, respBody)
	}

	var fwd messages.MessagesRequest
	if err := json.Unmarshal(fake.MessagesReqs[0], &fwd); err != nil {
		t.Fatal(err)
	}
	// Expect: summary turn, assistant continuation, live user turn.
	if len(fwd.Messages) != 3 {
		t.Fatalf("forwarded %d messages, want 3: %s", len(fwd.Messages), fake.MessagesReqs[0])
	}
	blocks, _ := fwd.Messages[0].ContentAsBlocks()
	if !strings.Contains(blocks[0].(*messages.TextBlock).Text, "kubernetes at length") {
		t.Error("summary turn missing the reconstituted summary")
	}
	if fwd.Messages[1].Role != "assistant" {
		t.Errorf("message[1].role = %q, want assistant continuation", fwd.Messages[1].Role)
	}
	if strings.Contains(string(fake.MessagesReqs[0]), "old question") {
		t.Error("pre-compaction history leaked upstream")
	}
	if strings.Contains(string(fake.MessagesReqs[0]), `"compaction"`) {
		t.Error("compaction block leaked upstream")
	}
}

func TestAnthropicBlobTamperMatrix(t *testing.T) {
	tampered := mintBlob(t, "real summary")
	tampered = tampered[:len(tampered)-4] + "AAAA"

	cases := []struct {
		name        string
		block       string
		wantForward string // substring the forwarded body must contain
		wantAbsent  string
	}{
		{
			// Bad MAC + plaintext content: honor the client-visible text.
			name:        "tampered blob with content",
			block:       `{"type":"compaction","content":"plaintext summary","encrypted_content":"` + tampered + `"}`,
			wantForward: "plaintext summary",
			wantAbsent:  "old question",
		},
		{
			// Bad MAC + null content: failed-compaction no-op, history kept.
			name:        "tampered blob without content",
			block:       `{"type":"compaction","content":null,"encrypted_content":"` + tampered + `"}`,
			wantForward: "old question",
			wantAbsent:  "compaction",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h, fake := newProxy(t)
			body := `{"model":"m","max_tokens":100,"messages":[` +
				`{"role":"user","content":"old question"},` +
				`{"role":"assistant","content":[` + tc.block + `]},` +
				`{"role":"user","content":"and now?"}]}`

			status, resp := post(t, h, "/v1/messages", body)
			if status != 200 {
				t.Fatalf("status = %d: %s", status, resp)
			}
			fwd := string(fake.MessagesReqs[0])
			if !strings.Contains(fwd, tc.wantForward) {
				t.Errorf("forwarded body missing %q:\n%s", tc.wantForward, fwd)
			}
			if strings.Contains(fwd, tc.wantAbsent) {
				t.Errorf("forwarded body should not contain %q:\n%s", tc.wantAbsent, fwd)
			}
		})
	}
}

func TestAnthropicPauseAfterCompaction(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 100

	body := strings.Replace(compactEditMessages,
		`"trigger":{"type":"input_tokens","value":50}`,
		`"trigger":{"type":"input_tokens","value":50},"pause_after_compaction":true`, 1)

	status, respBody := post(t, h, "/v1/messages", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, respBody)
	}

	var resp messages.MessagesResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.StopReason == nil || *resp.StopReason != "compaction" {
		t.Errorf("stop_reason = %v, want compaction", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].BlockType() != "compaction" {
		t.Errorf("content should be the compaction block only: %s", respBody)
	}
	if len(resp.Usage.Iterations) != 1 || resp.Usage.Iterations[0].Type != "compaction" {
		t.Errorf("iterations = %+v", resp.Usage.Iterations)
	}
	if resp.Usage.InputTokens != 0 || resp.Usage.OutputTokens != 0 {
		t.Errorf("top-level usage = %d/%d, want 0/0", resp.Usage.InputTokens, resp.Usage.OutputTokens)
	}
	if len(fake.MessagesReqs) != 0 {
		t.Error("generation ran despite pause_after_compaction")
	}
}

func TestAnthropicContentNullNormalized(t *testing.T) {
	h, fake := newProxy(t)

	body := `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":null}]}`
	status, resp := post(t, h, "/v1/messages", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, resp)
	}
	if strings.Contains(string(fake.MessagesReqs[0]), "null") {
		t.Errorf("null content not normalized: %s", fake.MessagesReqs[0])
	}
}

func TestAnthropicStreamingCompaction(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 100
	fake.MessagesFrames = frameList{
		{"message_start", `{"type":"message_start","message":{"id":"msg_up1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":9,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":9,"output_tokens":2}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}.frames()

	body := strings.Replace(compactEditMessages, `"max_tokens":100`, `"max_tokens":100,"stream":true`, 1)
	status, respBody := post(t, h, "/v1/messages", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, respBody)
	}

	frames := parseSSE(t, respBody)

	// Every frame must parse through the slipspace event types.
	var events []messages.StreamEvent
	for _, f := range frames {
		ev, err := messages.UnmarshalStreamEvent([]byte(f.data))
		if err != nil {
			t.Fatalf("frame does not parse: %v\n%s", err, f.data)
		}
		events = append(events, ev)
	}

	// Expected order: message_start, compaction trio at index 0, text trio at
	// index 1, message_delta, message_stop.
	wantOrder := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_stop", // compaction @0
		"content_block_start", "content_block_delta", "content_block_stop", // text @1
		"message_delta", "message_stop",
	}
	if len(events) != len(wantOrder) {
		t.Fatalf("got %d events, want %d:\n%s", len(events), len(wantOrder), respBody)
	}
	for i, want := range wantOrder {
		if events[i].EventType() != want {
			t.Errorf("event[%d] = %s, want %s", i, events[i].EventType(), want)
		}
	}

	// Compaction block start at index 0 with type compaction.
	cbs := events[1].(*messages.ContentBlockStartEvent)
	if cbs.Index != 0 || cbs.ContentBlock.BlockType() != "compaction" {
		t.Errorf("first block start = index %d type %s", cbs.Index, cbs.ContentBlock.BlockType())
	}
	// Upstream text block shifted to index 1.
	tbs := events[4].(*messages.ContentBlockStartEvent)
	if tbs.Index != 1 {
		t.Errorf("upstream block index = %d, want 1", tbs.Index)
	}
	// compaction_delta carries the summary and a verifiable blob.
	var delta struct {
		Delta struct {
			Type      string `json:"type"`
			Content   string `json:"content"`
			Encrypted string `json:"encrypted_content"`
		} `json:"delta"`
	}
	_ = json.Unmarshal([]byte(frames[2].data), &delta)
	if delta.Delta.Type != "compaction_delta" || delta.Delta.Content != fake.SummaryText {
		t.Errorf("compaction_delta = %+v", delta.Delta)
	}
	if _, err := compactstate.Decode(delta.Delta.Encrypted, testKey); err != nil {
		t.Errorf("streamed blob does not verify: %v", err)
	}
	// message_delta gained iterations.
	md := events[7].(*messages.MessageDeltaEvent)
	if len(md.Usage.Iterations) != 2 {
		t.Errorf("message_delta iterations = %+v", md.Usage.Iterations)
	}
}

func TestAnthropicTruncatedStreamSynthesizesError(t *testing.T) {
	h, fake := newProxy(t)
	fake.MessagesFrames = frameList{
		{"message_start", `{"type":"message_start","message":{"id":"msg_up1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":9,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
	}.frames()

	body := strings.Replace(plainMessages, `"max_tokens":100`, `"max_tokens":100,"stream":true`, 1)
	status, respBody := post(t, h, "/v1/messages", body)
	if status != 200 {
		t.Fatalf("status = %d", status)
	}
	frames := parseSSE(t, respBody)
	last := frames[len(frames)-1]
	if last.event != "error" || !strings.Contains(last.data, "unexpectedly") {
		t.Errorf("last frame = %+v, want synthesized error", last)
	}
}

func TestAnthropicCountTokens(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 42

	status, body := post(t, h, "/v1/messages/count_tokens", plainMessages)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	if got := mustJSON(t, body)["input_tokens"]; got != float64(42) {
		t.Errorf("input_tokens = %v, want 42", got)
	}

	// With an incoming block, the counted payload must be the substituted
	// history, not the raw one.
	blob := mintBlob(t, "short summary")
	withBlock := `{"model":"m","messages":[` +
		`{"role":"user","content":"enormous old history"},` +
		`{"role":"assistant","content":[{"type":"compaction","content":"short summary","encrypted_content":"` + blob + `"}]},` +
		`{"role":"user","content":"next"}]}`
	status, _ = post(t, h, "/v1/messages/count_tokens", withBlock)
	if status != 200 {
		t.Fatalf("status = %d", status)
	}
	counted := string(fake.CountReqs[len(fake.CountReqs)-1])
	if strings.Contains(counted, "enormous old history") {
		t.Error("count payload contains pre-compaction history")
	}
	if !strings.Contains(counted, "short summary") {
		t.Error("count payload missing the substituted summary")
	}
}

func TestAnthropicDegradedDependencies(t *testing.T) {
	t.Run("tokenizer down skips trigger", func(t *testing.T) {
		h, fake := newProxy(t)
		fake.TokenizerFails = true

		status, body := post(t, h, "/v1/messages", compactEditMessages)
		if status != 200 {
			t.Fatalf("status = %d: %s", status, body)
		}
		if strings.Contains(string(body), `"compaction"`) {
			t.Error("compaction ran without a token count")
		}
	})

	t.Run("summarizer failure is a no-op", func(t *testing.T) {
		h, fake := newProxy(t)
		fake.TokenCount = 100
		fake.SummarizeFails = true

		status, body := post(t, h, "/v1/messages", compactEditMessages)
		if status != 200 {
			t.Fatalf("status = %d: %s", status, body)
		}
		if strings.Contains(string(body), `"compaction"`) {
			t.Error("compaction block emitted despite summarizer failure")
		}
		var fwd messages.MessagesRequest
		_ = json.Unmarshal(fake.MessagesReqs[0], &fwd)
		if len(fwd.Messages) != 3 {
			t.Errorf("full history not preserved on summarizer failure")
		}
	})
}

// Compaction triggered mid-tool-loop must keep the trailing tool exchange
// intact: severing a tool_use from its tool_result orphans the pending result
// and the model loses it.
func TestAnthropicTriggerKeepsToolLoopIntact(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 100 // over the trigger of 50

	body := `{"model":"m","max_tokens":100,` +
		`"context_management":{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":50}}]},` +
		`"messages":[` +
		`{"role":"user","content":"old big question"},` +
		`{"role":"assistant","content":"old answer"},` +
		`{"role":"user","content":"fetch the incidents doc"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"get_document","input":{"name":"incidents"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"HUGE INCIDENTS PAYLOAD"}]}]}`

	status, resp := post(t, h, "/v1/messages", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, resp)
	}

	var fwd messages.MessagesRequest
	if err := json.Unmarshal(fake.MessagesReqs[0], &fwd); err != nil {
		t.Fatal(err)
	}
	// Expect: summary turn + the intact trailing loop (user ask, tool_use,
	// tool_result), with only the older turns summarized away.
	if len(fwd.Messages) != 4 {
		t.Fatalf("forwarded %d messages, want 4: %s", len(fwd.Messages), fake.MessagesReqs[0])
	}
	if s, _ := fwd.Messages[1].ContentAsString(); s != "fetch the incidents doc" {
		t.Errorf("tail[0] = %q, want the live user ask", s)
	}
	got := string(fake.MessagesReqs[0])
	if !strings.Contains(got, `"tool_use"`) || !strings.Contains(got, "HUGE INCIDENTS PAYLOAD") {
		t.Error("tool loop was severed by compaction")
	}
	if strings.Contains(got, "old big question") {
		t.Error("head was not summarized away")
	}
}

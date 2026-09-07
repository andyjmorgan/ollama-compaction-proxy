package proxytest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/andyjmorgan/slipspace-gateway/protocols/openai/responses"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/compactstate"
)

const plainResponses = `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}`

const compactParamResponses = `{"model":"m",` +
	`"context_management":[{"type":"compaction","compact_threshold":50}],` +
	`"input":[` +
	`{"type":"message","role":"user","content":[{"type":"input_text","text":"first question"}]},` +
	`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"first answer"}]},` +
	`{"type":"message","role":"user","content":[{"type":"input_text","text":"second question"}]}]}`

func mintOpenAIBlob(t *testing.T, summary string) string {
	t.Helper()
	blob, err := compactstate.Encode(compactstate.Blob{Kind: "openai", Summary: summary}, testKey)
	if err != nil {
		t.Fatal(err)
	}
	return blob
}

func TestOpenAIPassthroughIsByteIdentical(t *testing.T) {
	h, fake := newProxy(t)

	status, body := post(t, h, "/v1/responses", plainResponses)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	if got := string(fake.ResponsesReqs[0]); got != plainResponses {
		t.Errorf("forwarded body was rewritten:\n got %s\nwant %s", got, plainResponses)
	}
	if !strings.Contains(string(body), `"id":"resp_up1"`) {
		t.Errorf("response not passed through: %s", body)
	}
}

func TestOpenAIStatelessness(t *testing.T) {
	h, _ := newProxy(t)

	cases := []struct {
		name string
		body string
	}{
		{"previous_response_id", `{"model":"m","input":"hi","previous_response_id":"resp_123"}`},
		{"conversation", `{"model":"m","input":"hi","conversation":"conv_123"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, body := post(t, h, "/v1/responses", tc.body)
			if status != 400 {
				t.Fatalf("status = %d, want 400: %s", status, body)
			}
			if !strings.Contains(string(body), "stateless") {
				t.Errorf("error should explain the statelessness contract: %s", body)
			}
		})
	}
}

func TestOpenAIThresholdCompacts(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 100 // over the threshold of 50

	status, body := post(t, h, "/v1/responses", compactParamResponses)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}

	var resp responses.ResponsesResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("response does not parse as ResponsesResponse: %v", err)
	}
	if resp.ID == "resp_up1" {
		t.Error("outer id was not re-minted")
	}
	if len(resp.Output) < 2 || resp.Output[0].Type != "compaction" {
		t.Fatalf("compaction item not first in output: %s", body)
	}
	item := resp.Output[0]
	if item.ID == "" {
		t.Error("compaction item has no id")
	}
	var encrypted string
	_ = json.Unmarshal(item.EncryptedContent, &encrypted)
	if _, err := compactstate.Decode(encrypted, testKey); err != nil {
		t.Errorf("emitted blob does not verify: %v", err)
	}
	if raw, ok := item.Extra["created_by"]; !ok || string(raw) != "null" {
		t.Errorf("created_by = %s, want explicit null", raw)
	}
	if _, ok := resp.Extra["compaction_usage"]; !ok {
		t.Error("compaction_usage extension missing")
	}

	// Forwarded request: earlier user kept, assistant summarized away,
	// summary item injected, live turn kept, no compaction leakage.
	fwd := string(fake.ResponsesReqs[0])
	if strings.Contains(fwd, "first answer") {
		t.Error("assistant history leaked upstream")
	}
	if !strings.Contains(fwd, "first question") || !strings.Contains(fwd, "second question") {
		t.Error("user messages should be kept verbatim")
	}
	if !strings.Contains(fwd, fake.SummaryText) {
		t.Error("summary item missing from forwarded input")
	}
	if strings.Contains(fwd, "context_management") {
		t.Error("context_management leaked upstream")
	}
}

func TestOpenAIBelowThresholdForwardsAll(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 10

	status, body := post(t, h, "/v1/responses", compactParamResponses)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	if strings.Contains(string(body), `"compaction"`) {
		t.Error("compaction item emitted below threshold")
	}
	fwd := string(fake.ResponsesReqs[0])
	if !strings.Contains(fwd, "first answer") {
		t.Error("history should be forwarded intact below threshold")
	}
	if strings.Contains(fwd, "context_management") {
		t.Error("context_management leaked upstream")
	}
}

func TestOpenAICompactionTrigger(t *testing.T) {
	t.Run("final position forces compaction", func(t *testing.T) {
		h, fake := newProxy(t)
		// Append the trigger as the final input item.
		body := compactParamResponses[:len(compactParamResponses)-2] +
			`,{"type":"compaction_trigger"}]}`

		status, respBody := post(t, h, "/v1/responses", body)
		if status != 200 {
			t.Fatalf("status = %d: %s", status, respBody)
		}
		if !strings.Contains(string(respBody), `"compaction"`) {
			t.Error("forced compaction did not run")
		}
		if strings.Contains(string(fake.ResponsesReqs[0]), "compaction_trigger") {
			t.Error("compaction_trigger leaked upstream")
		}
	})

	t.Run("non-final position is a 400", func(t *testing.T) {
		h, _ := newProxy(t)
		body := `{"model":"m","input":[{"type":"compaction_trigger"},{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`
		status, respBody := post(t, h, "/v1/responses", body)
		if status != 400 || !strings.Contains(string(respBody), "final") {
			t.Fatalf("status = %d: %s", status, respBody)
		}
	})
}

func TestOpenAIIncomingItemSubstitutes(t *testing.T) {
	h, fake := newProxy(t)
	blob := mintOpenAIBlob(t, "we were debugging the gateway")

	body := `{"model":"m","input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"old user question"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"old assistant answer"}]},` +
		`{"type":"function_call","name":"lookup","arguments":"{}","call_id":"c1"},` +
		`{"type":"compaction","id":"ci_old","encrypted_content":"` + blob + `","created_by":null},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"live question"}]}]}`

	status, respBody := post(t, h, "/v1/responses", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, respBody)
	}

	fwd := string(fake.ResponsesReqs[0])
	if !strings.Contains(fwd, "old user question") {
		t.Error("pre-compaction USER message must survive verbatim")
	}
	if strings.Contains(fwd, "old assistant answer") || strings.Contains(fwd, `"function_call"`) {
		t.Error("assistant/tool history should be replaced by the summary")
	}
	if !strings.Contains(fwd, "debugging the gateway") {
		t.Error("summary item missing")
	}
	if strings.Contains(fwd, `"type":"compaction"`) {
		t.Error("compaction item leaked upstream (Ollama would 400)")
	}
	if !strings.Contains(fwd, "live question") {
		t.Error("post-compaction tail lost")
	}
}

func TestOpenAIInvalidBlobRejected(t *testing.T) {
	h, _ := newProxy(t)
	blob := mintOpenAIBlob(t, "real")
	tampered := blob[:len(blob)-4] + "AAAA"

	body := `{"model":"m","input":[` +
		`{"type":"compaction","id":"ci_x","encrypted_content":"` + tampered + `"},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`

	status, respBody := post(t, h, "/v1/responses", body)
	if status != 400 {
		t.Fatalf("status = %d, want 400: %s", status, respBody)
	}
	if !strings.Contains(string(respBody), "compaction state") {
		t.Errorf("error should name the compaction state: %s", respBody)
	}
}

func TestOpenAIResponsesCompactEndpoint(t *testing.T) {
	h, fake := newProxy(t)

	body := `{"model":"m","input":[` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"user one"}]},` +
		`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"assistant one"}]},` +
		`{"type":"message","role":"user","content":[{"type":"input_text","text":"user two"}]}]}`

	status, respBody := post(t, h, "/v1/responses/compact", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, respBody)
	}

	resp := mustJSON(t, respBody)
	if resp["object"] != "response.compaction" {
		t.Errorf("object = %v", resp["object"])
	}
	output := resp["output"].([]any)
	// All user messages verbatim, then exactly one compaction item.
	if len(output) != 3 {
		t.Fatalf("output has %d items, want 2 users + 1 compaction: %s", len(output), respBody)
	}
	last := output[2].(map[string]any)
	if last["type"] != "compaction" {
		t.Errorf("last item = %v, want compaction", last["type"])
	}
	if _, err := compactstate.Decode(last["encrypted_content"].(string), testKey); err != nil {
		t.Errorf("blob does not verify: %v", err)
	}
	if strings.Contains(string(respBody), "assistant one") {
		t.Error("assistant content leaked into compact output")
	}
	usage := resp["usage"].(map[string]any)
	if usage["input_tokens"] != float64(fake.SummaryPromptTokens) {
		t.Errorf("usage = %v, want summarizer usage", usage)
	}
}

func TestOpenAIStreamingCompaction(t *testing.T) {
	h, fake := newProxy(t)
	fake.TokenCount = 100
	fake.ResponsesFrames = frameList{
		{"response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_up1","object":"response","status":"in_progress","output":[]}}`},
		{"response.in_progress", `{"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_up1","object":"response","status":"in_progress","output":[]}}`},
		{"response.output_item.added", `{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_s1","type":"message","status":"in_progress","role":"assistant","content":[]}}`},
		{"response.output_text.delta", `{"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_s1","output_index":0,"content_index":0,"delta":"hi"}`},
		{"response.output_text.done", `{"type":"response.output_text.done","sequence_number":4,"item_id":"msg_s1","output_index":0,"content_index":0,"text":"hi"}`},
		{"response.output_item.done", `{"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":{"id":"msg_s1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hi"}]}}`},
		{"response.completed", `{"type":"response.completed","sequence_number":6,"response":{"id":"resp_up1","object":"response","status":"completed","output":[{"id":"msg_s1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":10,"output_tokens":2,"total_tokens":12}}}`},
	}.frames()

	body := strings.Replace(compactParamResponses, `{"model":"m",`, `{"model":"m","stream":true,`, 1)
	status, respBody := post(t, h, "/v1/responses", body)
	if status != 200 {
		t.Fatalf("status = %d: %s", status, respBody)
	}

	frames := parseSSE(t, respBody)

	// Every frame parses through slipspace's event types, and sequence
	// numbers are strictly consecutive from 0.
	wantOrder := []string{
		"response.created",
		"response.output_item.added", "response.output_item.done", // compaction @0
		"response.in_progress",
		"response.output_item.added", // message, shifted to output_index 1
		"response.output_text.delta", "response.output_text.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(frames) != len(wantOrder) {
		t.Fatalf("got %d frames, want %d:\n%s", len(frames), len(wantOrder), respBody)
	}
	for i, f := range frames {
		ev, err := responses.UnmarshalStreamEvent([]byte(f.data))
		if err != nil {
			t.Fatalf("frame %d does not parse: %v\n%s", i, err, f.data)
		}
		if ev.EventType() != wantOrder[i] {
			t.Errorf("frame[%d] = %s, want %s", i, ev.EventType(), wantOrder[i])
		}
		var seq struct {
			SequenceNumber *int `json:"sequence_number"`
		}
		_ = json.Unmarshal([]byte(f.data), &seq)
		if seq.SequenceNumber == nil || *seq.SequenceNumber != i {
			t.Errorf("frame[%d] sequence_number = %v, want %d", i, seq.SequenceNumber, i)
		}
	}

	// The injected pair carries the compaction item; upstream item events
	// shifted to output_index 1.
	var added struct {
		OutputIndex int `json:"output_index"`
		Item        struct {
			Type             string  `json:"type"`
			EncryptedContent *string `json:"encrypted_content"`
		} `json:"item"`
	}
	_ = json.Unmarshal([]byte(frames[1].data), &added)
	if added.Item.Type != "compaction" || added.OutputIndex != 0 {
		t.Errorf("injected added = %+v", added)
	}
	var done struct {
		Item struct {
			EncryptedContent string `json:"encrypted_content"`
		} `json:"item"`
	}
	_ = json.Unmarshal([]byte(frames[2].data), &done)
	if _, err := compactstate.Decode(done.Item.EncryptedContent, testKey); err != nil {
		t.Errorf("streamed blob does not verify: %v", err)
	}
	var shifted struct {
		OutputIndex int `json:"output_index"`
	}
	_ = json.Unmarshal([]byte(frames[4].data), &shifted)
	if shifted.OutputIndex != 1 {
		t.Errorf("upstream item output_index = %d, want 1", shifted.OutputIndex)
	}

	// response.completed: compaction item prepended, id minted, usage kept.
	var completed struct {
		Response struct {
			ID     string `json:"id"`
			Output []struct {
				Type string `json:"type"`
			} `json:"output"`
			Usage struct {
				InputTokens int `json:"input_tokens"`
			} `json:"usage"`
			CompactionUsage map[string]int `json:"compaction_usage"`
		} `json:"response"`
	}
	_ = json.Unmarshal([]byte(frames[8].data), &completed)
	if completed.Response.ID == "resp_up1" {
		t.Error("terminal response id not re-minted")
	}
	if len(completed.Response.Output) != 2 || completed.Response.Output[0].Type != "compaction" {
		t.Errorf("terminal output = %+v", completed.Response.Output)
	}
	if completed.Response.Usage.InputTokens != 10 {
		t.Error("generation usage must pass through untouched")
	}
	if completed.Response.CompactionUsage["output_tokens"] != fake.SummaryCompletionTokens {
		t.Errorf("compaction_usage = %+v", completed.Response.CompactionUsage)
	}
}

func TestOpenAITruncatedStreamSynthesizesFailure(t *testing.T) {
	h, fake := newProxy(t)
	fake.ResponsesFrames = frameList{
		{"response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_up1","object":"response","status":"in_progress","output":[]}}`},
	}.frames()

	body := `{"model":"m","stream":true,"input":"hello"}`
	status, respBody := post(t, h, "/v1/responses", body)
	if status != 200 {
		t.Fatalf("status = %d", status)
	}
	frames := parseSSE(t, respBody)
	last := frames[len(frames)-1]
	var peek struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal([]byte(last.data), &peek)
	if peek.Type != "response.failed" {
		t.Errorf("last frame = %s, want response.failed", last.data)
	}
}

func TestOpenAIStringInputPassthrough(t *testing.T) {
	h, fake := newProxy(t)

	body := `{"model":"m","input":"just a string"}`
	status, _ := post(t, h, "/v1/responses", body)
	if status != 200 {
		t.Fatalf("status = %d", status)
	}
	if got := string(fake.ResponsesReqs[0]); got != body {
		t.Errorf("string input rewritten: %s", got)
	}
}

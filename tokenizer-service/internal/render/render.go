// Package render turns extracted prompt fragments into the text a model would
// actually receive, using Ollama's own rendering machinery.
//
// Three tiers, in order of fidelity:
//
//  1. the model's Go-native renderer  (gemma4, glimmer, qwen3-coder, ...)
//  2. the model's Go chat template    (qwen2.5-coder, gpt-oss, ...)
//  3. plain concatenation of the extracted text
//
// A lower tier is used whenever the one above is unavailable or errors, so a
// count is always produced. Rendering is best-effort by design: the spec makes
// exact parity with inference a non-goal.
package render

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/extract"
)

// Tier names the rendering path that produced a Result.
type Tier string

const (
	TierRenderer Tier = "renderer"
	TierTemplate Tier = "template"
	TierConcat   Tier = "concat"
)

// Model carries the rendering metadata for one Ollama model.
type Model struct {
	Renderer string // RENDERER directive, "" if the model uses its template
	Template string

	// Thinking reports the model's thinking capability. Ollama enables
	// thinking by default on such models, which adds a block to the rendered
	// prompt, so ignoring it undercounts every request.
	Thinking bool

	// System is the model's own default system prompt. Ollama prepends it
	// when a request supplies no system message of its own, and for models
	// like qwen2.5-coder it is substantial.
	System string
}

// Result is rendered prompt text plus how it was produced.
type Result struct {
	Text string
	Tier Tier

	// Media counts non-text content blocks seen while normalizing. Their
	// model-specific token cost is not accounted for; this exists so the
	// omission is visible in logs.
	Media int

	// Think records whether thinking was enabled for this render.
	Think bool

	// Fallback records why a higher tier was skipped, for logging. It never
	// contains prompt content.
	Fallback string
}

// Render produces the text to tokenize for req against model m.
func Render(req *extract.Request, m Model) Result {
	n := normalize(req)
	n.applyDefaultSystem(m.System)
	think := thinkValue(req.Think, m.Thinking)

	if m.Renderer != "" {
		text, err := renderers.RenderWithRenderer(m.Renderer, n.Messages, n.Tools, think)
		if err == nil {
			return Result{Text: text + n.extra(), Tier: TierRenderer, Media: n.Media, Think: think.Bool()}
		}
		// Unknown or failing renderer: drop a tier rather than fail the count.
		return renderTemplate(m, n, think, "renderer "+m.Renderer+": "+err.Error())
	}

	return renderTemplate(m, n, think, "")
}

// thinkValue mirrors Ollama's rule: an explicit think setting wins, otherwise
// thinking-capable models default to on and others to unset.
func thinkValue(raw json.RawMessage, capable bool) *api.ThinkValue {
	if len(raw) > 0 {
		var v any
		if err := json.Unmarshal(raw, &v); err == nil {
			switch v.(type) {
			case bool, string:
				return &api.ThinkValue{Value: v}
			}
		}
	}
	if capable {
		return &api.ThinkValue{Value: true}
	}
	return nil
}

func renderTemplate(m Model, n normalized, think *api.ThinkValue, fallback string) Result {
	if strings.TrimSpace(m.Template) != "" {
		tmpl, err := template.Parse(m.Template)
		if err == nil {
			values := template.Values{
				Messages:   n.Messages,
				Tools:      n.Tools,
				Think:      think.Bool(),
				IsThinkSet: think != nil,
			}
			if think != nil {
				values.ThinkLevel = think.String()
			}

			var b strings.Builder
			if err = tmpl.Execute(&b, values); err == nil {
				return Result{
					Text:  b.String() + n.extra(),
					Tier:  TierTemplate,
					Media: n.Media,
					Think: think.Bool(),
					// Ollama renders Jinja chat templates inside llama.cpp
					// rather than in Go, so a Jinja template parses here only
					// by accident; see the Fallback set below when it does not.
					Fallback: fallback,
				}
			}
		}
		if fallback == "" {
			fallback = "template: " + err.Error()
		}
	}

	return Result{Text: n.concat(), Tier: TierConcat, Media: n.Media, Think: think.Bool(), Fallback: fallback}
}

// normalized is the request expressed in Ollama's own types.
type normalized struct {
	Messages []api.Message
	Tools    api.Tools
	Media    int

	// toolsRaw holds the tools JSON when it could not be parsed into
	// api.Tools, so its text still contributes to the count.
	toolsRaw string
}

// extra returns content that could not be handed to a renderer or template and
// so is appended to the rendered text.
func (n normalized) extra() string {
	if n.toolsRaw == "" {
		return ""
	}
	return "\n" + n.toolsRaw
}

func (n normalized) concat() string {
	var b strings.Builder
	for _, m := range n.Messages {
		if m.Role != "" {
			b.WriteString(m.Role)
			b.WriteString(": ")
		}
		b.WriteString(m.Content)
		b.WriteString("\n")
	}
	if len(n.Tools) > 0 {
		b.WriteString(n.Tools.String())
	}
	b.WriteString(n.extra())
	return b.String()
}

func normalize(req *extract.Request) normalized {
	var n normalized

	// System and instructions become leading system messages; both the
	// renderers and template.Execute pull system content out of the message
	// list themselves.
	for _, raw := range []json.RawMessage{req.System, req.Instructions} {
		if text, media := textOf(raw); text != "" || media > 0 {
			n.Media += media
			if text != "" {
				n.Messages = append(n.Messages, api.Message{Role: "system", Content: text})
			}
		}
	}

	n.appendMessages(req.Messages)

	// "input" is either a message list (OpenAI Responses shape) or a bare
	// string. Try it as messages first, fall back to treating it as a prompt.
	if req.Input != nil {
		before := len(n.Messages)
		n.appendMessages(req.Input)
		if len(n.Messages) == before {
			n.appendUser(req.Input)
		}
	}

	n.appendUser(req.Prompt)

	if req.Tools != nil {
		if err := json.Unmarshal(req.Tools, &n.Tools); err != nil || len(n.Tools) == 0 {
			n.Tools = nil
			n.toolsRaw = string(req.Tools)
		}
	}

	return n
}

// applyDefaultSystem prepends the model's own system prompt when the request
// did not provide one, matching Ollama's behavior.
func (n *normalized) applyDefaultSystem(system string) {
	if system == "" {
		return
	}
	if len(n.Messages) > 0 && n.Messages[0].Role == "system" {
		return
	}
	n.Messages = append([]api.Message{{Role: "system", Content: system}}, n.Messages...)
}

func (n *normalized) appendUser(raw json.RawMessage) {
	if text, media := textOf(raw); text != "" || media > 0 {
		n.Media += media
		if text != "" {
			n.Messages = append(n.Messages, api.Message{Role: "user", Content: text})
		}
	}
}

// appendMessages decodes a message array leniently. api.Message.Content is a
// string, but Anthropic- and OpenAI-Responses-shaped requests send content as
// an array of typed blocks, so each message is decoded field by field.
func (n *normalized) appendMessages(raw json.RawMessage) {
	if raw == nil {
		return
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		// Not a message array at all — count whatever text it holds.
		if text, media := textOf(raw); text != "" || media > 0 {
			n.Media += media
			if text != "" {
				n.Messages = append(n.Messages, api.Message{Role: "user", Content: text})
			}
		}
		return
	}

	for _, item := range items {
		msg := api.Message{Role: "user"}

		var role string
		if err := json.Unmarshal(item["role"], &role); err == nil && role != "" {
			msg.Role = strings.ToLower(role)
		}

		text, media := textOf(item["content"])
		n.Media += media
		msg.Content = text

		// OpenAI Responses items carry their payload outside "content":
		// function_call in "arguments", function_call_output in "output".
		// Both reach the model, so both must be counted.
		for _, key := range []string{"arguments", "output"} {
			if raw, ok := item[key]; ok {
				if extra, m := textOf(raw); extra != "" {
					n.Media += m
					msg.Content = strings.TrimSpace(msg.Content + "\n" + extra)
				}
			}
		}

		if raw, ok := item["tool_calls"]; ok {
			if err := json.Unmarshal(raw, &msg.ToolCalls); err != nil {
				// Unparseable tool calls still cost tokens.
				msg.Content = strings.TrimSpace(msg.Content + "\n" + string(raw))
			}
		}
		var toolName string
		if err := json.Unmarshal(item["name"], &toolName); err == nil {
			msg.ToolName = toolName
		}
		if err := json.Unmarshal(item["tool_name"], &toolName); err == nil && toolName != "" {
			msg.ToolName = toolName
		}

		// Native renderers consume both fields. In particular Glimmer resolves
		// tool-result labels through ToolCallID and renders prior thinking.
		_ = json.Unmarshal(item["tool_call_id"], &msg.ToolCallID)
		_ = json.Unmarshal(item["thinking"], &msg.Thinking)

		if msg.Content == "" && msg.Thinking == "" && len(msg.ToolCalls) == 0 && msg.ToolName == "" && msg.ToolCallID == "" {
			continue
		}
		n.Messages = append(n.Messages, msg)
	}
}

// mediaBlockTypes are content-block types whose token cost is model-specific
// and deliberately not estimated here.
var mediaBlockTypes = []string{"image", "audio", "video", "input_image", "input_audio", "image_url"}

// textOf reduces an arbitrary JSON value to its human-readable text, returning
// the number of non-text media blocks it skipped.
func textOf(raw json.RawMessage) (string, int) {
	if len(raw) == 0 {
		return "", 0
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, 0
	}

	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var parts []string
		var media int
		for _, b := range blocks {
			text, m := textOf(b)
			media += m
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n"), media
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		var kind string
		_ = json.Unmarshal(obj["type"], &kind)
		for _, m := range mediaBlockTypes {
			if strings.Contains(kind, m) {
				return "", 1
			}
		}
		for _, key := range []string{"text", "content", "input", "arguments"} {
			if v, ok := obj[key]; ok {
				if text, media := textOf(v); text != "" {
					return text, media
				}
			}
		}
		// An object with no recognized text field: count its JSON, since it
		// was in a prompt-bearing position and will reach the model somehow.
		return string(raw), 0
	}

	// Numbers, booleans, null: their literal form is what a template emits.
	return strings.Trim(string(raw), `"`), 0
}

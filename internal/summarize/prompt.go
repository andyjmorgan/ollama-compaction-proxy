package summarize

// DefaultInstructions is the summarization system prompt used when the caller
// supplies none. Per the Anthropic contract, caller-provided instructions
// REPLACE this entirely rather than supplementing it.
const DefaultInstructions = `You are compacting a long conversation so it can continue in less context.
Write a summary that will REPLACE the conversation history entirely — the
model continuing the conversation will see only your summary and the most
recent turns. Capture, in order:

1. Context and goal: who the user is working with and what they are trying to do.
2. Key decisions and facts established so far, including exact names, values,
   paths, and identifiers that later turns may refer back to.
3. Important tool results worth retaining, condensed to their conclusions.
4. Current state: what has been completed and what is in progress.
5. The pending task: what the conversation is about to do next.

Write plainly and densely. Do not address the user, do not mention that this
is a summary, and do not add commentary.`

// SummaryWrapper frames the summary text when it is injected back into a
// conversation as a synthetic user turn.
const summaryWrapperPrefix = "<conversation_summary>\n"
const summaryWrapperSuffix = "\n</conversation_summary>\n" +
	"[Everything above this summary was compacted from the earlier conversation.\n" +
	"Continue the conversation seamlessly; do not mention the compaction.]"

// WrapSummary produces the synthetic turn text carrying summary.
func WrapSummary(summary string) string {
	return summaryWrapperPrefix + summary + summaryWrapperSuffix
}

package summarize

import "testing"

// The precedence chain is the whole point of the override: a caller's own
// instructions must still replace everything, per the Anthropic contract.
func TestResolveInstructions(t *testing.T) {
	for _, tc := range []struct {
		name   string
		house  string
		caller string
		want   string
	}{
		{"built-in when nothing is set", "", "", DefaultInstructions},
		{"house prompt replaces built-in", "HOUSE", "", "HOUSE"},
		{"caller replaces house prompt", "HOUSE", "CALLER", "CALLER"},
		{"blank house prompt falls back to built-in", "   \n", "", DefaultInstructions},
		{"blank caller falls back to house prompt", "HOUSE", "  ", "HOUSE"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveInstructions(&Summarizer{Instructions: tc.house}, tc.caller); got != tc.want {
				t.Fatalf("got %.40q, want %.40q", got, tc.want)
			}
		})
	}
}

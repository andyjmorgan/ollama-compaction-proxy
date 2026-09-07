package upstream

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Frame is one server-sent event: an optional event name and its data payload.
// Multi-line data is joined with newlines per the SSE specification.
type Frame struct {
	Event string
	Data  []byte
}

// IsDone reports whether the frame is the OpenAI-style "[DONE]" sentinel.
func (f Frame) IsDone() bool {
	return bytes.Equal(bytes.TrimSpace(f.Data), []byte("[DONE]"))
}

// maxLineBytes bounds a single SSE line; generous because a whole tool-call
// argument object can arrive in one data line.
const maxLineBytes = 8 << 20

// FrameReader reads SSE frames from an upstream response body.
type FrameReader struct {
	sc *bufio.Scanner
}

// NewFrameReader wraps r for frame-at-a-time reading.
func NewFrameReader(r io.Reader) *FrameReader {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), maxLineBytes)
	return &FrameReader{sc: sc}
}

// Next returns the next frame, or io.EOF when the stream ends cleanly at a
// frame boundary. Comment lines (":heartbeat") are skipped; id/retry lines are
// ignored (neither dialect uses them).
func (r *FrameReader) Next() (Frame, error) {
	var frame Frame
	var data [][]byte
	seen := false

	for r.sc.Scan() {
		line := r.sc.Bytes()
		switch {
		case len(bytes.TrimSpace(line)) == 0:
			if seen {
				frame.Data = bytes.Join(data, []byte("\n"))
				return frame, nil
			}
			// Leading blank line: keep scanning.
		case line[0] == ':':
			// Comment / heartbeat.
		case bytes.HasPrefix(line, []byte("event:")):
			frame.Event = strings.TrimSpace(string(line[len("event:"):]))
			seen = true
		case bytes.HasPrefix(line, []byte("data:")):
			d := bytes.TrimPrefix(line, []byte("data:"))
			d = bytes.TrimPrefix(d, []byte(" "))
			// Copy: the scanner reuses its buffer.
			data = append(data, append([]byte(nil), d...))
			seen = true
		}
	}

	if err := r.sc.Err(); err != nil {
		return Frame{}, err
	}
	if seen {
		// Stream ended mid-frame without the trailing blank line; deliver
		// what we have so terminal-event bookkeeping sees it.
		frame.Data = bytes.Join(data, []byte("\n"))
		return frame, nil
	}
	return Frame{}, io.EOF
}

// FrameWriter emits SSE frames to a client, flushing per frame.
//
// The frame layout ("event: X\ndata: {json}\n\n") mirrors what both provider
// dialects and Ollama's compat endpoints emit.
type FrameWriter struct {
	w       io.Writer
	flusher http.Flusher
}

// NewFrameWriter prepares w for SSE output, setting the response headers.
// Call before writing any body bytes.
func NewFrameWriter(w http.ResponseWriter) *FrameWriter {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	return &FrameWriter{w: w, flusher: flusher}
}

// Write emits one frame. An empty event name writes a bare data frame, which
// is how OpenAI-dialect streams arrive.
func (fw *FrameWriter) Write(f Frame) error {
	var b bytes.Buffer
	if f.Event != "" {
		b.WriteString("event: ")
		b.WriteString(f.Event)
		b.WriteString("\n")
	}
	b.WriteString("data: ")
	b.Write(f.Data)
	b.WriteString("\n\n")

	if _, err := fw.w.Write(b.Bytes()); err != nil {
		return fmt.Errorf("write frame: %w", err)
	}
	if fw.flusher != nil {
		fw.flusher.Flush()
	}
	return nil
}

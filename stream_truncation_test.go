package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// An upstream that drops mid-generation must not look like a short but
// successful answer. Each decoder tracks whether it saw a terminal event and
// reports a truncation through Close; the encoders then render that as the
// target protocol's error frame.

func TestOpenAIChatStreamDecoderReportsTruncation(t *testing.T) {
	cases := []struct {
		name       string
		events     []string
		wantClosed bool
	}{
		{
			name:       "dropped after content",
			events:     []string{`{"id":"c1","object":"chat.completion.chunk","created":1,"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`},
			wantClosed: true,
		},
		{
			name: "finish_reason ends the stream",
			events: []string{
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
				`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			},
			wantClosed: false,
		},
		{
			name: "done sentinel ends the stream",
			events: []string{
				`{"id":"c1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
				`[DONE]`,
			},
			wantClosed: false,
		},
		{
			name:       "never started",
			events:     nil,
			wantClosed: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decoder, err := NewOpenAIChatAdapter().NewStreamDecoder(StreamDecodeOptions{})
			if err != nil {
				t.Fatalf("NewStreamDecoder() error = %v", err)
			}
			for _, event := range tc.events {
				if _, err := decoder.Decode(RawStreamEvent{Data: []byte(event)}); err != nil {
					t.Fatalf("Decode(%s) error = %v", event, err)
				}
			}
			assertDecoderClose(t, decoder, tc.wantClosed)
		})
	}
}

func TestAnthropicStreamDecoderReportsTruncation(t *testing.T) {
	start := `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`
	delta := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`
	messageDelta := `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`
	messageStop := `{"type":"message_stop"}`

	cases := []struct {
		name       string
		events     []string
		wantClosed bool
	}{
		{name: "dropped after content", events: []string{start, delta}, wantClosed: true},
		{name: "message_stop ends the stream", events: []string{start, delta, messageDelta, messageStop}, wantClosed: false},
		{name: "stop_reason alone ends the stream", events: []string{start, delta, messageDelta}, wantClosed: false},
		{name: "error event is a deliberate end", events: []string{start, `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`}, wantClosed: false},
		{name: "never started", events: nil, wantClosed: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decoder, err := NewAnthropicMessagesAdapter().NewStreamDecoder(StreamDecodeOptions{})
			if err != nil {
				t.Fatalf("NewStreamDecoder() error = %v", err)
			}
			for _, event := range tc.events {
				if _, err := decoder.Decode(RawStreamEvent{Data: []byte(event)}); err != nil {
					t.Fatalf("Decode(%s) error = %v", event, err)
				}
			}
			assertDecoderClose(t, decoder, tc.wantClosed)
		})
	}
}

func TestOpenAIResponsesStreamDecoderReportsTruncation(t *testing.T) {
	created := `{"type":"response.created","sequence_number":1,"response":{"id":"resp_1","status":"in_progress","model":"gpt-5"}}`
	delta := `{"type":"response.output_text.delta","sequence_number":2,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"hi"}`
	completed := `{"type":"response.completed","sequence_number":3,"response":{"id":"resp_1","status":"completed","output":[]}}`

	cases := []struct {
		name       string
		events     []string
		wantClosed bool
	}{
		{name: "dropped after content", events: []string{created, delta}, wantClosed: true},
		{name: "completed ends the stream", events: []string{created, delta, completed}, wantClosed: false},
		{name: "never started", events: nil, wantClosed: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
			if err != nil {
				t.Fatalf("NewStreamDecoder() error = %v", err)
			}
			for _, event := range tc.events {
				if _, err := decoder.Decode(RawStreamEvent{Data: []byte(event)}); err != nil {
					t.Fatalf("Decode(%s) error = %v", event, err)
				}
			}
			assertDecoderClose(t, decoder, tc.wantClosed)
		})
	}
}

func assertDecoderClose(t *testing.T, decoder StreamDecoder, wantTruncated bool) {
	t.Helper()
	parts, err := decoder.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !wantTruncated {
		if len(parts) != 0 {
			t.Fatalf("Close() = %+v, want nothing for a complete stream", parts)
		}
		return
	}
	if len(parts) != 1 || parts[0].Type != StreamError {
		t.Fatalf("Close() = %+v, want one StreamError part", parts)
	}
	if errorMessage(parts[0].Error) != ErrStreamTruncated.Error() {
		t.Fatalf("error = %#v", parts[0].Error)
	}
}

// A Claude upstream that drops mid-stream must reach a chat client as an error,
// not as a finish_reason the client would read as a complete answer.
func TestTruncatedAnthropicUpstreamReachesChatClientAsError(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("no anthropic-to-chat bridge")
	}

	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I will now"}}`,
	}
	var parts []StreamPart
	for _, event := range events {
		decoded, err := decoder.Decode(RawStreamEvent{Data: []byte(event)})
		if err != nil {
			t.Fatalf("Decode() error = %v", err)
		}
		parts = append(parts, decoded...)
	}

	closed, err := decoder.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	parts = append(parts, closed...)

	var out []RawStreamEvent
	for _, part := range parts {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(%s) error = %v", part.Type, err)
		}
		out = append(out, encoded...)
	}
	closedEvents, err := encoder.Close()
	if err != nil {
		t.Fatalf("encoder Close() error = %v", err)
	}
	out = append(out, closedEvents...)

	var sawError bool
	for _, event := range out {
		var payload map[string]any
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			continue
		}
		inner, ok := payload["error"].(map[string]any)
		if !ok {
			continue
		}
		if message, _ := inner["message"].(string); strings.Contains(message, "without a terminal event") {
			sawError = true
		}
	}
	if !sawError {
		t.Fatalf("client saw no truncation error: %+v", out)
	}
}

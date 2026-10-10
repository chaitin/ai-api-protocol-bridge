package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// A raw frame is an upstream event this package does not model. It is not model
// output, so it must not become assistant content — which is what rendering it
// with fmt.Sprint did, putting Go syntax like "{ 0 <nil> }" in the answer.
func TestRawStreamTextIsJSON(t *testing.T) {
	type event struct {
		Type  string `json:"type"`
		Index int    `json:"index"`
	}
	got := rawStreamText(event{Type: "some_event", Index: 3})
	if got != `{"type":"some_event","index":3}` {
		t.Fatalf("rawStreamText() = %q", got)
	}
	if strings.Contains(got, "<nil>") || strings.Contains(got, "{ ") {
		t.Fatalf("rawStreamText() = %q, want JSON", got)
	}
	if got := rawStreamText(`{"a":1}`); got != `{"a":1}` {
		t.Fatalf("rawStreamText() = %q", got)
	}
	if got := rawStreamText(nil); got != "" {
		t.Fatalf("rawStreamText(nil) = %q", got)
	}
}

// The frame has nowhere to go on the wire. Putting it in content corrupted the
// answer, and inventing a field or an event type for it would put something on
// the wire that other implementations have never seen — which is exactly what a
// non-official upstream or downstream tends to reject. So it is dropped, and the
// optional callback is how a host that wants it gets it.
func TestRawStreamFramesGoToTheCallbackNotTheWire(t *testing.T) {
	cases := []struct {
		name     string
		encoder  func(onWarning func(Warning)) (StreamEncoder, error)
		protocol Protocol
	}{
		{
			name: "chat completions",
			encoder: func(onWarning func(Warning)) (StreamEncoder, error) {
				return NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4", Created: 1, OnWarning: onWarning})
			},
			protocol: ProtocolOpenAIChat,
		},
		{
			name: "openai responses",
			encoder: func(onWarning func(Warning)) (StreamEncoder, error) {
				return NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4", OnWarning: onWarning})
			},
			protocol: ProtocolOpenAIResponses,
		},
		{
			name: "anthropic messages",
			encoder: func(onWarning func(Warning)) (StreamEncoder, error) {
				return NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet-4", OnWarning: onWarning})
			},
			protocol: ProtocolAnthropicMessages,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var warnings []Warning
			encoder, err := tc.encoder(func(warning Warning) { warnings = append(warnings, warning) })
			if err != nil {
				t.Fatalf("NewStreamEncoder() error = %v", err)
			}

			events, err := encoder.Encode(StreamPart{Type: StreamRaw, RawValue: map[string]any{"type": "some_event"}})
			if err != nil {
				t.Fatalf("Encode() error = %v", err)
			}
			if len(events) != 0 {
				t.Fatalf("events = %s, want nothing on the wire", events[0].Data)
			}

			if len(warnings) != 1 {
				t.Fatalf("warnings = %+v", warnings)
			}
			warning := warnings[0]
			if warning.Code != LossUnsupportedStreamEvent || warning.Severity != SeverityInfo {
				t.Fatalf("warning = %+v", warning)
			}
			if warning.To != tc.protocol {
				t.Fatalf("warning.To = %q, want %q", warning.To, tc.protocol)
			}
			if warning.Detail != `{"type":"some_event"}` {
				t.Fatalf("warning.Detail = %q, want the frame as JSON", warning.Detail)
			}
			if strings.Contains(warning.Detail, "<nil>") {
				t.Fatalf("Go syntax in the detail: %q", warning.Detail)
			}
		})
	}
}

// The callback is optional, and not registering one is what every host did before
// it existed. That path has to stay quiet rather than panic.
func TestRawStreamFramesAreDroppedWithoutACallback(t *testing.T) {
	for _, protocol := range []Protocol{ProtocolOpenAIChat, ProtocolOpenAIResponses, ProtocolAnthropicMessages} {
		t.Run(string(protocol), func(t *testing.T) {
			encoder, err := adapterForProtocol(t, protocol).NewStreamEncoder(StreamEncodeOptions{Model: "m"})
			if err != nil {
				t.Fatalf("NewStreamEncoder() error = %v", err)
			}
			events, err := encoder.Encode(StreamPart{Type: StreamRaw, RawValue: map[string]any{"type": "some_event"}})
			if err != nil {
				t.Fatalf("Encode() error = %v", err)
			}
			if len(events) != 0 {
				t.Fatalf("events = %+v", events)
			}
		})
	}
}

// The bridges used to assemble their inner encoder out of the options struct
// field by field. That silently drops any option nobody remembers to add: Created
// was lost by five of the six, and OnWarning would have been lost by all of them,
// which would have made the callback above work through an adapter and quietly do
// nothing through a bridge — the path production actually takes.
func TestEveryBridgePropagatesStreamEncodeOptions(t *testing.T) {
	protocols := []Protocol{ProtocolOpenAIChat, ProtocolOpenAIResponses, ProtocolAnthropicMessages}

	for _, inbound := range protocols {
		for _, upstream := range protocols {
			if inbound == upstream {
				continue
			}
			t.Run(string(inbound)+"_to_"+string(upstream), func(t *testing.T) {
				bridge, ok := NewCrossFamilyBridgeForProtocol(inbound, upstream)
				if !ok {
					t.Fatalf("no bridge for %s to %s", inbound, upstream)
				}

				var warnings []Warning
				encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{
					Model:     "upstream-model",
					Created:   12345,
					OnWarning: func(warning Warning) { warnings = append(warnings, warning) },
				})
				if err != nil {
					t.Fatalf("NewStreamEncoder() error = %v", err)
				}

				events, err := encoder.Encode(StreamPart{Type: StreamRaw, RawValue: map[string]any{"type": "some_event"}})
				if err != nil {
					t.Fatalf("Encode() error = %v", err)
				}
				if len(warnings) != 1 || warnings[0].Code != LossUnsupportedStreamEvent {
					t.Fatalf("warnings = %+v, want the callback to reach every bridge", warnings)
				}
				if len(events) != 0 {
					t.Fatalf("events = %+v, want nothing on the wire", events)
				}

				// Created only has a wire field in chat completions, so only the
				// cells whose client speaks chat can be checked for it.
				if inbound != ProtocolOpenAIChat {
					return
				}

				start, err := encoder.Encode(StreamPart{Type: StreamStart, ID: "chatcmpl-1"})
				if err != nil {
					t.Fatalf("Encode(StreamStart) error = %v", err)
				}
				if len(start) != 1 {
					t.Fatalf("start events = %+v", start)
				}
				var chunk openAIChatStreamChunk
				if err := json.Unmarshal(start[0].Data, &chunk); err != nil {
					t.Fatalf("Unmarshal() error = %v", err)
				}
				if chunk.Created != 12345 {
					t.Fatalf("created = %d, want the requested 12345", chunk.Created)
				}
			})
		}
	}
}

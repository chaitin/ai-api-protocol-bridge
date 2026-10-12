package protocolbridge

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// Golden snapshots pin the bytes each bridge puts on the wire.
//
// Hand-written assertions say what someone thought to check; a snapshot says
// what the encoder actually produces, so an unintended change to a field name,
// a nesting level, or the order of stream frames shows up as a diff instead of
// passing because no assertion happened to look there. The cost is that a
// snapshot also accepts a mistake that was there when it was written, which is
// why the cases below are built from the traffic-shaped fixtures and why the
// interesting invariants still have their own tests.
//
// Run `go test ./... -update` to rewrite the files after an intended change,
// then read the diff.

var updateGolden = flag.Bool("update", false, "rewrite the golden files under testdata/golden")

// goldenVolatile rewrites values that differ between runs, so a snapshot fails
// on a semantic change rather than on a clock. A raw epoch that reaches a
// snapshot is normalised to zero; identifiers the fixtures already fix are left
// alone on purpose, because a change to how an id is derived is a change worth
// seeing.
var goldenVolatile = []struct {
	pattern     *regexp.Regexp
	replacement string
}{
	{regexp.MustCompile(`"created"\s*:\s*\d+`), `"created": 0`},
	{regexp.MustCompile(`"sequence_number"\s*:\s*\d+`), `"sequence_number": 0`},
}

func normalizeGolden(t *testing.T, raw []byte) string {
	t.Helper()

	// Round-tripping through any sorts object keys and drops the encoder's
	// field-order accidents, so the snapshot reflects content rather than
	// struct layout. Struct field order is still visible, because the second
	// marshal re-encodes what the first produced.
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", raw, err)
	}
	sorted, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatalf("MarshalIndent() error = %v", err)
	}

	text := string(sorted)
	for _, rule := range goldenVolatile {
		text = rule.pattern.ReplaceAllString(text, rule.replacement)
	}
	return text + "\n"
}

func assertGolden(t *testing.T, name string, raw []byte) {
	t.Helper()

	got := normalizeGolden(t, raw)
	path := filepath.Join("testdata", "golden", name+".json")

	if *updateGolden {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("MkdirAll(%s) error = %v", filepath.Dir(path), err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatalf("WriteFile(%s) error = %v", path, err)
		}
		return
	}

	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v; run go test ./... -update to create it", path, err)
	}
	if string(want) == got {
		return
	}
	t.Fatalf("%s does not match the snapshot.\n--- want ---\n%s\n--- got ---\n%s\nrun go test ./... -update if the change is intended",
		path, strings.TrimRight(string(want), "\n"), strings.TrimRight(got, "\n"))
}

// goldenRequestCases covers every bridge cell's request encoding. One fixture
// per inbound protocol is enough: the point is the shape each upstream
// receives, and the fixtures carry the fields an agent actually sends.
var goldenRequestCases = []struct {
	name     string
	inbound  Protocol
	upstream Protocol
	body     string
}{
	{"anthropic_to_chat_request", ProtocolAnthropicMessages, ProtocolOpenAIChat, claudeCodeRequest},
	{"anthropic_to_responses_request", ProtocolAnthropicMessages, ProtocolOpenAIResponses, claudeCodeRequest},
	{"responses_to_chat_request", ProtocolOpenAIResponses, ProtocolOpenAIChat, codexRequest},
	{"responses_to_anthropic_request", ProtocolOpenAIResponses, ProtocolAnthropicMessages, codexRequest},
	{"chat_to_anthropic_request", ProtocolOpenAIChat, ProtocolAnthropicMessages, chatClientRequest},
	{"chat_to_responses_request", ProtocolOpenAIChat, ProtocolOpenAIResponses, chatClientRequest},
}

func TestGoldenRequestEncoding(t *testing.T) {
	for _, tc := range goldenRequestCases {
		t.Run(tc.name, func(t *testing.T) {
			bridge, ok := NewCrossFamilyBridgeForProtocol(tc.inbound, tc.upstream)
			if !ok {
				t.Fatalf("no bridge for %s to %s", tc.inbound, tc.upstream)
			}
			req, err := adapterForProtocol(t, tc.inbound).DecodeRequest([]byte(tc.body))
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "upstream-model"})
			if err != nil {
				t.Fatalf("EncodeUpstreamRequest() error = %v", err)
			}
			assertGolden(t, tc.name, raw)
		})
	}
}

func adapterForProtocol(t *testing.T, protocol Protocol) Adapter {
	t.Helper()
	switch protocol {
	case ProtocolOpenAIChat:
		return NewOpenAIChatAdapter()
	case ProtocolOpenAIResponses:
		return NewOpenAIResponsesAdapter()
	case ProtocolAnthropicMessages:
		return NewAnthropicMessagesAdapter()
	}
	t.Fatalf("no adapter for %q", protocol)
	return nil
}

// The stream snapshots exist for the two directions where the frame order is
// load-bearing rather than incidental: a Responses client needs the trailing
// usage folded into response.completed, and a chat client needs it split out
// after finish_reason. Both are held to that by their own tests; the snapshot
// is what notices if a frame appears, disappears, or changes shape.
// goldenChatUpstreamChunks is a chat upstream stream in the order OpenAI sends
// it: content, then finish_reason, then a usage-only chunk, then the sentinel.
var goldenChatUpstreamChunks = []string{
	`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"upstream-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
	`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"upstream-model","choices":[{"index":0,"delta":{"content":"README.md"},"finish_reason":null}]}`,
	`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"upstream-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"upstream-model","choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":8,"total_tokens":1208}}`,
	`[DONE]`,
}

// The name of each case is the bridge cell, and the snapshot is what the client
// receives: a chat client over a Responses upstream, and a Responses client over
// a chat upstream.
func TestGoldenChatClientOverResponsesUpstreamStream(t *testing.T) {
	events := relayResponsesStreamToChat(t, responsesTextStream)
	assertGolden(t, "chat_to_responses_stream", renderStreamEvents(t, events))
}

func TestGoldenResponsesClientOverChatUpstreamStream(t *testing.T) {
	events := relayChatStreamToResponses(t, goldenChatUpstreamChunks)
	assertGolden(t, "responses_to_chat_stream", renderStreamEvents(t, events))
}

// renderStreamEvents flattens the emitted frames into one JSON document so a
// snapshot reads as the sequence the client receives.
func renderStreamEvents(t *testing.T, events []RawStreamEvent) []byte {
	t.Helper()

	frames := make([]map[string]any, 0, len(events))
	for _, event := range events {
		frame := map[string]any{}
		if event.Event != "" {
			frame["event"] = event.Event
		}
		// [DONE] is not JSON; keep it verbatim so the snapshot shows it.
		var payload any
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			frame["data"] = string(event.Data)
		} else {
			frame["data"] = payload
		}
		frames = append(frames, frame)
	}
	raw, err := json.Marshal(frames)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	return raw
}

package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNewCrossFamilyBridgeForProtocolSelectsTheAskedForProtocol(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(anthropic_messages, openai_chat) ok = false, want true")
	}
	if got := bridge.UpstreamProtocol(); got != ProtocolOpenAIChat {
		t.Fatalf("UpstreamProtocol() = %q, want %q", got, ProtocolOpenAIChat)
	}
	if got := bridge.InboundProtocol(); got != ProtocolAnthropicMessages {
		t.Fatalf("InboundProtocol() = %q, want %q", got, ProtocolAnthropicMessages)
	}

	// The family-based lookup is what this constructor exists to work around:
	// an Anthropic inbound asking for the OpenAI family gets Responses.
	family, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, FamilyOpenAI)
	if !ok {
		t.Fatal("NewCrossFamilyBridge(anthropic_messages, openai) ok = false, want true")
	}
	if got := family.UpstreamProtocol(); got != ProtocolOpenAIResponses {
		t.Fatalf("family lookup UpstreamProtocol() = %q, want %q; the protocol-precise constructor would be pointless otherwise", got, ProtocolOpenAIResponses)
	}
}

func TestNewCrossFamilyBridgeForProtocolDefersForExistingPairs(t *testing.T) {
	cases := []struct {
		name     string
		inbound  Protocol
		upstream Protocol
		want     bool
		wantUp   Protocol
	}{
		{"anthropic to responses", ProtocolAnthropicMessages, ProtocolOpenAIResponses, true, ProtocolOpenAIResponses},
		{"chat to anthropic", ProtocolOpenAIChat, ProtocolAnthropicMessages, true, ProtocolAnthropicMessages},
		{"responses to anthropic", ProtocolOpenAIResponses, ProtocolAnthropicMessages, true, ProtocolAnthropicMessages},
		// Chat and Responses are one family but two wire formats, and no
		// fallback resolves them: a family lookup returns nothing, and the
		// caller's default of "reuse the inbound adapter" would post chat JSON
		// at a Responses endpoint. So this pair needs a bridge like any other,
		// even though both halves are just the two adapters.
		{"chat to responses", ProtocolOpenAIChat, ProtocolOpenAIResponses, true, ProtocolOpenAIResponses},
		{"responses to chat", ProtocolOpenAIResponses, ProtocolOpenAIChat, true, ProtocolOpenAIChat},
		// Only the identity needs no bridge: it is re-encoded through the one
		// shared adapter.
		{"identical", ProtocolOpenAIChat, ProtocolOpenAIChat, false, ""},
		{"identical responses", ProtocolOpenAIResponses, ProtocolOpenAIResponses, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bridge, ok := NewCrossFamilyBridgeForProtocol(tc.inbound, tc.upstream)
			if ok != tc.want {
				t.Fatalf("ok = %v, want %v", ok, tc.want)
			}
			if !ok {
				return
			}
			if got := bridge.UpstreamProtocol(); got != tc.wantUp {
				t.Fatalf("UpstreamProtocol() = %q, want %q", got, tc.wantUp)
			}
		})
	}
}

func TestAnthropicToOpenAIChatBridgeEncodeUpstreamRequest(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol() ok = false, want true")
	}

	maxTokens := 64
	req := &LLMRequest{
		Protocol: ProtocolAnthropicMessages,
		Model:    "claude-sonnet",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}},
		},
		MaxOutputTokens: &maxTokens,
		Tools:           []Tool{{Type: ToolFunction, Name: "no_args", InputSchema: map[string]any{"type": "object"}}},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded["model"] != "gpt-5.4" {
		t.Fatalf("model = %v, want the override gpt-5.4", decoded["model"])
	}
	if decoded["max_completion_tokens"] != float64(64) {
		t.Fatalf("max_completion_tokens = %v, want 64", decoded["max_completion_tokens"])
	}
	messages, ok := decoded["messages"].([]any)
	if !ok || len(messages) != 2 {
		t.Fatalf("messages = %v, want the system and user turns", decoded["messages"])
	}
	if role := messages[0].(map[string]any)["role"]; role != "system" {
		t.Fatalf("first message role = %v, want system", role)
	}
	if tools, ok := decoded["tools"].([]any); !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want the declared function", decoded["tools"])
	}
	// The Anthropic-only field must not reach an OpenAI upstream.
	if _, present := decoded["input"]; present {
		t.Fatal("encode produced a Responses-shaped payload")
	}
}

func TestAnthropicToOpenAIChatBridgeDecodeUpstreamResponse(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol() ok = false, want true")
	}

	raw := []byte(`{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"model": "gpt-5.4",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
	}`)

	resp, err := bridge.DecodeUpstreamResponse(raw)
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	if resp.Protocol != ProtocolOpenAIChat {
		t.Fatalf("Protocol = %q, want %q", resp.Protocol, ProtocolOpenAIChat)
	}
	parts, finish := firstResponseContent(resp)
	if finish != FinishStop {
		t.Errorf("finish reason = %q, want %q", finish, FinishStop)
	}
	if len(parts) != 1 || parts[0].Text == nil || parts[0].Text.Text != "hi there" {
		t.Fatalf("content = %#v, want the assistant text", parts)
	}
}

// TestAnthropicToOpenAIChatStreamEncoderRebasesUsage pins the one piece of
// protocol knowledge the chat bridge cannot delegate: the chat decoder reports
// prompt tokens the OpenAI way (a total with the cached portion broken out),
// while Anthropic expects input_tokens to exclude the cached tokens. Passing
// them through unchanged reports every cached token twice.
func TestAnthropicToOpenAIChatStreamEncoderRebasesUsage(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol() ok = false, want true")
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	promptTokens := 100
	cachedTokens := 40
	outputTokens := 10
	events, err := encoder.Encode(StreamPart{
		Type: StreamFinish,
		Usage: Usage{
			InputTokens:       &promptTokens,
			CachedInputTokens: &cachedTokens,
			OutputTokens:      &outputTokens,
		},
	})
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	usage := anthropicUsageFromStreamEvents(t, events)
	if got := intValue(usage.InputTokens); got != 60 {
		t.Errorf("input_tokens = %d, want 60 (100 total minus 40 cached)", got)
	}
	if got := intValue(usage.CacheReadInputTokens); got != 40 {
		t.Errorf("cache_read_input_tokens = %d, want 40", got)
	}
	if got := intValue(usage.OutputTokens); got != 10 {
		t.Errorf("output_tokens = %d, want 10", got)
	}
}

// anthropicUsageFromStreamEvents finds the usage an Anthropic stream reports,
// which arrives either as a message_delta usage or a message_start message usage.
func anthropicUsageFromStreamEvents(t *testing.T, events []RawStreamEvent) anthropicUsage {
	t.Helper()
	for _, event := range events {
		var envelope struct {
			Type    string          `json:"type"`
			Usage   *anthropicUsage `json:"usage"`
			Message *struct {
				Usage anthropicUsage `json:"usage"`
			} `json:"message"`
		}
		if err := json.Unmarshal(event.Data, &envelope); err != nil {
			continue
		}
		if envelope.Usage != nil {
			return *envelope.Usage
		}
		if envelope.Message != nil {
			return envelope.Message.Usage
		}
	}
	t.Fatalf("no usage found in %d stream events", len(events))
	return anthropicUsage{}
}

// A chat upstream never announces a content block: its text arrives as bare
// content deltas. Anthropic opens every block before streaming into it, so the
// bridge has to emit the content_block_start itself. Without it the client has
// a delta for an index it was never told about — and now that the branch also
// emits the matching content_block_stop, the pair would be half a block.
func TestAnthropicToOpenAIChatBridgeOpensAndClosesTextBlocks(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(anthropic_messages, openai_chat) ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	order := make([]string, 0, 8)
	encode := func(parts []StreamPart) {
		t.Helper()
		for _, part := range parts {
			events, err := encoder.Encode(part)
			if err != nil {
				t.Fatalf("Encode(%s) error = %v", part.Type, err)
			}
			for _, event := range events {
				order = append(order, event.Event)
			}
		}
	}

	for _, chunk := range []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		`[DONE]`,
	} {
		parts, err := decoder.Decode(chatStreamChunk(t, chunk))
		if err != nil {
			t.Fatalf("Decode(%s) error = %v", chunk, err)
		}
		encode(parts)
	}

	closed, err := decoder.Close()
	if err != nil {
		t.Fatalf("decoder Close() error = %v", err)
	}
	encode(closed)
	tail, err := encoder.Close()
	if err != nil {
		t.Fatalf("encoder Close() error = %v", err)
	}
	for _, event := range tail {
		order = append(order, event.Event)
	}

	want := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if len(order) != len(want) {
		t.Fatalf("events = %v, want %v", order, want)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("events = %v, want %v", order, want)
		}
	}
}

// A chat upstream reports reasoning as bare reasoning_content deltas, so an
// Anthropic client is never told where the thinking block begins or ends.
// Anthropic opens every block before streaming into it, so both halves have to
// come from here: a client otherwise reads thinking_delta for an index nothing
// announced and, since the decoder ends the reasoning, a content_block_stop for
// a block that never opened.
func TestAnthropicToOpenAIChatBridgeOpensAndClosesThinkingBlocks(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(anthropic_messages, openai_chat) ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	type frame struct {
		event    string
		index    float64
		hasIndex bool
		block    string
		delta    string
	}
	frames := make([]frame, 0, 12)
	record := func(event RawStreamEvent) {
		t.Helper()
		var payload map[string]any
		if err := json.Unmarshal(event.Data, &payload); err != nil {
			t.Fatalf("Unmarshal(%s) error = %v", event.Event, err)
		}
		current := frame{event: event.Event}
		if index, ok := payload["index"].(float64); ok {
			current.index, current.hasIndex = index, true
		}
		if block, ok := payload["content_block"].(map[string]any); ok {
			current.block, _ = block["type"].(string)
		}
		if delta, ok := payload["delta"].(map[string]any); ok {
			current.delta, _ = delta["type"].(string)
		}
		frames = append(frames, current)
	}
	consume := func(parts []StreamPart) {
		t.Helper()
		for _, part := range parts {
			events, err := encoder.Encode(part)
			if err != nil {
				t.Fatalf("Encode(%s) error = %v", part.Type, err)
			}
			for _, event := range events {
				record(event)
			}
		}
	}

	for _, chunk := range []string{
		`{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think. "},"finish_reason":null}]}`,
		`{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"reasoning_content":"Two plus two."},"finish_reason":null}]}`,
		`{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{"content":"Four."},"finish_reason":null}]}`,
		`{"id":"chatcmpl-a","object":"chat.completion.chunk","created":1,"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`[DONE]`,
	} {
		parts, err := decoder.Decode(chatStreamChunk(t, chunk))
		if err != nil {
			t.Fatalf("Decode(%s) error = %v", chunk, err)
		}
		consume(parts)
	}
	closed, err := decoder.Close()
	if err != nil {
		t.Fatalf("decoder Close() error = %v", err)
	}
	consume(closed)
	tail, err := encoder.Close()
	if err != nil {
		t.Fatalf("encoder Close() error = %v", err)
	}
	for _, event := range tail {
		record(event)
	}

	// The thinking block must be announced first, carry the streamed deltas, and
	// be stopped once.
	thinkingAt := -1
	for i, current := range frames {
		if current.event == "content_block_start" && current.block == "thinking" {
			thinkingAt = i
			break
		}
	}
	if thinkingAt < 0 {
		t.Fatalf("no thinking content_block_start in %+v", frames)
	}
	thinkingIndex := frames[thinkingAt].index
	var thinkingDeltas strings.Builder
	stops := 0
	for _, current := range frames[thinkingAt:] {
		if !current.hasIndex || current.index != thinkingIndex {
			continue
		}
		switch current.event {
		case "content_block_start":
			if current != frames[thinkingAt] {
				t.Fatalf("the thinking index was started twice: %+v", frames)
			}
		case "content_block_delta":
			if current.delta != "thinking_delta" {
				t.Fatalf("delta on the thinking index = %q, want thinking_delta", current.delta)
			}
			thinkingDeltas.WriteString(".")
		case "content_block_stop":
			stops++
		default:
			t.Fatalf("unexpected event on the thinking index: %+v", current)
		}
	}
	if thinkingDeltas.Len() != 2 {
		t.Fatalf("thinking deltas = %d, want the two the upstream sent", thinkingDeltas.Len())
	}
	if stops != 1 {
		t.Fatalf("thinking block stops = %d, want one", stops)
	}

	// The text that follows is its own block, with its own start and stop.
	textAt := -1
	for i, current := range frames {
		if current.event == "content_block_start" && current.block == "text" {
			textAt = i
			break
		}
	}
	if textAt < 0 {
		t.Fatalf("no text content_block_start in %+v", frames)
	}
	if frames[textAt].index == thinkingIndex {
		t.Fatalf("text block reused the thinking index %v", thinkingIndex)
	}
	if textAt < thinkingAt {
		t.Fatalf("the text block was announced before the thinking block: %+v", frames)
	}
}

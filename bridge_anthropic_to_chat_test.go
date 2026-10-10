package protocolbridge

import (
	"encoding/json"
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

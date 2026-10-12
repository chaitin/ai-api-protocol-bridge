package protocolbridge

import (
	"encoding/json"
	"testing"
)

// Anthropic only caches a prefix that a cache_control breakpoint marks, and a
// cache write is billed above the normal input rate. The encoder therefore has
// to be explicit about when it invents one: the request's own preference wins,
// and a host can turn the default off.

func cacheControlBlocks(t *testing.T, raw []byte) []map[string]any {
	t.Helper()

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", raw, err)
	}

	var blocks []map[string]any
	if system, ok := decoded["system"].([]any); ok {
		for _, item := range system {
			if block, ok := item.(map[string]any); ok {
				if _, ok := block["cache_control"]; ok {
					blocks = append(blocks, block)
				}
			}
		}
	}
	if messages, ok := decoded["messages"].([]any); ok {
		for _, item := range messages {
			message, ok := item.(map[string]any)
			if !ok {
				continue
			}
			content, ok := message["content"].([]any)
			if !ok {
				continue
			}
			for _, entry := range content {
				if block, ok := entry.(map[string]any); ok {
					if _, ok := block["cache_control"]; ok {
						blocks = append(blocks, block)
					}
				}
			}
		}
	}
	return blocks
}

func encodeChatToAnthropic(t *testing.T, cache *bool, policy CacheControlPolicy) []byte {
	t.Helper()

	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolAnthropicMessages)
	if !ok {
		t.Fatal("no chat-to-anthropic bridge")
	}
	req := &LLMRequest{
		Protocol: ProtocolOpenAIChat,
		Model:    "claude-sonnet-4",
		Cache:    cache,
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are a coding agent."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "hi"}}}},
		},
	}
	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet-4", CacheControl: policy})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	return raw
}

func TestAnthropicCacheControlPolicy(t *testing.T) {
	enabled := true
	disabled := false

	cases := []struct {
		name        string
		cache       *bool
		policy      CacheControlPolicy
		wantBlocks  int
		explanation string
	}{
		{name: "auto adds one breakpoint", policy: CacheControlAuto, wantBlocks: 1, explanation: "nothing is cached without a breakpoint"},
		{name: "disabled adds none", policy: CacheControlDisabled, wantBlocks: 0, explanation: "the host opted out of cache writes"},
		{name: "enabled adds one", policy: CacheControlEnabled, wantBlocks: 1, explanation: "explicitly requested"},
		{name: "request asking for cache wins over disabled", cache: &enabled, policy: CacheControlDisabled, wantBlocks: 1, explanation: "the caller expressed a preference"},
		{name: "request refusing cache wins over enabled", cache: &disabled, policy: CacheControlEnabled, wantBlocks: 0, explanation: "the caller expressed a preference"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := encodeChatToAnthropic(t, tc.cache, tc.policy)
			blocks := cacheControlBlocks(t, raw)
			if len(blocks) != tc.wantBlocks {
				t.Fatalf("cache_control blocks = %d, want %d (%s): %s", len(blocks), tc.wantBlocks, tc.explanation, raw)
			}
			for _, block := range blocks {
				control, ok := block["cache_control"].(map[string]any)
				if !ok || control["type"] != "ephemeral" {
					t.Fatalf("cache_control = %+v", block["cache_control"])
				}
			}
		})
	}
}

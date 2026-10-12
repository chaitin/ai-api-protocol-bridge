package protocolbridge

import (
	"bytes"
	"testing"
)

// ProviderOptions sits on the request, the message, the part and the tool, and
// nothing reads it. A host that sets it expecting provider-specific passthrough
// gets a silent no-op, which is the failure mode this package exists to avoid.
//
// The fields cannot be deleted without a major version, so this pins the
// behaviour instead: setting them anywhere must leave the encoded request
// byte-identical. If someone wires passthrough up, this fails and the test — and
// the deprecation notes — have to be updated together, on purpose.
func TestProviderOptionsDoNotReachTheUpstream(t *testing.T) {
	build := func(withProviderOptions bool) *LLMRequest {
		req := &LLMRequest{
			Protocol:        ProtocolOpenAIChat,
			Model:           "upstream-model",
			MaxOutputTokens: intPtr(1024),
			Prompt: []Message{{
				Role:  RoleUser,
				Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}},
			}},
			Tools: []Tool{{
				Type:        ToolFunction,
				Name:        "shell",
				Description: "Run a shell command.",
				InputSchema: map[string]any{"type": "object"},
			}},
		}
		if !withProviderOptions {
			return req
		}

		options := map[string]any{
			"anthropic": map[string]any{"top_k": 5},
			"openai":    map[string]any{"service_tier": "flex"},
		}
		req.ProviderOptions = options
		req.Prompt[0].ProviderOptions = options
		req.Prompt[0].Parts[0].ProviderOptions = options
		req.Tools[0].ProviderOptions = options
		return req
	}

	protocols := []Protocol{ProtocolOpenAIChat, ProtocolOpenAIResponses, ProtocolAnthropicMessages}

	// The identity case: the adapter's own encoder.
	for _, protocol := range protocols {
		t.Run("adapter/"+string(protocol), func(t *testing.T) {
			adapter := adapterForProtocol(t, protocol)
			plain, err := adapter.EncodeRequest(build(false), EncodeRequestOptions{Model: "upstream-model"})
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			withOptions, err := adapter.EncodeRequest(build(true), EncodeRequestOptions{Model: "upstream-model"})
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			if !bytes.Equal(plain, withOptions) {
				t.Fatalf("ProviderOptions changed the request:\n--- without ---\n%s\n--- with ---\n%s", plain, withOptions)
			}
		})
	}

	// Every cross-protocol cell, since the bridges build their own wire request
	// rather than delegating to the target adapter.
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
				plain, err := bridge.EncodeUpstreamRequest(build(false), EncodeRequestOptions{Model: "upstream-model"})
				if err != nil {
					t.Fatalf("EncodeUpstreamRequest() error = %v", err)
				}
				withOptions, err := bridge.EncodeUpstreamRequest(build(true), EncodeRequestOptions{Model: "upstream-model"})
				if err != nil {
					t.Fatalf("EncodeUpstreamRequest() error = %v", err)
				}
				if !bytes.Equal(plain, withOptions) {
					t.Fatalf("ProviderOptions changed the request:\n--- without ---\n%s\n--- with ---\n%s", plain, withOptions)
				}
			})
		}
	}
}

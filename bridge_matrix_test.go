package protocolbridge

import "testing"

// The bridge matrix is the repository's contract with its host: for every pair
// of distinct protocols there is exactly one bridge, and it knows which way
// round it is. A missing cell is not a compile error and not a crash — the host
// gets ok = false, falls back to reusing the inbound adapter, and posts one
// protocol's JSON at the other protocol's endpoint. Pinning every cell here is
// what turns that silent failure into a red test.

// bridgeableProtocols lists the protocols the repository converts between. The
// count is asserted below, so adding a protocol without deciding its bridges
// fails rather than quietly shrinking the matrix.
var bridgeableProtocols = []Protocol{
	ProtocolOpenAIChat,
	ProtocolOpenAIResponses,
	ProtocolAnthropicMessages,
}

const wantProtocolCount = 3

func TestBridgeMatrixIsComplete(t *testing.T) {
	if len(bridgeableProtocols) != wantProtocolCount {
		t.Fatalf("bridgeableProtocols has %d entries, want %d; every pair below must be revisited", len(bridgeableProtocols), wantProtocolCount)
	}

	// Every ordered pair of distinct protocols needs a bridge. Three protocols
	// give six cells.
	checked := 0
	for _, inbound := range bridgeableProtocols {
		for _, upstream := range bridgeableProtocols {
			if inbound == upstream {
				continue
			}
			checked++
			name := string(inbound) + " to " + string(upstream)
			t.Run(name, func(t *testing.T) {
				bridge, ok := NewCrossFamilyBridgeForProtocol(inbound, upstream)
				if !ok {
					t.Fatalf("no bridge for %s; the host would fall back to reusing the inbound adapter", name)
				}
				if got := bridge.InboundProtocol(); got != inbound {
					t.Fatalf("InboundProtocol() = %q, want %q", got, inbound)
				}
				if got := bridge.UpstreamProtocol(); got != upstream {
					t.Fatalf("UpstreamProtocol() = %q, want %q; a family lookup returned the wrong target", got, upstream)
				}
				// A bridge that cannot build its stream halves is not usable
				// against a streaming client, which is the normal case for a
				// coding agent.
				if _, err := bridge.NewStreamDecoder(StreamDecodeOptions{}); err != nil {
					t.Fatalf("NewStreamDecoder() error = %v", err)
				}
				if _, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "test-model"}); err != nil {
					t.Fatalf("NewStreamEncoder() error = %v", err)
				}
			})
		}
	}

	want := len(bridgeableProtocols) * (len(bridgeableProtocols) - 1)
	if checked != want {
		t.Fatalf("checked %d pairs, want %d", checked, want)
	}
}

func TestBridgeMatrixIdentityNeedsNoBridge(t *testing.T) {
	for _, protocol := range bridgeableProtocols {
		if bridge, ok := NewCrossFamilyBridgeForProtocol(protocol, protocol); ok {
			t.Fatalf("NewCrossFamilyBridgeForProtocol(%q, %q) returned a %T; an identity needs no conversion", protocol, protocol, bridge)
		}
	}
}

// The family-based constructor is retained, but it cannot express the two
// OpenAI targets. This pins the ambiguity so the protocol-precise constructor
// stays the documented one.
func TestFamilyBridgeLookupIsAmbiguousForAnthropicInbound(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, FamilyOpenAI)
	if !ok {
		t.Fatal("NewCrossFamilyBridge(anthropic_messages, openai) ok = false, want true")
	}
	if got := bridge.UpstreamProtocol(); got != ProtocolOpenAIResponses {
		t.Fatalf("UpstreamProtocol() = %q, want %q", got, ProtocolOpenAIResponses)
	}

	// The protocol-precise lookup does reach the other OpenAI target.
	precise, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(anthropic_messages, openai_chat) ok = false, want true")
	}
	if got := precise.UpstreamProtocol(); got != ProtocolOpenAIChat {
		t.Fatalf("UpstreamProtocol() = %q, want %q", got, ProtocolOpenAIChat)
	}
}

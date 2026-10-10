package protocolbridge

type CrossFamilyBridge interface {
	InboundProtocol() Protocol
	UpstreamProtocol() Protocol
	EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error)
	DecodeUpstreamResponse(raw []byte) (*LLMResponse, error)
	NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error)
	NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error)
}

type anthropicToOpenAIResponsesBridge struct {
	adapter OpenAIResponsesAdapter
}

type openAIResponsesToAnthropicBridge struct {
	adapter AnthropicMessagesAdapter
}

type openAIChatToAnthropicBridge struct {
	adapter AnthropicMessagesAdapter
}

const (
	FamilyOpenAI    = "openai"
	FamilyAnthropic = "anthropic"
)

func NewCrossFamilyBridge(inbound Protocol, upstreamFamily string) (CrossFamilyBridge, bool) {
	switch {
	case inbound == ProtocolAnthropicMessages && upstreamFamily == FamilyOpenAI:
		return anthropicToOpenAIResponsesBridge{adapter: NewOpenAIResponsesAdapter()}, true
	case inbound == ProtocolOpenAIResponses && upstreamFamily == FamilyAnthropic:
		return openAIResponsesToAnthropicBridge{adapter: NewAnthropicMessagesAdapter()}, true
	case inbound == ProtocolOpenAIChat && upstreamFamily == FamilyAnthropic:
		return openAIChatToAnthropicBridge{adapter: NewAnthropicMessagesAdapter()}, true
	default:
		return nil, false
	}
}

// NewCrossFamilyBridgeForProtocol returns the bridge for an exact inbound and
// upstream protocol pair.
//
// Prefer this over NewCrossFamilyBridge. Families are not specific enough: an
// Anthropic inbound can reach either OpenAI target, so a family lookup returns
// the Responses bridge whatever the caller actually wanted. Callers that know
// their upstream's protocol should say so, and a caller that only knows the
// family should be aware it may get the wrong one.
//
// A pair within one family returns false, as does an unserved pair: those need
// no conversion, or none exists.
func NewCrossFamilyBridgeForProtocol(inbound, upstream Protocol) (CrossFamilyBridge, bool) {
	if inbound == upstream {
		return nil, false
	}
	switch {
	case inbound == ProtocolAnthropicMessages && upstream == ProtocolOpenAIChat:
		return anthropicToOpenAIChatBridge{adapter: NewOpenAIChatAdapter()}, true
	case inbound == ProtocolOpenAIResponses && upstream == ProtocolOpenAIChat:
		return openAIResponsesToOpenAIChatBridge{adapter: NewOpenAIChatAdapter()}, true
	}
	return NewCrossFamilyBridge(inbound, familyForProtocol(upstream))
}

func familyForProtocol(protocol Protocol) string {
	switch protocol {
	case ProtocolOpenAIChat, ProtocolOpenAIResponses:
		return FamilyOpenAI
	case ProtocolAnthropicMessages:
		return FamilyAnthropic
	default:
		return ""
	}
}

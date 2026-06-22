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

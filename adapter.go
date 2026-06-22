package protocolbridge

type EncodeRequestOptions struct {
	Model string
}

type EncodeResponseOptions struct {
	Model string
}

type StreamDecodeOptions struct{}

type StreamEncodeOptions struct {
	Model string
}

type Adapter interface {
	Protocol() Protocol

	DecodeRequest(raw []byte) (*LLMRequest, error)
	EncodeRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error)

	DecodeResponse(raw []byte) (*LLMResponse, error)
	EncodeResponse(resp *LLMResponse, opts EncodeResponseOptions) ([]byte, error)

	NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error)
	NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error)

	EncodeError(err error) ([]byte, int)
}

type StreamDecoder interface {
	Decode(event RawStreamEvent) ([]StreamPart, error)
	Close() ([]StreamPart, error)
}

type StreamEncoder interface {
	Encode(part StreamPart) ([]RawStreamEvent, error)
	Close() ([]RawStreamEvent, error)
	EncodeError(err error) []RawStreamEvent
}

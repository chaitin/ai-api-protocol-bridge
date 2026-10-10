package protocolbridge

type EncodeRequestOptions struct {
	Model string
}

type EncodeResponseOptions struct {
	Model string

	// Created is the Unix timestamp in seconds to stamp on the response. Zero
	// means "use the current time"; hosts that need deterministic output for
	// snapshots or caching should set it.
	Created int64
}

type StreamDecodeOptions struct{}

type StreamEncodeOptions struct {
	Model string

	// Created is the Unix timestamp in seconds to stamp on every event of the
	// stream. Zero means "stamp the first event and reuse that value", which is
	// what a real provider does: one timestamp for the whole completion.
	Created int64
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

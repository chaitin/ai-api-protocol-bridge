package protocolbridge

import (
	"errors"
	"testing"
)

type stubAdapter struct {
	protocol Protocol
}

func (s stubAdapter) Protocol() Protocol { return s.protocol }

func (s stubAdapter) DecodeRequest([]byte) (*LLMRequest, error) { return nil, nil }

func (s stubAdapter) EncodeRequest(*LLMRequest, EncodeRequestOptions) ([]byte, error) {
	return nil, nil
}

func (s stubAdapter) DecodeResponse([]byte) (*LLMResponse, error) { return nil, nil }

func (s stubAdapter) EncodeResponse(*LLMResponse, EncodeResponseOptions) ([]byte, error) {
	return nil, nil
}

func (s stubAdapter) NewStreamDecoder(StreamDecodeOptions) (StreamDecoder, error) { return nil, nil }

func (s stubAdapter) NewStreamEncoder(StreamEncodeOptions) (StreamEncoder, error) { return nil, nil }

func (s stubAdapter) EncodeError(error) ([]byte, int) { return nil, 500 }

func TestRegistryAdapter(t *testing.T) {
	registry, err := NewRegistry(stubAdapter{protocol: ProtocolOpenAIChat})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}

	adapter, ok := registry.Adapter(ProtocolOpenAIChat)
	if !ok {
		t.Fatal("Adapter() ok = false")
	}
	if adapter.Protocol() != ProtocolOpenAIChat {
		t.Fatalf("adapter.Protocol() = %q", adapter.Protocol())
	}

	if _, ok := registry.Adapter(ProtocolAnthropicMessages); ok {
		t.Fatal("Adapter() returned unsupported protocol")
	}
}

func TestRegistryRejectsDuplicateAdapter(t *testing.T) {
	_, err := NewRegistry(
		stubAdapter{protocol: ProtocolOpenAIChat},
		stubAdapter{protocol: ProtocolOpenAIChat},
	)
	if err == nil {
		t.Fatal("NewRegistry() error = nil")
	}
}

func TestRegistryRejectsEmptyProtocol(t *testing.T) {
	_, err := NewRegistry(stubAdapter{})
	if err == nil {
		t.Fatal("NewRegistry() error = nil")
	}
}

func TestRegistryRejectsNilAdapter(t *testing.T) {
	_, err := NewRegistry(nil)
	if err == nil {
		t.Fatal("NewRegistry() error = nil")
	}
}

func TestRegistryMustAdapter(t *testing.T) {
	registry, err := NewRegistry(stubAdapter{protocol: ProtocolOpenAIChat})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}

	if _, err := registry.MustAdapter(ProtocolOpenAIChat); err != nil {
		t.Fatalf("MustAdapter() error = %v", err)
	}

	if _, err := registry.MustAdapter(ProtocolAnthropicMessages); err == nil {
		t.Fatal("MustAdapter() error = nil")
	}
}

func TestStubAdapterEncodeError(t *testing.T) {
	_, status := stubAdapter{}.EncodeError(errors.New("boom"))
	if status != 500 {
		t.Fatalf("EncodeError() status = %d", status)
	}
}

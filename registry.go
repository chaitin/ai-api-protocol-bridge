package protocolbridge

import "fmt"

type Registry struct {
	adapters map[Protocol]Adapter
}

func NewRegistry(adapters ...Adapter) (*Registry, error) {
	registry := &Registry{adapters: make(map[Protocol]Adapter, len(adapters))}
	for _, adapter := range adapters {
		if adapter == nil {
			return nil, fmt.Errorf("protocolbridge: nil adapter")
		}
		protocol := adapter.Protocol()
		if protocol == "" {
			return nil, fmt.Errorf("protocolbridge: adapter has empty protocol")
		}
		if _, exists := registry.adapters[protocol]; exists {
			return nil, fmt.Errorf("protocolbridge: duplicate adapter for protocol %q", protocol)
		}
		registry.adapters[protocol] = adapter
	}
	return registry, nil
}

func (r *Registry) Adapter(protocol Protocol) (Adapter, bool) {
	if r == nil {
		return nil, false
	}
	adapter, ok := r.adapters[protocol]
	return adapter, ok
}

func (r *Registry) MustAdapter(protocol Protocol) (Adapter, error) {
	adapter, ok := r.Adapter(protocol)
	if !ok {
		return nil, fmt.Errorf("protocolbridge: unsupported protocol %q", protocol)
	}
	return adapter, nil
}

package protocolbridge

func anthropicUsageToResponsesUsage(usage Usage) Usage {
	converted := usage
	cacheRead := usage.CacheReadInputTokens
	if cacheRead == nil {
		cacheRead = usage.CachedInputTokens
	}
	totalInput := intValue(usage.InputTokens) + intValue(usage.CacheCreationInputTokens)
	if cacheRead != nil {
		totalInput += intValue(cacheRead)
	}
	convertedInput := totalInput
	converted.InputTokens = &convertedInput
	if cacheRead != nil {
		cached := intValue(cacheRead)
		converted.CachedInputTokens = &cached
	} else {
		converted.CachedInputTokens = nil
	}
	converted.CacheReadInputTokens = nil
	converted.CacheCreationInputTokens = nil
	return converted
}

func responsesUsageToAnthropicUsage(usage Usage) Usage {
	converted := usage
	cached := intValue(usage.CachedInputTokens)
	input := clampNonNegative(intValue(usage.InputTokens) - cached)
	convertedInput := input
	converted.InputTokens = &convertedInput
	converted.CacheReadInputTokens = usage.CachedInputTokens
	converted.CacheCreationInputTokens = nil
	return converted
}

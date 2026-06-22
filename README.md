# AI API Protocol Bridge

AI API Protocol Bridge 是一个 Go 协议转换包，用来在 OpenAI 和 Anthropic
的 API 协议形态之间转换请求、响应和 SSE 流式事件。

它的定位很窄：只负责协议层的编解码与跨协议映射，不负责 HTTP 代理、鉴权、
上游路由、计费、调用记录或数据库访问。你可以把它放在网关、代理或模型调度
服务里，作为“入口协议”和“上游协议”之间的转换层。

## 能力概览

- 将不同厂商的请求 JSON 解码为统一的 `LLMRequest`。
- 将统一请求编码为目标协议的上游请求 JSON。
- 将上游响应解码为统一的 `LLMResponse`，再编码回入口协议。
- 通过 `StreamDecoder` / `StreamEncoder` 转换 SSE 流式事件。
- 支持文本、图片/文档文件、reasoning、refusal、工具调用、工具结果、JSON
  响应格式、用量统计等常见语义。

## 支持的协议

| 协议 | 常量 | Adapter |
| --- | --- | --- |
| OpenAI Chat Completions | `ProtocolOpenAIChat` | `NewOpenAIChatAdapter()` |
| OpenAI Responses | `ProtocolOpenAIResponses` | `NewOpenAIResponsesAdapter()` |
| Anthropic Messages | `ProtocolAnthropicMessages` | `NewAnthropicMessagesAdapter()` |

## 支持的跨协议桥

| 入口协议 | 上游模型家族 | 实际上游协议 |
| --- | --- | --- |
| OpenAI Chat Completions | Anthropic | Anthropic Messages |
| OpenAI Responses | Anthropic | Anthropic Messages |
| Anthropic Messages | OpenAI | OpenAI Responses |

Anthropic 入口转 OpenAI 上游时，会使用 OpenAI Responses，而不是 OpenAI
Chat Completions。

## 安装

当前 `go.mod` 声明的 module path 是：

```bash
go get github.com/chaitin/ai-api-protocol-bridge
```

Go 版本要求见 [go.mod](./go.mod)。

## 基本模型

包内转换分两层：

```text
原始协议 JSON
  -> Adapter.DecodeRequest / DecodeResponse
  -> LLMRequest / LLMResponse
  -> Adapter.EncodeRequest / EncodeResponse
  -> 目标协议 JSON
```

跨 OpenAI / Anthropic 家族调用时，可以用 `NewCrossFamilyBridge` 直接得到
上游协议的编码器和流式转换器：

```text
入口请求 JSON -> 入口 Adapter -> LLMRequest
LLMRequest -> CrossFamilyBridge.EncodeUpstreamRequest -> 上游请求 JSON
上游响应 JSON -> CrossFamilyBridge.DecodeUpstreamResponse -> LLMResponse
LLMResponse -> 入口 Adapter.EncodeResponse -> 入口协议响应 JSON
```

## 非流式示例

下面示例把 OpenAI Chat Completions 请求转换为 Anthropic Messages 上游请求。

```go
package main

import (
	"fmt"

	protocolbridge "github.com/chaitin/ai-api-protocol-bridge"
)

func main() {
	inbound := protocolbridge.NewOpenAIChatAdapter()

	req, err := inbound.DecodeRequest([]byte(`{
		"model": "gpt-4.1",
		"messages": [
			{"role": "user", "content": "hello"}
		]
	}`))
	if err != nil {
		panic(err)
	}

	bridge, ok := protocolbridge.NewCrossFamilyBridge(
		protocolbridge.ProtocolOpenAIChat,
		protocolbridge.FamilyAnthropic,
	)
	if !ok {
		panic("unsupported bridge")
	}

	upstreamBody, err := bridge.EncodeUpstreamRequest(req, protocolbridge.EncodeRequestOptions{
		Model: "claude-sonnet-4",
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(string(upstreamBody))
}
```

上游返回后，可以先解码成统一响应，再编码回入口协议：

```go
upstreamResp, err := bridge.DecodeUpstreamResponse(rawAnthropicResponse)
if err != nil {
	panic(err)
}

clientBody, err := inbound.EncodeResponse(upstreamResp, protocolbridge.EncodeResponseOptions{
	Model: req.Model,
})
if err != nil {
	panic(err)
}
```

## 流式转换

流式转换使用统一的 `StreamPart` 作为中间事件：

```text
上游 SSE event -> StreamDecoder -> StreamPart -> StreamEncoder -> 入口 SSE event
```

使用跨协议桥时，decoder 面向上游协议，encoder 面向入口协议：

```go
decoder, err := bridge.NewStreamDecoder(protocolbridge.StreamDecodeOptions{})
if err != nil {
	panic(err)
}

encoder, err := bridge.NewStreamEncoder(protocolbridge.StreamEncodeOptions{
	Model: req.Model,
})
if err != nil {
	panic(err)
}

parts, err := decoder.Decode(protocolbridge.RawStreamEvent{
	Event: "content_block_delta",
	Data:  rawSSEData,
})
if err != nil {
	panic(err)
}

for _, part := range parts {
	events, err := encoder.Encode(part)
	if err != nil {
		panic(err)
	}
	for _, event := range events {
		_ = event // 写回客户端 SSE
	}
}

tailParts, err := decoder.Close()
if err != nil {
	panic(err)
}
for _, part := range tailParts {
	events, err := encoder.Encode(part)
	if err != nil {
		panic(err)
	}
	_ = events
}

finalEvents, err := encoder.Close()
if err != nil {
	panic(err)
}
_ = finalEvents
```

`RawStreamEvent` 只表示已经解析出的 SSE 事件，不负责从 HTTP body 中切分 SSE。
调用方需要自己完成网络读写、重连、flush 和错误处理。

## Registry

如果你的服务需要按协议名查找适配器，可以使用 `Registry`：

```go
registry, err := protocolbridge.NewRegistry(
	protocolbridge.NewOpenAIChatAdapter(),
	protocolbridge.NewOpenAIResponsesAdapter(),
	protocolbridge.NewAnthropicMessagesAdapter(),
)
if err != nil {
	panic(err)
}

adapter, err := registry.MustAdapter(protocolbridge.ProtocolOpenAIResponses)
if err != nil {
	panic(err)
}

_ = adapter
```

## 数据结构说明

`LLMRequest` 是统一请求模型，主要字段包括：

- `Model`：入口模型名；编码上游请求时可通过 `EncodeRequestOptions.Model` 覆盖。
- `Prompt`：统一消息列表，支持 `system`、`developer`、`user`、`assistant`、
  `tool` 等角色。
- `MaxOutputTokens`、`Temperature`、`StopSequences`、`TopP`、`TopK` 等生成参数。
- `ResponseFormat`：文本或 JSON 输出格式。
- `Reasoning`、`ReasoningBudgetTokens`、`ReasoningEffort`、`ReasoningSummary`：
  reasoning 相关配置。
- `Tools`、`ToolChoice`、`ParallelToolCalls`：工具声明与工具选择策略。
- `State`、`Include`、`Cache`、`Metadata`：协议可选能力。

`Part` 是消息内容块，支持：

- `text`：普通文本。
- `file`：图片或文档，支持 URL、base64 data、file id 等来源。
- `reasoning`：推理内容、签名或加密内容。
- `refusal`：拒答内容。
- `tool-call`：模型发起的工具调用。
- `tool-result`：工具执行结果，支持文本、JSON、错误和多段内容。

`LLMResponse` 是统一响应模型，包含响应内容、choices、finish reason、usage、
provider metadata 和 warnings。`BillingUsage()` 会按协议差异归一化可计费的
输入、缓存输入和输出 token。

## 兼容性说明

- 跨协议请求会尽量保留双方都能表达的语义；目标协议不支持的字段可能被忽略、
  降级为 warning 文本，或在编码阶段返回错误。
- OpenAI Responses 上游编码不支持 `StopSequences`，传入后会返回错误。
- OpenAI 与 Anthropic 的 cache / usage 口径不同，跨协议响应会做必要的用量映射。
- Anthropic thinking 与强制工具选择存在协议限制，必要时会将工具选择降级为
  `auto`。
- Provider-specific 字段不会被当作完整透传能力；调用方应只依赖统一模型和目标
  协议明确支持的字段。

## 包边界

这个包不包含以下能力：

- API key 管理或鉴权。
- 上游供应商选择、模型路由、重试、熔断或负载均衡。
- HTTP handler、中间件、SSE 解析器或客户端实现。
- 计费系统、审计日志、调用记录或数据库访问。
- prompt 管理、内容安全策略或业务层错误码。

这些能力应由调用方所在的网关或业务服务实现。

## 开发

运行测试：

```bash
go test ./...
```

查看公开 API 时，可以从 [adapter.go](./adapter.go)、[types.go](./types.go) 和
[bridge_cross_family.go](./bridge_cross_family.go) 开始。

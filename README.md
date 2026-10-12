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

用 `NewCrossFamilyBridgeForProtocol(inbound, upstream)` 按**精确协议对**取桥：

| 入口协议 | 上游协议 | 说明 |
| --- | --- | --- |
| OpenAI Responses | OpenAI Chat Completions | codex 打到只支持 Chat 的上游 |
| OpenAI Chat Completions | Anthropic Messages | |
| OpenAI Responses | Anthropic Messages | |
| Anthropic Messages | OpenAI Chat Completions | |
| Anthropic Messages | OpenAI Responses | |
| OpenAI Chat Completions | OpenAI Responses | |

六个跨协议方向全部可用；只有完全相同的协议对（如 Chat → Chat）返回 `false`，
因为它不需要转换。`bridge_matrix_test.go` 会遍历并钉住每一个方向。

`NewCrossFamilyBridge(inbound, upstreamFamily)` 是按**家族**取桥的旧接口，仍然可用，
但家族不足以定位目标：Anthropic 入口有两个 OpenAI 目标，家族查询一律返回
Responses 桥，无论调用方实际想要哪个。已知上游协议时应改用
`NewCrossFamilyBridgeForProtocol`。

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

## 转换损耗

两个协议之间不可能什么都表达得下。丢东西的时候本包不会沉默，也不会把它伪装成
prompt 的一部分：

- 每个解码器/编码器把遇到的损耗记录到 `LLMRequest.Warnings`（入口方向）或
  `LLMResponse.Warnings`（出口方向）。每条 `Warning` 带 `Code`、`Severity`、
  `Path`（如 `messages[3].content[1]`）以及转换方向 `From`/`To`。
- `EncodeRequestOptions.LossPolicy` / `EncodeResponseOptions.LossPolicy` 决定什么
  不可接受：`Allow`（零值，记录后继续）、`Safe`（额外拒绝会改变模型行为的损耗，
  即 `SeverityError`）、`Strict`（拒绝任何丢了内容的损耗，即 severity 高于
  `SeverityInfo`）。被拒绝时返回 `*ConversionError`，逐条列出损耗，同时损耗仍
  写在对象上。

已经上报的损耗包括：Chat `file` 部件承载不了的文档、Anthropic 无法解析的文件 id
或媒体类型、目标协议没有对应物的工具、Responses API 没有的 stop sequences 字段、
只有单一 assistant 轮的协议承载不了的多候选，以及 chat completions 没有的
reasoning 字段。

不会做的事：把警告文本写进 system prompt 或对话内容。那会改变模型看到的字节、让
prompt cache 前缀每轮失效，而且模型会以为那是用户说的话。

## 兼容性说明

- 跨协议请求会尽量保留双方都能表达的语义；目标协议表达不了的字段会被记录下来，
  见上文「转换损耗」。
- OpenAI Responses 没有 stop sequences 字段，因此传入的 `StopSequences` 会被记录
  为损耗；默认继续，`LossPolicySafe`/`Strict` 下会失败。
- OpenAI 与 Anthropic 的 cache / usage 口径不同，跨协议响应会做必要的用量映射。
  `Usage` 的同一个字段在两种口径下含义不同，只有配合协议才有意义；换算用
  `BillingUsage()`。
- 流式上游中途断流（没有终止事件）会产生 `StreamError`，而不是看起来正常的短回答。
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

# 有意改动线上格式后，重写 golden 快照并阅读 diff
go test ./... -update
```

查看公开 API 时，可以从 [adapter.go](./adapter.go)、[types.go](./types.go) 和
[bridge_cross_family.go](./bridge_cross_family.go) 开始。

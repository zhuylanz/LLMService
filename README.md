# @lapage/llm-service

Unified TypeScript service wrapper for OpenAI, OpenAI-compatible, Claude, and Gemini APIs.

## Design Principle

Use one service and one configuration shape for every provider:

```ts
const service = new LLMService({ provider, apiKey, baseURL });
```

The package intentionally avoids provider-specific service classes and factory
functions. Provider differences belong in configuration, not in consumer code.

## Features

- One service interface for multiple providers.
- One initialization pattern for all providers.
- Chat completion and streaming support.
- Embeddings support across OpenAI, OpenAI-compatible endpoints, and Gemini.
- OpenAI/OpenAI-compatible helpers for images, moderation, and transcription.
- Consistent response shape with error handling.

## Installation

```bash
npm install @lapage/llm-service
```

## Quick Start

```ts
import { LLMService } from '@lapage/llm-service';

const service = new LLMService({
  provider: 'openai',
  apiKey: process.env.OPENAI_API_KEY!,
});

const result = await service.createChatCompletion([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Write one sentence about TypeScript.' },
]);

if (result.success) {
  console.log(result.content);
} else {
  console.error(result.error);
}
```

## Provider Configuration

```ts
import { LLMService } from '@lapage/llm-service';

const openai = new LLMService({
  provider: 'openai',
  apiKey: process.env.OPENAI_API_KEY!,
});

const customOpenAI = new LLMService({
  provider: 'custom_openai',
  apiKey: process.env.CUSTOM_OPENAI_API_KEY!,
  baseURL: process.env.CUSTOM_OPENAI_BASE_URL!,
});

const claude = new LLMService({
  provider: 'claude',
  apiKey: process.env.ANTHROPIC_API_KEY!,
});

const gemini = new LLMService({
  provider: 'gemini',
  apiKey: process.env.GEMINI_API_KEY!,
});
```

## OpenAI-Compatible Endpoints

Use `custom_openai` when your provider exposes an OpenAI-compatible API, such
as `/chat/completions`, `/embeddings`, or other OpenAI SDK-compatible routes.
Pass the provider endpoint with `baseURL`.

```ts
const service = new LLMService({
  provider: 'custom_openai',
  apiKey: process.env.CUSTOM_OPENAI_API_KEY!,
  baseURL: 'https://your-provider.example.com/v1',
});

const result = await service.createChatCompletion(
  [{ role: 'user', content: 'Hello from a compatible endpoint.' }],
  'your-provider-model',
);
```

## API

### LLMService Constructor

```ts
new LLMService(options)
```

```ts
interface LLMServiceOptions {
  provider?: 'openai' | 'custom_openai' | 'claude' | 'gemini';
  apiKey: string;
  organization?: string;
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
  apiVersion?: string;
  vertexAI?: boolean;
  project?: string;
  location?: string;
}
```

If `provider` is omitted, it defaults to `openai`.

### Methods

- createChatCompletion(messages, model?, temperature?, maxTokens?)
- createChatCompletionStream(messages, model?, temperature?, maxTokens?)
- createEmbeddings(input, model?)
- generateImage(prompt, n?, size?, quality?)
- createModeration(input)
- createTranscription(audioFilePath, options?)

All methods return:

```ts
interface LLMServiceResponse<T> {
  success: boolean;
  data?: T;
  content?: string;
  stream?: AsyncIterable<unknown>;
  images?: unknown[];
  results?: unknown[];
  error?: {
    status?: number;
    type?: string;
    message: string;
    requestId?: string;
  };
}
```

## Provider Notes

- OpenAI: supports all methods.
- Custom OpenAI-compatible: uses the OpenAI SDK with `baseURL`; method support depends on the compatible endpoint.
- Claude: supports chat and stream. Embeddings/images/moderation/transcription return unsupported operation errors.
- Gemini: supports chat, stream, and embeddings. Images/moderation/transcription return unsupported operation errors.
- Default model aliases are mapped by provider for convenience:
  - chat default gpt-4o-mini maps to claude-sonnet-4-5 and gemini-2.5-flash
  - stream default gpt-4o maps to claude-sonnet-4-5 and gemini-2.5-flash
  - embeddings default text-embedding-3-small maps to text-embedding-004 on gemini

## Streaming Example

```ts
const streamResult = await service.createChatCompletionStream([
  { role: 'user', content: 'Stream a short explanation of recursion.' },
]);

if (streamResult.success && streamResult.stream) {
  for await (const chunk of streamResult.stream) {
    console.log(chunk);
  }
}
```

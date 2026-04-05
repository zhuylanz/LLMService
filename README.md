# @lapage/llm-service

Unified TypeScript service wrapper for OpenAI, Claude, and Gemini APIs.

## Features

- One service interface for multiple providers.
- Chat completion and streaming support.
- Embeddings support across OpenAI and Gemini.
- OpenAI-only helpers for images, moderation, and transcription.
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

## Provider Helpers

```ts
import {
  OpenAIService,
  ClaudeService,
  GeminiService,
  createOpenAIService,
  createClaudeService,
  createGeminiService,
} from '@lapage/llm-service';

const openai = new OpenAIService(process.env.OPENAI_API_KEY!);
const claude = createClaudeService(process.env.ANTHROPIC_API_KEY!);
const gemini = new GeminiService(process.env.GEMINI_API_KEY!);
```

## API

### LLMService Constructor

```ts
new LLMService(configOrApiKey, organization?)
```

You can pass either:

- apiKey string (defaults provider to openai)
- options object:

```ts
interface LLMServiceOptions {
  provider?: 'openai' | 'claude' | 'gemini';
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


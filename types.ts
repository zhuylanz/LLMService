import type { Message as AnthropicMessage } from '@anthropic-ai/sdk/resources/messages/messages';
import type { Stream as AnthropicStream } from '@anthropic-ai/sdk/streaming';
import type { GenerateContentResponse as GeminiGenerateContentResponse } from '@google/genai';
import type {
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/chat/completions';
import type {
  CreateEmbeddingResponse,
  Embedding,
} from 'openai/resources/embeddings';
import type { ImagesResponse } from 'openai/resources/images';
import type { ModerationCreateResponse } from 'openai/resources/moderations';
import type { Transcription } from 'openai/resources/audio/transcriptions';
import type { Stream as OpenAIStream } from 'openai/streaming';

export type LLMProvider = 'openai' | 'claude' | 'gemini';

export type LLMMessageRole =
  | 'system'
  | 'user'
  | 'assistant'
  | 'function'
  | 'tool';

export interface LLMMessage {
  role: LLMMessageRole;
  content: string | Array<Record<string, unknown>>;
  name?: string;
}

export interface LLMError {
  status?: number;
  type?: string;
  message: string;
  requestId?: string;
}

export interface LLMServiceResponse<T> {
  success: boolean;
  data?: T;
  content?: string;
  stream?: AsyncIterable<unknown>;
  images?: unknown[];
  results?: unknown[];
  error?: LLMError;
}

export interface LLMServiceOptions {
  provider?: LLMProvider;
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

export interface GeminiOpenAICompatibleEmbedding {
  object: 'list';
  model: string;
  data: Array<Embedding>;
}

export interface ParsedMessageInput {
  system: string[];
  chat: LLMMessage[];
}

export type LLMChatCompletionData =
  | ChatCompletion
  | AnthropicMessage
  | GeminiGenerateContentResponse;

export type LLMChatStream =
  | OpenAIStream<ChatCompletionChunk>
  | AnthropicStream<
      import('@anthropic-ai/sdk/resources/messages/messages').RawMessageStreamEvent
    >
  | AsyncGenerator<GeminiGenerateContentResponse>;

export type LLMEmbeddingData =
  | CreateEmbeddingResponse
  | GeminiOpenAICompatibleEmbedding;

export type LLMImageData = ImagesResponse;

export type LLMModerationData = ModerationCreateResponse;

export type LLMTranscriptionData = Transcription;

export type OpenAIMessageRole = LLMMessageRole;
export type OpenAIMessage = LLMMessage;
export type OpenAIServiceResponse<T> = LLMServiceResponse<T>;

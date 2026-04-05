import type { MessageParam as AnthropicMessageParam } from '@anthropic-ai/sdk/resources/messages/messages';
import type {
  Content as GeminiContent,
  EmbedContentResponse as GeminiEmbedContentResponse,
  Part as GeminiPart,
} from '@google/genai';
import type { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import type { Embedding } from 'openai/resources/embeddings';
import type {
  GeminiOpenAICompatibleEmbedding,
  LLMMessage,
  ParsedMessageInput,
} from './types';

export const parseMessages = (messages: LLMMessage[]): ParsedMessageInput => {
  const system: string[] = [];
  const chat: LLMMessage[] = [];

  for (const message of messages) {
    if (message.role === 'system') {
      system.push(normalizeMessageContent(message));
      continue;
    }

    chat.push(message);
  }

  return { system, chat };
};

export const normalizeMessageContent = (message: LLMMessage): string => {
  if (typeof message.content === 'string') {
    return message.content;
  }

  return message.content
    .map((part) => {
      const maybeText = part['text'];
      if (typeof maybeText === 'string') {
        return maybeText;
      }
      return JSON.stringify(part);
    })
    .join('\n');
};

export const toOpenAIMessages = (
  messages: LLMMessage[],
): ChatCompletionMessageParam[] => {
  return messages.map((message) => {
    if (
      (message.role === 'function' || message.role === 'tool') &&
      !message.name
    ) {
      throw new Error(`${message.role} message requires a name property`);
    }

    return {
      role: message.role,
      content: normalizeMessageContent(message),
      name: message.name,
    } as ChatCompletionMessageParam;
  });
};

export const toClaudeMessages = (messages: LLMMessage[]) => {
  const parsed = parseMessages(messages);
  const claudeMessages: AnthropicMessageParam[] = parsed.chat.map((message) => {
    const content = normalizeMessageContent(message);

    if (message.role === 'assistant') {
      return {
        role: 'assistant',
        content,
      };
    }

    if (message.role === 'function' || message.role === 'tool') {
      const prefixed = message.name
        ? `[${message.role}:${message.name}] ${content}`
        : `[${message.role}] ${content}`;

      return {
        role: 'user',
        content: prefixed,
      };
    }

    return {
      role: 'user',
      content,
    };
  });

  return {
    system: parsed.system.length > 0 ? parsed.system.join('\n\n') : undefined,
    messages: claudeMessages,
  };
};

export const toGeminiMessages = (messages: LLMMessage[]) => {
  const parsed = parseMessages(messages);

  const contents: GeminiContent[] = parsed.chat.map((message) => {
    const text = normalizeMessageContent(message);
    const part: GeminiPart = {
      text:
        message.role === 'function' || message.role === 'tool'
          ? message.name
            ? `[${message.role}:${message.name}] ${text}`
            : `[${message.role}] ${text}`
          : text,
    };

    return {
      role: message.role === 'assistant' ? 'model' : 'user',
      parts: [part],
    };
  });

  return {
    systemInstruction:
      parsed.system.length > 0 ? parsed.system.join('\n\n') : undefined,
    contents,
  };
};

export const resolveChatModel = (provider: string, model: string): string => {
  if (provider === 'claude' && model === 'gpt-4o-mini') {
    return 'claude-sonnet-4-5';
  }

  if (provider === 'gemini' && model === 'gpt-4o-mini') {
    return 'gemini-2.5-flash';
  }

  return model;
};

export const resolveStreamModel = (provider: string, model: string): string => {
  if (provider === 'claude' && model === 'gpt-4o') {
    return 'claude-sonnet-4-5';
  }

  if (provider === 'gemini' && model === 'gpt-4o') {
    return 'gemini-2.5-flash';
  }

  return model;
};

export const resolveEmbeddingModel = (
  provider: string,
  model: string,
): string => {
  if (provider === 'gemini' && model === 'text-embedding-3-small') {
    return 'text-embedding-004';
  }

  return model;
};

export const normalizeGeminiEmbeddings = (
  response: GeminiEmbedContentResponse,
  model: string,
): GeminiOpenAICompatibleEmbedding => {
  const data: Embedding[] = (response.embeddings ?? []).map(
    (embedding, index): Embedding => ({
      object: 'embedding',
      embedding: embedding.values ?? [],
      index,
    }),
  );

  return {
    object: 'list',
    model,
    data,
  };
};

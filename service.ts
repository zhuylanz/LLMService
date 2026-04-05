import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI } from '@google/genai';
import OpenAI from 'openai';
import type { TranscriptionCreateParams } from 'openai/resources/audio/transcriptions';
import { buildErrorResponse, unsupportedOperationResponse } from './errors';
import {
  normalizeGeminiEmbeddings,
  resolveChatModel,
  resolveEmbeddingModel,
  resolveStreamModel,
  toClaudeMessages,
  toGeminiMessages,
  toOpenAIMessages,
} from './mappers';
import type {
  LLMChatCompletionData,
  LLMChatStream,
  LLMEmbeddingData,
  LLMImageData,
  LLMMessage,
  LLMModerationData,
  LLMProvider,
  LLMServiceOptions,
  LLMServiceResponse,
  LLMTranscriptionData,
} from './types';

export class LLMService {
  readonly provider: LLMProvider;
  private readonly apiKey: string;
  private readonly organization?: string;
  private readonly baseURL?: string;
  private readonly timeout: number;
  private readonly maxRetries: number;
  private readonly apiVersion?: string;
  private readonly vertexAI: boolean;
  private readonly project?: string;
  private readonly location?: string;

  private readonly openAIClient?: OpenAI;
  private readonly anthropicClient?: Anthropic;
  private readonly geminiClient?: GoogleGenAI;

  constructor(
    configOrApiKey: string | LLMServiceOptions,
    organization?: string,
  ) {
    const config = this.normalizeConfig(configOrApiKey, organization);

    this.provider = config.provider;
    this.apiKey = config.apiKey;
    this.organization = config.organization;
    this.baseURL = config.baseURL;
    this.timeout = config.timeout;
    this.maxRetries = config.maxRetries;
    this.apiVersion = config.apiVersion;
    this.vertexAI = config.vertexAI;
    this.project = config.project;
    this.location = config.location;

    if (this.provider === 'openai') {
      this.openAIClient = new OpenAI({
        apiKey: this.apiKey,
        organization: this.organization,
        baseURL: this.baseURL,
        timeout: this.timeout,
        maxRetries: this.maxRetries,
      });
    }

    if (this.provider === 'claude') {
      this.anthropicClient = new Anthropic({
        apiKey: this.apiKey,
        baseURL: this.baseURL,
        timeout: this.timeout,
        maxRetries: this.maxRetries,
      });
    }

    if (this.provider === 'gemini') {
      this.geminiClient = new GoogleGenAI({
        apiKey: this.vertexAI ? undefined : this.apiKey,
        vertexai: this.vertexAI,
        project: this.project,
        location: this.location,
        apiVersion: this.apiVersion,
        httpOptions: {
          baseUrl: this.baseURL,
          timeout: this.timeout,
        },
      });
    }
  }

  private normalizeConfig(
    configOrApiKey: string | LLMServiceOptions,
    organization?: string,
  ): Required<
    Pick<
      LLMServiceOptions,
      'provider' | 'apiKey' | 'timeout' | 'maxRetries' | 'vertexAI'
    >
  > &
    Omit<
      LLMServiceOptions,
      'provider' | 'apiKey' | 'timeout' | 'maxRetries' | 'vertexAI'
    > {
    if (typeof configOrApiKey === 'string') {
      return {
        provider: 'openai',
        apiKey: configOrApiKey,
        organization,
        timeout: 30_000,
        maxRetries: 3,
        vertexAI: false,
      };
    }

    return {
      provider: configOrApiKey.provider ?? 'openai',
      apiKey: configOrApiKey.apiKey,
      organization: configOrApiKey.organization,
      baseURL: configOrApiKey.baseURL,
      timeout: configOrApiKey.timeout ?? 30_000,
      maxRetries: configOrApiKey.maxRetries ?? 3,
      apiVersion: configOrApiKey.apiVersion,
      vertexAI: configOrApiKey.vertexAI ?? false,
      project: configOrApiKey.project,
      location: configOrApiKey.location,
    };
  }

  private ensureOpenAIClient(): OpenAI {
    if (!this.openAIClient) {
      throw new Error('OpenAI client is not initialized for this provider');
    }
    return this.openAIClient;
  }

  private ensureAnthropicClient(): Anthropic {
    if (!this.anthropicClient) {
      throw new Error('Anthropic client is not initialized for this provider');
    }
    return this.anthropicClient;
  }

  private ensureGeminiClient(): GoogleGenAI {
    if (!this.geminiClient) {
      throw new Error('Gemini client is not initialized for this provider');
    }
    return this.geminiClient;
  }

  async createChatCompletion(
    messages: LLMMessage[],
    model: string = 'gpt-4o-mini',
    temperature: number = 1,
    maxTokens?: number,
  ): Promise<LLMServiceResponse<LLMChatCompletionData>> {
    try {
      const resolvedModel = resolveChatModel(this.provider, model);

      if (this.provider === 'openai') {
        const client = this.ensureOpenAIClient();
        const formattedMessages = toOpenAIMessages(messages);

        const response = await client.chat.completions.create({
          model: resolvedModel,
          messages: formattedMessages,
          temperature,
          max_tokens: maxTokens,
        });

        return {
          success: true,
          data: response,
          content: response.choices[0]?.message.content ?? '',
        };
      }

      if (this.provider === 'claude') {
        const client = this.ensureAnthropicClient();
        const payload = toClaudeMessages(messages);

        const response = await client.messages.create({
          model: resolvedModel,
          messages: payload.messages,
          system: payload.system,
          temperature,
          max_tokens: maxTokens ?? 1024,
        });

        const content = response.content
          .filter((block) => block.type === 'text')
          .map((block) => block.text)
          .join('');

        return {
          success: true,
          data: response,
          content,
        };
      }

      const client = this.ensureGeminiClient();
      const payload = toGeminiMessages(messages);

      const response = await client.models.generateContent({
        model: resolvedModel,
        contents: payload.contents,
        config: {
          systemInstruction: payload.systemInstruction,
          temperature,
          maxOutputTokens: maxTokens,
        },
      });

      return {
        success: true,
        data: response,
        content: response.text ?? '',
      };
    } catch (error) {
      return buildErrorResponse(error);
    }
  }

  async createChatCompletionStream(
    messages: LLMMessage[],
    model: string = 'gpt-4o',
    temperature: number = 1,
    maxTokens?: number,
  ): Promise<LLMServiceResponse<LLMChatStream>> {
    try {
      const resolvedModel = resolveStreamModel(this.provider, model);

      if (this.provider === 'openai') {
        const client = this.ensureOpenAIClient();
        const formattedMessages = toOpenAIMessages(messages);

        const stream = await client.chat.completions.create({
          model: resolvedModel,
          messages: formattedMessages,
          temperature,
          max_tokens: maxTokens,
          stream: true,
        });

        return {
          success: true,
          data: stream,
          stream,
        };
      }

      if (this.provider === 'claude') {
        const client = this.ensureAnthropicClient();
        const payload = toClaudeMessages(messages);

        const stream = await client.messages.create({
          model: resolvedModel,
          messages: payload.messages,
          system: payload.system,
          temperature,
          max_tokens: maxTokens ?? 1024,
          stream: true,
        });

        return {
          success: true,
          data: stream,
          stream,
        };
      }

      const client = this.ensureGeminiClient();
      const payload = toGeminiMessages(messages);

      const stream = await client.models.generateContentStream({
        model: resolvedModel,
        contents: payload.contents,
        config: {
          systemInstruction: payload.systemInstruction,
          temperature,
          maxOutputTokens: maxTokens,
        },
      });

      return {
        success: true,
        data: stream,
        stream,
      };
    } catch (error) {
      return buildErrorResponse(error);
    }
  }

  async generateImage(
    prompt: string,
    n: number = 1,
    size:
      | '1024x1024'
      | '512x512'
      | '256x256'
      | '1792x1024'
      | '1024x1792' = '1024x1024',
    quality: 'standard' | 'hd' = 'standard',
  ): Promise<LLMServiceResponse<LLMImageData>> {
    try {
      if (this.provider !== 'openai') {
        return unsupportedOperationResponse(this.provider, 'generateImage');
      }

      const client = this.ensureOpenAIClient();

      const response = await client.images.generate({
        prompt,
        n,
        size,
        quality,
      });

      return {
        success: true,
        data: response,
        images: response.data,
      };
    } catch (error) {
      return buildErrorResponse(error);
    }
  }

  async createEmbeddings(
    input: string | string[],
    model: string = 'text-embedding-3-small',
  ): Promise<LLMServiceResponse<LLMEmbeddingData>> {
    try {
      const resolvedModel = resolveEmbeddingModel(this.provider, model);

      if (this.provider === 'openai') {
        const client = this.ensureOpenAIClient();

        const response = await client.embeddings.create({
          model: resolvedModel,
          input,
        });

        return {
          success: true,
          data: response,
        };
      }

      if (this.provider === 'claude') {
        return unsupportedOperationResponse(this.provider, 'createEmbeddings');
      }

      const client = this.ensureGeminiClient();
      const values = Array.isArray(input) ? input : [input];

      const response = await client.models.embedContent({
        model: resolvedModel,
        contents: values,
      });

      return {
        success: true,
        data: normalizeGeminiEmbeddings(response, resolvedModel),
      };
    } catch (error) {
      return buildErrorResponse(error);
    }
  }

  async createModeration(
    input: string | string[],
  ): Promise<LLMServiceResponse<LLMModerationData>> {
    try {
      if (this.provider !== 'openai') {
        return unsupportedOperationResponse(this.provider, 'createModeration');
      }

      const client = this.ensureOpenAIClient();

      const response = await client.moderations.create({
        input,
      });

      return {
        success: true,
        data: response,
        results: response.results,
      };
    } catch (error) {
      return buildErrorResponse(error);
    }
  }

  async createTranscription(
    audioFilePath: string,
    options?: {
      model?: string;
      language?: string;
      prompt?: string;
      responseFormat?: 'json' | 'text' | 'srt' | 'verbose_json' | 'vtt';
      temperature?: number;
    },
  ): Promise<LLMServiceResponse<LLMTranscriptionData>> {
    try {
      if (this.provider !== 'openai') {
        return unsupportedOperationResponse(
          this.provider,
          'createTranscription',
        );
      }

      const client = this.ensureOpenAIClient();
      const fs = await import('fs');
      const file = fs.createReadStream(audioFilePath);

      const request: TranscriptionCreateParams = {
        file,
        model: options?.model ?? 'whisper-1',
        language: options?.language,
        prompt: options?.prompt,
        response_format: options?.responseFormat,
        temperature: options?.temperature,
      };

      const response = await client.audio.transcriptions.create(request);

      return {
        success: true,
        data: response,
        content:
          typeof response === 'object' && 'text' in response
            ? response.text
            : String(response),
      };
    } catch (error) {
      return buildErrorResponse(error);
    }
  }
}

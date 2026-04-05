import Anthropic from '@anthropic-ai/sdk';
import { ApiError as GeminiApiError } from '@google/genai';
import OpenAI from 'openai';
import type { LLMServiceResponse } from './types';

export const extractRequestId = (
  headers: Headers | undefined,
): string | undefined => {
  if (!headers) {
    return undefined;
  }

  return headers.get('request-id') ?? headers.get('x-request-id') ?? undefined;
};

export const buildErrorResponse = (
  error: unknown,
): LLMServiceResponse<never> => {
  if (error instanceof OpenAI.APIError) {
    return {
      success: false,
      error: {
        status: error.status,
        type: error.name,
        message: error.message,
        requestId: error.requestID ?? undefined,
      },
    };
  }

  if (error instanceof Anthropic.APIError) {
    return {
      success: false,
      error: {
        status: error.status,
        type: error.name,
        message: error.message,
        requestId: error.requestID ?? extractRequestId(error.headers),
      },
    };
  }

  if (error instanceof GeminiApiError) {
    return {
      success: false,
      error: {
        status: error.status,
        type: error.name,
        message: error.message,
      },
    };
  }

  if (error instanceof Error) {
    return {
      success: false,
      error: {
        type: error.name,
        message: error.message,
      },
    };
  }

  return {
    success: false,
    error: {
      message: 'Unknown error',
    },
  };
};

export const unsupportedOperationResponse = (
  provider: string,
  operation: string,
): LLMServiceResponse<never> => {
  return {
    success: false,
    error: {
      message: `${operation} is not supported for provider: ${provider}`,
    },
  };
};

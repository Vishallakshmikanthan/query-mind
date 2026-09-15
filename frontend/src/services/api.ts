import { SSEEvent } from '../types';

const AUTH_TOKEN_KEY = 'dataflow_auth_token';

export interface StreamChatParams {
  message: string;
  sessionId: string;
  showSql?: boolean;
  onEvent: (event: SSEEvent) => void;
  onError: (error: Error) => void;
  signal?: AbortSignal;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

/**
 * Return the stored auth token, if any.
 */
export function getAuthToken(): string | null {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  } catch {
    return null;
  }
}

/**
 * Store an auth token in localStorage.
 */
export function setAuthToken(token: string | null): void {
  try {
    if (token) {
      localStorage.setItem(AUTH_TOKEN_KEY, token);
    } else {
      localStorage.removeItem(AUTH_TOKEN_KEY);
    }
  } catch {
    // ignore storage errors
  }
}

/**
 * Build headers for authenticated API requests.
 */
function getAuthHeaders(): Record<string, string> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Wrapper around fetch that injects the stored auth token.
 */
export async function authenticatedFetch(
  input: RequestInfo,
  init: RequestInit = {}
): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const authHeaders = getAuthHeaders();
  const isFormData = typeof FormData !== 'undefined' && init.body instanceof FormData;

  Object.entries(authHeaders).forEach(([key, value]) => {
    if (key.toLowerCase() === 'content-type' && isFormData) {
      // Do not set Content-Type for FormData; browser sets multipart/form-data with boundary
      return;
    }
    if (!headers.has(key)) {
      headers.set(key, value);
    }
  });
  return fetch(input, { ...init, headers });
}

/**
 * Exchange username/password for an access token.
 */
export async function login(credentials: LoginCredentials): Promise<LoginResponse> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
  if (!response.ok) {
    const errorJson = await response.json().catch(() => ({}));
    throw new Error(errorJson?.detail?.message || `Login failed: ${response.status}`);
  }
  const data: LoginResponse = await response.json();
  setAuthToken(data.access_token);
  return data;
}

/**
 * Clear the stored auth token.
 */
export function logout(): void {
  setAuthToken(null);
}

export async function streamChat({
  message,
  sessionId,
  showSql = true,
  onEvent,
  onError,
  signal,
}: StreamChatParams): Promise<void> {
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        message,
        session_id: sessionId,
        options: {
          show_sql: showSql,
          stream: true,
        },
      }),
      signal,
    });

    if (!response.ok) {
      const errorJson = await response.json().catch(() => ({}));
      const errorMsg = errorJson?.detail?.message || `HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMsg);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || ''; // keep remaining incomplete buffer segment

      for (const block of lines) {
        const trimmed = block.trim();
        if (!trimmed) continue;

        const dataLines = trimmed.split('\n');
        for (const line of dataLines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (jsonStr) {
              try {
                const event: SSEEvent = JSON.parse(jsonStr);
                onEvent(event);
              } catch (err) {
                console.warn('Failed to parse SSE JSON:', jsonStr, err);
              }
            }
          }
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('Stream request aborted');
      return;
    }
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

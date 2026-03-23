/**
 * Parses the streaming plain-text response from the ECEasy backend.
 *
 * The backend emits one continuous stream in three phases, delimited by
 * sentinel strings:
 *
 *   Phase 1 – Sources JSON array
 *   \n\n__LLM_RESPONSE__\n\n
 *   Phase 2 – LLM answer text (streamed token-by-token, may contain [[citation:N]])
 *   \n\n__RELATED_QUESTIONS__\n\n
 *   Phase 3 – Related questions JSON array  (appended once the LLM finishes)
 */

export interface Source {
  id: string;
  name: string;
  url: string;
  snippet: string;
  // Optional fields that may be present
  isFamilyFriendly?: boolean;
  displayUrl?: string;
  deepLinks?: { snippet: string; name: string; url: string }[];
  dateLastCrawled?: string;
  cachedPageUrl?: string;
  language?: string;
  primaryImageOfPage?: {
    thumbnailUrl: string;
    width: number;
    height: number;
    imageId: string;
  };
  isNavigational?: boolean;
}

export interface Relate {
  question: string;
}

export interface LlmRuntimeConfig {
  llmProvider?: 'openai' | 'deepseek';
  apiKey?: string;
  useServerKey?: boolean;
}

const LLM_SPLIT = '__LLM_RESPONSE__';
const RELATED_SPLIT = '__RELATED_QUESTIONS__';

/**
 * Converts raw LLM markdown text with [[citation:N]] tokens into
 * markdown links [citation](N) that the ChatMessage renderer can process.
 */
export function markdownParse(text: string): string {
  return text
    .replace(/\[\[([cC])itation/g, '[citation')
    .replace(/[cC]itation:(\d+)]]/g, 'citation:$1]')
    .replace(/\[\[([cC]itation:\d+)]](?!])/g, `[$1]`)
    .replace(/\[[cC]itation:(\d+)]/g, '[citation]($1)');
}

export async function parseStreaming(
  controller: AbortController,
  query: string,
  searchUuid: string,
  llmConfig: LlmRuntimeConfig,
  onSources: (sources: Source[]) => void,
  onMarkdown: (markdown: string) => void,
  onRelates: (relates: Relate[]) => void,
  onError?: (status: number) => void,
): Promise<void> {
  const response = await fetch('/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: '*/*',
    },
    signal: controller.signal,
    body: JSON.stringify({
      query,
      search_uuid: searchUuid,
      llm_provider: llmConfig.llmProvider,
      api_key: llmConfig.apiKey,
      use_server_key: llmConfig.useServerKey,
    }),
  });

  if (response.status !== 200) {
    onError?.(response.status);
    return;
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let chunks = '';
  let sourcesEmitted = false;

  const updateMarkdown = (raw: string) => {
    if (raw.includes(RELATED_SPLIT)) {
      const [md] = raw.split(RELATED_SPLIT);
      onMarkdown(markdownParse(md));
    } else {
      onMarkdown(markdownParse(raw));
    }
  };

  while (true) {
    const { done, value } = await reader.read();

    if (value) {
      chunks += decoder.decode(value, { stream: !done });
    }

    if (chunks.includes(LLM_SPLIT)) {
      const [sourcesPart, rest] = chunks.split(LLM_SPLIT);

      if (!sourcesEmitted) {
        try {
          onSources(JSON.parse(sourcesPart.trim()));
        } catch {
          onSources([]);
        }
        sourcesEmitted = true;
      }

      updateMarkdown(rest);
    }

    if (done) break;
  }

  // Parse related questions from the end of the accumulated stream
  if (chunks.includes(RELATED_SPLIT)) {
    const parts = chunks.split(RELATED_SPLIT);
    const relatesJson = parts[parts.length - 1];
    try {
      onRelates(JSON.parse(relatesJson.trim()));
    } catch {
      onRelates([]);
    }
  } else {
    onRelates([]);
  }
}


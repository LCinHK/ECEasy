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

export interface SuggestedImage {
  path: string;
  description: string;
  doc_type: string;
  source_relpath: string;
}

export interface LlmRuntimeConfig {
  llmProvider?: 'openai' | 'deepseek';
  apiKey?: string;
  useServerKey?: boolean;
  llmModel?: string;
}

export interface ParsedStreamPayload {
  sources: Source[];
  markdown: string;
  relates: Relate[];
  suggestedImages: SuggestedImage[];
}

const LLM_SPLIT = '__LLM_RESPONSE__';
const RELATED_SPLIT = '__RELATED_QUESTIONS__';
const IMAGES_SPLIT = '__SUGGESTED_IMAGES__';

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

function safeJsonParse<T>(raw: string, fallback: T): T {
  try {
    return JSON.parse(raw.trim()) as T;
  } catch {
    return fallback;
  }
}

function normalizeRawStreamPayload(raw: string): string {
  const hasEscapedMarkers =
    raw.includes(`\\n\\n${LLM_SPLIT}\\n\\n`) ||
    raw.includes(`\\n\\n${RELATED_SPLIT}\\n\\n`) ||
    raw.includes(`\\n\\n${IMAGES_SPLIT}\\n\\n`);

  if (!hasEscapedMarkers) {
    return raw;
  }

  return raw
    .replace(/\\r\\n/g, '\n')
    .replace(/\\n/g, '\n')
    .replace(/\\t/g, '\t');
}

export function parseStreamPayload(raw: string): ParsedStreamPayload {
  const normalizedRaw = normalizeRawStreamPayload(raw);
  const llmIndex = normalizedRaw.indexOf(LLM_SPLIT);
  if (llmIndex < 0) {
    return {
      sources: [],
      markdown: markdownParse(normalizedRaw),
      relates: [],
      suggestedImages: [],
    };
  }

  const sourcesPart = normalizedRaw.slice(0, llmIndex);
  const llmPart = normalizedRaw.slice(llmIndex + LLM_SPLIT.length);

  const relatedIndex = llmPart.lastIndexOf(RELATED_SPLIT);
  const imagesIndex = llmPart.lastIndexOf(IMAGES_SPLIT);

  const markdownEndCandidates = [relatedIndex, imagesIndex].filter((i) => i >= 0);
  const markdownEnd = markdownEndCandidates.length > 0 ? Math.min(...markdownEndCandidates) : llmPart.length;
  const markdown = markdownParse(llmPart.slice(0, markdownEnd));

  let relates: Relate[] = [];
  if (relatedIndex >= 0) {
    const relatedStart = relatedIndex + RELATED_SPLIT.length;
    const relatedEnd = imagesIndex >= 0 && imagesIndex > relatedIndex ? imagesIndex : llmPart.length;
    relates = safeJsonParse<Relate[]>(llmPart.slice(relatedStart, relatedEnd), []);
  }

  let suggestedImages: SuggestedImage[] = [];
  if (imagesIndex >= 0) {
    const imagesStart = imagesIndex + IMAGES_SPLIT.length;
    suggestedImages = safeJsonParse<SuggestedImage[]>(llmPart.slice(imagesStart), []);
  }

  return {
    sources: safeJsonParse<Source[]>(sourcesPart, []),
    markdown,
    relates,
    suggestedImages,
  };
}

export async function parseStreaming(
  controller: AbortController,
  query: string,
  searchUuid: string,
  llmConfig: LlmRuntimeConfig,
  onSources: (sources: Source[]) => void,
  onMarkdown: (markdown: string) => void,
  onRelates: (relates: Relate[]) => void,
  onSuggestedImages: (images: SuggestedImage[]) => void,
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
      llm_model: llmConfig.llmModel,
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
    const cutMarkers = [RELATED_SPLIT, IMAGES_SPLIT]
      .map((m) => raw.indexOf(m))
      .filter((idx) => idx >= 0);

    if (cutMarkers.length > 0) {
      const md = raw.slice(0, Math.min(...cutMarkers));
      onMarkdown(markdownParse(md));
      return;
    }

    onMarkdown(markdownParse(raw));
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

  const finalParsed = parseStreamPayload(chunks);
  if (!sourcesEmitted) {
    onSources(finalParsed.sources);
  }
  onRelates(finalParsed.relates);
  onSuggestedImages(finalParsed.suggestedImages);
}


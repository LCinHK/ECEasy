const COMMON_ENTITIES: Record<string, string> = {
  '&lt;': '<',
  '&gt;': '>',
  '&amp;': '&',
  '&quot;': '"',
  '&apos;': "'",
};

function decodeEntitiesOnce(text: string): string {
  if (typeof document !== 'undefined') {
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    return textarea.value;
  }

  return text.replace(/&(lt|gt|amp|quot|apos);/g, (match) => COMMON_ENTITIES[match] ?? match).replace(
    /&#(\d+);|&#x([a-fA-F0-9]+);/g,
    (match, dec, hex) => {
      const codePoint = dec ? Number.parseInt(dec, 10) : Number.parseInt(hex, 16);
      if (!Number.isFinite(codePoint)) return match;
      try {
        return String.fromCodePoint(codePoint);
      } catch {
        return match;
      }
    },
  );
}

function decodeHtmlEntities(text: string, maxPasses = 4): string {
  let current = text;

  for (let i = 0; i < maxPasses; i += 1) {
    const next = decodeEntitiesOnce(current);
    if (next === current) break;
    current = next;
  }

  return current;
}

function convertSimpleHtmlBlocksToMarkdown(text: string): string {
  return text
    .replace(/<\s*h([1-6])\b[^>]*>([\s\S]*?)<\s*\/\s*h\1\s*>/gi, (_match, level, inner) => {
      const heading = String(inner).replace(/\s+/g, ' ').trim();
      return heading ? `\n${'#'.repeat(Number(level))} ${heading}\n` : '\n';
    })
    .replace(/<\s*\/\s*h[1-6]\s*>/gi, '\n')
    .replace(/<\s*ul\b[^>]*>/gi, '\n')
    .replace(/<\s*\/\s*ul\s*>/gi, '\n')
    .replace(/<\s*ol\b[^>]*>/gi, '\n')
    .replace(/<\s*\/\s*ol\s*>/gi, '\n')
    .replace(/<\s*li\b[^>]*>/gi, '\n- ')
    .replace(/<\s*\/\s*li\s*>/gi, '\n')
    .replace(/<\s*(br|p|div|section|article)\b[^>]*>/gi, '\n')
    .replace(/<\s*\/\s*(p|div|section|article)\s*>/gi, '\n')
    .replace(/<\s*(strong|b)\b[^>]*>/gi, '**')
    .replace(/<\s*\/\s*(strong|b)\s*>/gi, '**')
    .replace(/<\s*(em|i)\b[^>]*>/gi, '*')
    .replace(/<\s*\/\s*(em|i)\s*>/gi, '*')
    .replace(/<\s*\/??\s*(tbody|thead|table|tr|td|th|span|font|section|article|div)\b[^>]*>/gi, ' ')
    .replace(/<(?!\/?(?:details|summary)\b)[^>]+>/gi, '');
}

function normalizeWhitespace(text: string): string {
  return text
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+\n/g, '\n')
    .trim();
}

function normalizeLatexContent(math: string): string {
  // Some providers escape LaTeX commands as \\alpha in plain text streams.
  return math.replace(/\\\\([A-Za-z]+)/g, '\\$1').trim();
}

function normalizeLatexDelimiters(text: string): string {
  let normalized = text;

  // Support TeX-style delimiters from streamed output: \( ... \) and \[ ... \].
  normalized = normalized.replace(/\\\[([\s\S]*?)\\]/g, (_match, inner: string) => {
    const content = normalizeLatexContent(inner);
    return content ? `\n$$\n${content}\n$$\n` : _match;
  });

  normalized = normalized.replace(/\\\(([\s\S]*?)\\\)/g, (_match, inner: string) => {
    const content = normalizeLatexContent(inner);
    return content ? `$${content}$` : _match;
  });

  return normalized;
}

export function normalizeAssistantContent(content: string): string {
  const decoded = decodeHtmlEntities(content);
  const withNormalizedLatex = normalizeLatexDelimiters(decoded);
  const withChainOfThought = withNormalizedLatex
    .replace(/<think>/g, '<details><summary>=== Chain of Thought ===</summary>')
    .replace(/<\/think>/g, '</details>');

  const htmlishPattern = /<\s*(ul|ol|li|h[1-6]|p|br|strong|b|em|i)\b/i;
  const normalized = htmlishPattern.test(withChainOfThought)
    ? convertSimpleHtmlBlocksToMarkdown(withChainOfThought)
    : withChainOfThought;

  return normalizeWhitespace(normalized);
}



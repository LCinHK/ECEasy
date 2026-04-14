import { useEffect, useId, useRef, useState } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: false,
  theme: 'default',
  suppressErrorRendering: true,
  securityLevel: 'loose',
});

interface MermaidProps {
  chart: string;
}

function sanitizeMermaid(source: string): string {
  let out = source.replace(/[\u2010-\u2015\u2212]/g, '-');

  const needsQuoting = /[()&;#,|><{}\[\]`]/;

  const isQuoted = (text: string): boolean => {
    const trimmed = text.trim();
    return trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"');
  };

  const normalizeLabelLineBreaks = (text: string): string => text.replace(/\\r?\\n/g, '<br/>').replace(/\r?\n/g, '<br/>');

  const sanitizeLabel = (raw: string): string => {
    const trimmed = raw.trim();
    if (!trimmed) return raw;

    // Preserve quoted labels, only normalize line-break markers inside.
    if (isQuoted(trimmed)) {
      const inner = trimmed.slice(1, -1);
      return `"${normalizeLabelLineBreaks(inner)}"`;
    }

    const withBreaks = normalizeLabelLineBreaks(trimmed);
    if (needsQuoting.test(withBreaks) || withBreaks.includes('<br/>')) {
      return `"${withBreaks.replace(/"/g, '\\"')}"`;
    }

    return withBreaks;
  };

  const isNodeLabelStart = (text: string, idx: number): boolean => {
    if (idx <= 0) return false;
    const prev = text[idx - 1];
    return /[A-Za-z0-9_\])}]/.test(prev);
  };

  const replacePair = (input: string, open: string, close: string): string => {
    let result = '';
    let i = 0;

    while (i < input.length) {
      if (input.startsWith(open, i) && isNodeLabelStart(input, i)) {
        const start = i + open.length;
        let j = start;
        let depth = 1;
        let inQuote = false;
        let escaped = false;

        while (j < input.length) {
          const ch = input[j];

          if (inQuote) {
            if (escaped) {
              escaped = false;
            } else if (ch === '\\') {
              escaped = true;
            } else if (ch === '"') {
              inQuote = false;
            }
            j += 1;
            continue;
          }

          if (ch === '"') {
            inQuote = true;
            j += 1;
            continue;
          }

          if (input.startsWith(open, j)) {
            depth += 1;
            j += open.length;
            continue;
          }

          if (input.startsWith(close, j)) {
            depth -= 1;
            if (depth === 0) break;
            j += close.length;
            continue;
          }

          j += 1;
        }

        if (j < input.length) {
          const inner = input.slice(start, j);
          result += `${open}${sanitizeLabel(inner)}${close}`;
          i = j + close.length;
          continue;
        }
      }

      result += input[i];
      i += 1;
    }

    return result;
  };

  // Longer delimiters first.
  out = replacePair(out, '((', '))');
  out = replacePair(out, '([', '])');
  out = replacePair(out, '{{', '}}');
  out = replacePair(out, '[', ']');
  out = replacePair(out, '(', ')');
  out = replacePair(out, '{', '}');

  // Asymmetric node: A>Label]
  out = out.replace(/([A-Za-z0-9_]+)>([^\[\]]*?)]/g, (_m, node, rawLabel) => `${node}>${sanitizeLabel(rawLabel)}]`);

  return out;
}

// Conservative fallback sanitizer used only if raw+primary sanitizer fail.
// It focuses on common flowchart node shapes and avoids deep structural rewrites.
function sanitizeMermaidFallback(source: string): string {
  const normalizeLabel = (raw: string): string => {
    const trimmed = raw.trim();
    if (!trimmed) return raw;

    const normalizeBreaks = (text: string) => text.replace(/\\r?\\n/g, '<br/>').replace(/\r?\n/g, '<br/>');
    const isQuoted = trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"');
    const body = isQuoted ? trimmed.slice(1, -1) : trimmed;
    return `"${normalizeBreaks(body).replace(/"/g, '\\"')}"`;
  };

  return source
    .replace(/[\u2010-\u2015\u2212]/g, '-')
    .replace(/\[([^\]\n]*)]/g, (_m, inner) => `[${normalizeLabel(inner)}]`)
    .replace(/\{([^}\n]*)}/g, (_m, inner) => `{${normalizeLabel(inner)}}`)
    .replace(/([A-Za-z0-9_]+)>([^\[\]\n]*?)]/g, (_m, node, inner) => `${node}>${normalizeLabel(inner)}]`);
}

export default function Mermaid({ chart }: MermaidProps) {
  const id = useId().replace(/:/g, '');   // mermaid IDs must not contain colons
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const render = async () => {
      const tryRender = async (candidate: string) => {
        const { svg, bindFunctions } = await mermaid.render(`mermaid-${id}`, candidate);
        if (cancelled || !containerRef.current) return;
        containerRef.current.innerHTML = svg;
        bindFunctions?.(containerRef.current);
        setError(null);
      };

      try {
        const raw = chart.trim();
        // 1) raw chart first; 2) primary sanitizer; 3) conservative fallback sanitizer
        try {
          await tryRender(raw);
          return;
        } catch {
          // continue to sanitizer strategies
        }

        const sanitized = sanitizeMermaid(raw);
        try {
          await tryRender(sanitized);
          return;
        } catch {
          // continue to fallback sanitizer
        }

        const fallbackSanitized = sanitizeMermaidFallback(raw);
        await tryRender(fallbackSanitized);
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message ?? 'Failed to render diagram.');
        }
      }
    };

    render();

    return () => {
      cancelled = true;
    };
  }, [chart, id]);

  if (error) {
    return (
      <pre className="text-red-500 text-xs whitespace-pre-wrap border border-red-300 rounded p-3 bg-red-50">
        {`[Mermaid render error]\n${error}`}
      </pre>
    );
  }

  return (
    <div
      ref={containerRef}
      className="my-4 flex justify-center overflow-x-auto"
      aria-label="Mermaid diagram"
    />
  );
}


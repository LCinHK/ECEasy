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

/**
 * Mermaid's parser treats bare parentheses, ampersands, and similar
 * characters as syntax tokens inside node labels.  Any label that is
 * not already wrapped in double-quotes and contains one of those
 * characters will cause a parse error.
 *
 * This function walks every node-label bracket pair and, when the
 * content is unquoted and contains a special character, wraps it in
 * double-quotes (escaping any pre-existing double-quotes inside).
 *
 * Bracket pairs handled:
 *   [text]   – rectangle
 *   (text)   – rounded rectangle  ← most common source of the bug
 *   {text}   – diamond
 *   ((text)) – circle
 *   ([text]) – stadium (cylinder)
 *   {{text}} – hexagon
 *   >text]   – asymmetric
 */
function sanitizeMermaid(source: string): string {
  // 1. Normalise Unicode look-alike hyphens/dashes to plain ASCII hyphens
  //    (e.g. U+2011 NON-BREAKING HYPHEN, U+2013 EN DASH, U+2014 EM DASH)
  let out = source.replace(/[\u2010-\u2015\u2212]/g, '-');

  // 2. Quote unquoted labels that contain special characters.
  //    We match each bracket style separately so we can handle nested
  //    delimiters correctly.
  //
  //    Pattern anatomy:
  //      (open)        – opening delimiter literal
  //      ("(?:[^"\\]|\\.)*")   – already-quoted string → leave alone
  //      |([^)\]}>]+)  – unquoted text → candidate for quoting
  //      (close)       – closing delimiter literal
  //
  //    Special chars that break Mermaid without quotes:
  //      ( ) & ; # , | > < { }
  const NEEDS_QUOTING = /[()&;#,|><{}]/;

  const quoteLabel = (label: string): string => {
    // Already quoted — strip outer quotes, escape inner ones, re-quote
    if (label.startsWith('"') && label.endsWith('"')) return label;
    if (NEEDS_QUOTING.test(label)) {
      const escaped = label.replace(/"/g, '\\"');
      return `"${escaped}"`;
    }
    return label;
  };

  // Helper: replace label inside a bracket pair
  const replacePair = (open: string, close: string): void => {
    // Escape special regex chars in the delimiter strings
    const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`${esc(open)}((?:"(?:[^"\\\\]|\\\\.)*"|[^${esc(close[0])}])*)${esc(close)}`, 'g');
    out = out.replace(re, (_match, inner) => `${open}${quoteLabel(inner)}${close}`);
  };

  // Order matters: longer/compound delimiters first
  replacePair('((',  '))');
  replacePair('([',  '])');
  replacePair('{{',  '}}');
  replacePair('[',   ']');
  replacePair('(',   ')');
  replacePair('{',   '}');

  // >text]  asymmetric node — inner text ends before ]
  // >text]  asymmetric node — use RegExp constructor to avoid linter noise on the ] escape
  out = out.replace(new RegExp('>([^\\[\\]]*)]', 'g'), (_match, inner) => `>${quoteLabel(inner)}]`);

  return out;
}

export default function Mermaid({ chart }: MermaidProps) {
  const id = useId().replace(/:/g, '');   // mermaid IDs must not contain colons
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const render = async () => {
      try {
        const sanitized = sanitizeMermaid(chart.trim());
        // mermaid.render() returns { svg, bindFunctions }
        const { svg, bindFunctions } = await mermaid.render(`mermaid-${id}`, sanitized);
        if (cancelled || !containerRef.current) return;
        containerRef.current.innerHTML = svg;
        bindFunctions?.(containerRef.current);
        setError(null);
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


import { useEffect, useState } from 'react';
import { Bot, User, BookText, MessageSquareQuote, ExternalLink, Image as ImageIcon } from 'lucide-react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/app/components/ui/popover';
import { ImageWithFallback } from '@/app/components/figma/ImageWithFallback';
import Mermaid from '@/app/components/Mermaid';
import { normalizeAssistantContent } from '@/app/utils/assistant-content';
import type { Source, Relate, SuggestedImage } from '@/app/utils/parse-streaming';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  /** Plain text for user messages; parsed markdown for assistant messages */
  content: string;
  /** RAG + web sources, present on assistant messages once stream begins */
  sources?: Source[];
  /** Related questions, present after the stream finishes */
  relates?: Relate[] | null;
  /** Suggested images, present after the stream finishes */
  suggestedImages?: SuggestedImage[];
  /** True while the LLM is still streaming this message */
  isStreaming?: boolean;
  /** Callback when the user clicks a related question chip */
  onRelatedQuestion?: (question: string) => void;
}

// ── Suggested images ──────────────────────────────────────────────────────────
function SuggestedImagesPanel({
  images,
  onOpen,
}: {
  images: SuggestedImage[];
  onOpen: (image: SuggestedImage) => void;
}) {
  if (images.length === 0) return null;

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 mb-2">
        <ImageIcon size={14} /> Suggested Images
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {images.map((img) => (
          <button
            key={img.source_relpath}
            type="button"
            onClick={() => onOpen(img)}
            className="w-full text-left bg-amber-100 hover:bg-amber-200 border border-amber-200 rounded-lg overflow-hidden transition-colors"
          >
            <ImageWithFallback
              src={img.path}
              alt={img.description}
              className="w-full h-40 object-cover"
            />
            <div className="p-2">
              <div className="text-sm font-medium text-gray-900 line-clamp-2">{img.description}</div>
              <div className="text-xs text-gray-500 mt-1">[{img.doc_type}]</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Citation popover bubble ──────────────────────────────────────────────────
function CitationBubble({ index, source }: { index: number; source: Source }) {
  const domain =
    source.url.startsWith('http://') || source.url.startsWith('https://')
      ? new URL(source.url).hostname
      : source.url;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <span
          title={source.name}
          className="inline-block cursor-pointer transform scale-75 origin-top-left font-medium bg-amber-200 hover:bg-amber-300 w-5 h-5 text-center text-xs leading-5 rounded-full align-top"
        >
          {index}
        </span>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="max-w-sm flex flex-col gap-2 bg-white shadow-lg text-xs z-50"
      >
        <div className="font-medium text-ellipsis overflow-hidden whitespace-nowrap">
          {source.name}
        </div>
        <div className="text-zinc-500 line-clamp-4 break-words">{source.snippet}</div>
        <div className="flex items-center gap-2 overflow-hidden">
          <img
            className="h-3 w-3 flex-none"
            alt={domain}
            src={`https://www.google.com/s2/favicons?domain=${domain}&sz=16`}
          />
          <a
            href={source.url}
            target="_blank"
            rel="noreferrer"
            className="text-blue-500 truncate hover:underline"
          >
            {source.url}
          </a>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// ── Sources grid ─────────────────────────────────────────────────────────────
function SourcesPanel({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 mb-2">
        <BookText size={14} /> Sources
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {sources.map((source, i) => {
          const domain =
            source.url.startsWith('http://') || source.url.startsWith('https://')
              ? new URL(source.url).hostname
              : source.url;
          const displayName = source.name.includes(', ..\\localData')
            ? source.name.replace(/,.*\\localData\\.*\\/, ', ')
            : source.name;
          return (
            <div
              key={source.id ?? i}
              className="relative text-xs py-3 px-3 bg-amber-100 hover:bg-amber-200 rounded-lg flex flex-col gap-1"
            >
              <a href={source.url} target="_blank" rel="noreferrer" className="absolute inset-0" />
              <div className="font-medium text-zinc-900 truncate">{displayName}</div>
              <div className="flex items-center gap-1 text-zinc-400 overflow-hidden">
                <img
                  className="h-3 w-3 flex-none"
                  alt={domain}
                  src={`https://www.google.com/s2/favicons?domain=${domain}&sz=16`}
                />
                <span className="truncate">{i + 1} – {domain}</span>
                <ExternalLink size={10} className="flex-none" />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Related questions ─────────────────────────────────────────────────────────
function RelatedPanel({
  relates,
  onSelect,
}: {
  relates: Relate[];
  onSelect?: (q: string) => void;
}) {
  if (relates.length === 0) return null;
  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 mb-2">
        <MessageSquareQuote size={14} /> Related Questions
      </div>
      <div className="flex flex-col gap-1">
        {relates.map(({ question }) => (
          <button
            key={question}
            onClick={() => onSelect?.(question)}
            className="text-left text-sm px-3 py-2 rounded-lg bg-amber-100 hover:bg-amber-200 text-gray-800 transition-colors"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export function ChatMessage({
  role,
  content,
  sources = [],
  relates,
  suggestedImages = [],
  isStreaming = false,
  onRelatedQuestion,
}: ChatMessageProps) {
  const isUser = role === 'user';
  const [expandedImage, setExpandedImage] = useState<SuggestedImage | null>(null);
  const renderedContent = isUser ? content : normalizeAssistantContent(content);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setExpandedImage(null);
      }
    };

    if (expandedImage) {
      window.addEventListener('keydown', onKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [expandedImage]);

  return (
    <div className={`flex gap-4 px-4 py-6 ${isUser ? 'bg-amber-100' : 'bg-amber-50'}`}>
      <div className="max-w-4xl mx-auto w-full flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              isUser
                ? 'bg-gradient-to-br from-amber-500 to-orange-500 text-white'
                : 'bg-gradient-to-br from-amber-500 to-yellow-500 text-white'
            }`}
          >
            {isUser ? <User size={18} /> : <Bot size={18} />}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 pt-1">
          {isUser ? (
            // User messages: plain pre-wrapped text
            <p className="text-gray-900 leading-7 whitespace-pre-wrap">{content}</p>
          ) : (
            <>
              {/* Assistant messages: rich Markdown with citations */}
              <div className="prose prose-sm max-w-none prose-headings:text-slate-900 prose-p:text-gray-900 prose-li:text-gray-900">
                <Markdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeRaw, rehypeKatex]}
                  components={{
                    // Render citation links as inline bubble popovers
                    a: ({ href, children, ...props }: any) => {
                      const idx = href ? parseInt(href, 10) : NaN;
                      const source = !isNaN(idx) ? sources[idx - 1] : undefined;
                      if (source) {
                        return <CitationBubble index={idx} source={source} />;
                      }
                      return (
                        <a href={href} target="_blank" rel="noreferrer" {...props}>
                          {children}
                        </a>
                      );
                    },
                    // Syntax-highlighted code blocks
                    code({ node, inline, className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || '');
                      const language = (match?.[1] ?? '').toLowerCase();

                      if (!inline && language === 'mermaid') {
                        return <Mermaid chart={String(children).replace(/\n$/, '')} />;
                      }

                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={dracula}
                          PreTag="div"
                          language={language || match[1]}
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {renderedContent}
                </Markdown>
              </div>

              {/* Streaming bounce indicator */}
              {isStreaming && (
                <div className="flex gap-1 mt-2">
                  <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              )}

              {/* Suggested images — shown only after streaming completes */}
              {!isStreaming && suggestedImages.length > 0 && (
                <SuggestedImagesPanel images={suggestedImages} onOpen={setExpandedImage} />
              )}

              {/* Sources — shown once we have them, even during streaming */}
              {sources.length > 0 && <SourcesPanel sources={sources} />}

              {/* Related questions — shown only after streaming completes */}
              {!isStreaming && relates && <RelatedPanel relates={relates} onSelect={onRelatedQuestion} />}
            </>
          )}
        </div>
      </div>

      {/* Image modal */}
      {expandedImage && (
        <div
          className="fixed inset-0 z-[100] bg-black/70 flex items-center justify-center p-4"
          onClick={() => setExpandedImage(null)}
          role="dialog"
          aria-modal="true"
          aria-label="Expanded suggested image"
        >
          <div
            className="relative w-full max-w-5xl bg-white rounded-xl overflow-hidden shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className="absolute top-3 right-3 z-10 px-3 py-1.5 text-xs rounded-md bg-black/70 text-white hover:bg-black/80"
              onClick={() => setExpandedImage(null)}
            >
              Close
            </button>
            <div className="bg-gray-50 max-h-[80vh] overflow-auto">
              <ImageWithFallback
                src={expandedImage.path}
                alt={expandedImage.description}
                className="w-full h-auto max-h-[72vh] object-contain bg-gray-100"
              />
            </div>
            <div className="p-3 border-t border-gray-200">
              <div className="text-sm font-medium text-gray-900">{expandedImage.description}</div>
              <div className="text-xs text-gray-500 mt-1">
                [{expandedImage.doc_type}] {expandedImage.source_relpath}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}



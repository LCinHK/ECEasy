import { useEffect, useMemo, useState } from 'react';
import { ChatMessage } from '@/app/components/ChatMessage';
import { parseStreamPayload } from '@/app/utils/parse-streaming';

interface FixtureFile {
  name: string;
  description?: string;
  stream: string;
}

const FIXTURE_PATH = `${import.meta.env.BASE_URL}debug/sample-stream-elec2991.json`;

export function DebugFixturePreview() {
  const [fixture, setFixture] = useState<FixtureFile | null>(null);
  const [error, setError] = useState<string>('');
  const [rawInput, setRawInput] = useState<string>('');
  const [activeRaw, setActiveRaw] = useState<string>('');

  useEffect(() => {
    let active = true;

    async function loadFixture() {
      try {
        const response = await fetch(FIXTURE_PATH, { cache: 'no-store' });
        if (!response.ok) {
          setError(`Failed to fetch fixture: HTTP ${response.status}`);
          return;
        }
        const payload = (await response.json()) as FixtureFile;
        if (!active) return;
        setFixture(payload);
        setRawInput(payload.stream);
        setActiveRaw(payload.stream);
      } catch (err) {
        if (!active) return;
        const message = err instanceof Error ? err.message : 'Unknown fixture loading error';
        setError(message);
      }
    }

    loadFixture();
    return () => {
      active = false;
    };
  }, []);

  const parsed = useMemo(() => {
    if (!activeRaw) return null;
    return parseStreamPayload(activeRaw);
  }, [activeRaw]);

  const applyPastedPayload = () => {
    const trimmed = rawInput.trim();
    if (!trimmed) {
      setError('Please paste a raw stream payload first.');
      return;
    }
    setError('');
    setActiveRaw(trimmed);
  };

  const resetToFixturePayload = () => {
    if (!fixture) return;
    setError('');
    setRawInput(fixture.stream);
    setActiveRaw(fixture.stream);
  };

  return (
    <div className="min-h-screen bg-amber-50 text-gray-900 p-4 md:p-8">
      <div className="mx-auto max-w-6xl space-y-4">
        <div className="rounded-2xl border border-amber-200 bg-white p-4 shadow-sm">
          <h1 className="text-2xl font-bold text-gray-900">ECEasy parser fixture preview</h1>
          <p className="mt-1 text-sm text-gray-600">
            Uses a backend-like raw stream fixture, runs it through <code>parseStreamPayload</code>, and renders via <code>ChatMessage</code>.
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Fixture: <code>{FIXTURE_PATH}</code>
          </p>
        </div>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>
        )}

        {!fixture && !error && (
          <div className="rounded-xl border border-amber-200 bg-white p-4 text-sm text-gray-600">Loading fixture...</div>
        )}

        {fixture && (
          <>
            <div className="rounded-xl border border-amber-200 bg-white p-4 text-sm text-gray-700">
              <div className="font-semibold">{fixture.name}</div>
              {fixture.description && <div className="mt-1 text-gray-600">{fixture.description}</div>}
              <div className="mt-2 text-xs text-gray-500">Debug mode: {activeRaw === fixture.stream ? 'fixture payload' : 'pasted payload'}</div>
            </div>

            <div className="rounded-xl border border-amber-200 bg-white p-4">
              <div className="text-sm font-semibold text-gray-800">Paste raw backend stream payload</div>
              <p className="mt-1 text-xs text-gray-600">
                Paste the exact concatenated response (`contexts + __LLM_RESPONSE__ + answer + __RELATED_QUESTIONS__ + ...`) and click Parse.
              </p>
              <textarea
                value={rawInput}
                onChange={(e) => setRawInput(e.target.value)}
                placeholder="Paste raw backend stream payload here"
                className="mt-3 min-h-48 w-full rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs font-mono text-gray-800 focus:outline-none focus:ring-2 focus:ring-amber-400"
              />
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={applyPastedPayload}
                  className="px-3 py-1.5 rounded-lg bg-amber-600 text-white text-sm hover:bg-amber-700"
                >
                  Parse pasted payload
                </button>
                <button
                  type="button"
                  onClick={resetToFixturePayload}
                  className="px-3 py-1.5 rounded-lg border border-amber-300 text-amber-900 text-sm hover:bg-amber-50"
                >
                  Reset to fixture payload
                </button>
              </div>
            </div>

            {parsed && (
              <div className="rounded-xl border border-amber-200 bg-white p-4 text-sm text-gray-700">
                <div className="mt-2 text-xs text-gray-500">
                  Parsed: {parsed.sources.length} sources, {parsed.relates.length} related questions, {parsed.suggestedImages.length} suggested images
                </div>
              </div>
            )}

            {parsed && (
              <ChatMessage
                role="assistant"
                content={parsed.markdown}
                sources={parsed.sources}
                relates={parsed.relates}
                suggestedImages={parsed.suggestedImages}
                isStreaming={false}
              />
            )}

            <details className="rounded-xl border border-amber-200 bg-white p-4">
              <summary className="cursor-pointer text-sm font-semibold text-gray-800">Show active raw stream</summary>
              <pre className="mt-3 whitespace-pre-wrap text-xs text-gray-700 bg-amber-50 rounded-lg p-3 overflow-x-auto">
                {activeRaw}
              </pre>
            </details>

            <details className="rounded-xl border border-amber-200 bg-white p-4">
              <summary className="cursor-pointer text-sm font-semibold text-gray-800">Show fixture raw stream</summary>
              <pre className="mt-3 whitespace-pre-wrap text-xs text-gray-700 bg-amber-50 rounded-lg p-3 overflow-x-auto">
                {fixture.stream}
              </pre>
            </details>
          </>
        )}
      </div>
    </div>
  );
}




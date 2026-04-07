import { ChatMessage } from '@/app/components/ChatMessage';
import type { Source, SuggestedImage } from '@/app/utils/parse-streaming';

const sampleSources: Source[] = [
  {
    id: 'sample-1',
    name: '[ELEC2991] Page 1, ELEC2991_Fall 2025-26.pdf',
    url: '/resource/ECEknowledge/course%20syllabus/ELEC/ELEC2991_Fall%202025-26.pdf',
    snippet: 'ELEC2991 is the department’s Industrial Experience (internship) course.',
  },
  {
    id: 'sample-2',
    name: '[ELEC1200] Page 1, ELEC1200_Fall 2025-26.pdf',
    url: '/resource/ECEknowledge/course%20syllabus/ELEC/ELEC1200_Fall%202025-26.pdf',
    snippet: 'The course is centered on weekly laboratories and pre-lab preparation.',
  },
];

const sampleImages: SuggestedImage[] = [
  {
    path: '/resource/ECEknowledge/Sample_ELEC_Study_Pattern.png',
    description: 'Sample ELEC study pattern',
    doc_type: 'course_syllabus',
    source_relpath: 'Sample_ELEC_Study_Pattern.png',
  },
];

const sampleAssistantContent = `### Sample parser preview\n\n&lt;ul&gt;\n&lt;li&gt;The course is centered on weekly laboratories that introduce important concepts; preparing before each lab and finishing the writeups on time will directly improve your understanding and grade [[citation:2]].&lt;/li&gt;\n&lt;li&gt;Read the lab instructions and lecture notes before the session so you spend lab time experimenting and troubleshooting, not just reading.&lt;/li&gt;\n&lt;/ul&gt;\n\n&lt;p&gt;You should also check the course syllabus for the latest grading and attendance rules [[citation:1]].&lt;/p&gt;`;

export function DebugSamplePreview() {
  return (
    <div className="min-h-screen bg-amber-50 text-gray-900 p-4 md:p-8">
      <div className="mx-auto max-w-5xl space-y-4">
        <div className="rounded-2xl border border-amber-200 bg-white p-4 shadow-sm">
          <h1 className="text-2xl font-bold text-gray-900">ECEasy parser debug preview</h1>
          <p className="mt-1 text-sm text-gray-600">
            This page feeds a sample response through the same rendering component as the chat UI.
            Use it with <code>?debugSample=1</code> in <code>npm run dev</code>.
          </p>
        </div>

        <ChatMessage
          role="assistant"
          content={sampleAssistantContent}
          sources={sampleSources}
          suggestedImages={sampleImages}
          isStreaming={false}
        />
      </div>
    </div>
  );
}


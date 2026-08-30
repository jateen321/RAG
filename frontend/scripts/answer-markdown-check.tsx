/**
 * Renders AnswerMarkdown to static HTML so the output can be eyeballed without
 * a browser. Not a unit test -- there is no test runner in this frontend -- but
 * it caught a real bug: blank lines between "1." and "2." used to emit one <ol>
 * per item, so the numbering restarted at 1.
 *
 *   cd frontend
 *   npx esbuild scripts/answer-markdown-check.tsx --bundle --format=esm \
 *     --platform=node --packages=external --outfile=/tmp/amc.mjs && node /tmp/amc.mjs
 *
 * The sample is a real answer from the app, trimmed, and deliberately ends with
 * an unclosed ** and a stray * to check they stay literal.
 */
import { renderToStaticMarkup } from 'react-dom/server';
import { AnswerMarkdown } from '../app/answer-markdown';

const sample = `Based on the provided passages, dealing with a "person with abilities" can be approached in a few ways:

A legacy answer says that intellect distinguishes humans [[Ancient_Indian_Psychology.pdf, पृष्ठ 16 / Page 16]].

A single-bracket answer cites an exact returned source [data/Ancient_Indian_Psychology.pdf, Page 214].

1.  **Through Collaboration and Mutual Support (Leveraging Complementary Abilities):**
    One passage illustrates a scenario where individuals combine their strengths. The cripple can sit on the blind man's shoulders and direct him ⟦data/Ancient_Indian_Psychology.pdf, पृष्ठ 214⟧.

2.  **Through Strategic Engagement and Diplomacy:**
    Chanakya's teachings offer strategies:
    *   **Honoring or Separation:** If a person accepts a proposal, they should be honored.
    *   **Caution Against Exploitation:** Straightforward individuals are often exploited ⟦सम्पूर्ण-चाणक्य-नीति.pdf, पृष्ठ 83⟧.

A current fact from the web ⟦Web 1, Web⟧.

A closing paragraph with an unclosed ** marker and a lone * asterisk.`;

const html = renderToStaticMarkup(<AnswerMarkdown text={sample} sources={[
  { page: 16, preview: 'Legacy bilingual evidence', source: 'Ancient_Indian_Psychology.pdf' },
  { page: 214, preview: 'Book evidence', source: 'data/Ancient_Indian_Psychology.pdf' },
  { page: 83, preview: 'Book evidence', source: 'सम्पूर्ण-चाणक्य-नीति.pdf' },
  { citation_label: 'Web 1', preview: '', source: 'Example News', source_type: 'web' },
]} />);

if (!html.includes('Source 1: Ancient_Indian_Psychology.pdf') || html.includes('[[Ancient_Indian_Psychology.pdf')) {
  throw new Error('Legacy bilingual citation did not render as a source marker.');
}
if (!html.includes('Source 2: data/Ancient_Indian_Psychology.pdf') || html.includes('[data/Ancient_Indian_Psychology.pdf, Page 214]')) {
  throw new Error('Exact single-bracket citation did not render as a source marker.');
}
if (!html.includes('Source 4: Example News') || html.includes('⟦Web 1, Web⟧')) {
  throw new Error('Web citation did not render as a source marker.');
}
console.log(html);

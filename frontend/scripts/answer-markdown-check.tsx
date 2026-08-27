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

1.  **Through Collaboration and Mutual Support (Leveraging Complementary Abilities):**
    One passage illustrates a scenario where individuals combine their strengths. The cripple can sit on the blind man's shoulders and direct him (data/Ancient_Indian_Psychology.pdf, पृष्ठ 214).

2.  **Through Strategic Engagement and Diplomacy:**
    Chanakya's teachings offer strategies:
    *   **Honoring or Separation:** If a person accepts a proposal, they should be honored.
    *   **Caution Against Exploitation:** Straightforward individuals are often exploited (सम्पूर्ण-चाणक्य-नीति.pdf, पृष्ठ 83).

A closing paragraph with an unclosed ** marker and a lone * asterisk.`;

console.log(renderToStaticMarkup(<AnswerMarkdown text={sample} />));

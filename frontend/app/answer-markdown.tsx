'use client';

import { Fragment, ReactNode } from 'react';

/**
 * Renders the small slice of Markdown the answer model actually emits.
 *
 * SYSTEM_PROMPT rule 7 asks for bullets and lists "only when the content
 * genuinely calls for one", so the live grammar is narrow: blank-line separated
 * blocks, `1.` ordered items, `*` or `-` bullets, and `**bold**` inline. That is
 * the whole supported set. Anything else -- tables, headings, links, nested
 * lists -- falls through as a plain paragraph with its source text intact.
 *
 * Handling five constructs exactly and passing the rest through verbatim keeps
 * the worst case "unstyled but readable". Half-supporting ten constructs would
 * instead corrupt real answers, and answers here are the product.
 *
 * Everything returns React elements. Nothing goes through
 * dangerouslySetInnerHTML: this text comes from a model quoting OCR'd documents,
 * so it must never be parsed as HTML.
 */

const BULLET = /^\s*[*-]\s+(.*)$/;
const ORDERED = /^\s*(\d+)\.\s+(.*)$/;

/** `**bold**` -> <strong>. Splits on the delimiter, so odd/unclosed markers
 *  simply stay as literal text rather than swallowing the rest of the answer. */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') && part.length > 4
      ? <strong key={i}>{part.slice(2, -2)}</strong>
      : <Fragment key={i}>{part}</Fragment>
  );
}

/** A list item's continuation lines are indented under it; join them into one
 *  flowing line so the paragraph wraps naturally instead of breaking mid-thought. */
const squash = (lines: string[]) => lines.join(' ').replace(/\s+/g, ' ').trim();

export function AnswerMarkdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let items: { ordered: boolean; parts: string[] }[] = [];

  const flushPara = () => {
    if (!para.length) return;
    blocks.push(<p key={`p${blocks.length}`}>{inline(squash(para))}</p>);
    para = [];
  };
  const flushList = () => {
    if (!items.length) return;
    const ordered = items[0].ordered;
    const rendered = items.map((it, i) => <li key={i}>{inline(squash(it.parts))}</li>);
    blocks.push(ordered
      ? <ol key={`l${blocks.length}`}>{rendered}</ol>
      : <ul key={`l${blocks.length}`}>{rendered}</ul>);
    items = [];
  };

  for (const line of lines) {
    // A blank line ends a paragraph but NOT a list. The model writes "loose"
    // lists -- a blank line between "1." and "2." -- and flushing there would
    // emit one <ol> per item, restarting the numbering at 1 each time. Only a
    // flush-left non-item line below actually closes the list.
    if (!line.trim()) { flushPara(); continue; }

    const ordered = ORDERED.exec(line);
    const bullet = BULLET.exec(line);

    if (ordered || bullet) {
      flushPara();
      const isOrdered = Boolean(ordered);
      // A switch between list styles starts a new list rather than mixing them.
      if (items.length && items[0].ordered !== isOrdered) flushList();
      items.push({ ordered: isOrdered, parts: [(ordered ? ordered[2] : bullet![1])] });
    } else if (items.length) {
      // Indented text under an item continues it; a flush-left line ends the list.
      if (/^\s+/.test(line)) items[items.length - 1].parts.push(line.trim());
      else { flushList(); para.push(line); }
    } else {
      para.push(line);
    }
  }
  flushPara();
  flushList();

  return <div className="answer-copy">{blocks}</div>;
}

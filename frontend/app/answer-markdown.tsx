'use client';

import { Fragment, ReactNode, useId } from 'react';

export type CitationSource = {
  page?: number;
  preview: string;
  source: string;
  source_type?: string;
  timestamp?: string;
  video_title?: string;
};

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
const INLINE_TOKEN = /(\*\*[^*]+\*\*|⟦[^⟦⟧\n]+⟧)/g;
const CITATION = /^⟦(.+),\s*(?:(Page|पृष्ठ|पेज|Document section)\s+(\d+)|(Timestamp)\s+(\d{1,2}:\d{2}))⟧$/i;

const normalized = (value: string) => value.trim().replace(/\\/g, '/').split('/').at(-1)?.toLocaleLowerCase() || '';

function citedSources(citation: string, sources: CitationSource[]) {
  const match = CITATION.exec(citation);
  if (!match) return null;

  const citedName = normalized(match[1]);
  const citedNumber = match[3] ? Number(match[3]) : null;
  const citedTimestamp = match[5] || null;
  const matches = sources.flatMap((source, index) => {
    const sourceNames = [source.source, source.video_title].filter(Boolean).map((name) => normalized(name!));
    const sameName = sourceNames.includes(citedName);
    if (!sameName) return [];
    if (citedTimestamp && source.timestamp === citedTimestamp) return [{ index, source }];
    return citedNumber === source.page ? [{ index, source }] : [];
  });

  return matches.length ? matches : null;
}

/** `**bold**` -> <strong>. Splits on the delimiter, so odd/unclosed markers
 *  simply stay as literal text rather than swallowing the rest of the answer. */
function inline(
  text: string,
  sources: CitationSource[],
  nextTooltipId: () => string,
  onCitationClick?: (source: CitationSource) => void,
): ReactNode[] {
  return text.split(INLINE_TOKEN).filter(Boolean).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }

    const citations = citedSources(part, sources);
    if (!citations) return <Fragment key={i}>{part}</Fragment>;

    return (
      <Fragment key={i}>
        {citations.map(({ index, source }) => {
          const tooltipId = nextTooltipId();
          return (
            <span className="citation" key={`${index}-${tooltipId}`}>
              <sup>
                <button
                  type="button"
                  aria-label={`Source ${index + 1}: ${source.source}`}
                  aria-describedby={tooltipId}
                  onClick={() => onCitationClick?.(source)}
                >
                  {index + 1}
                </button>
              </sup>
              <span className="citation-tooltip" id={tooltipId} role="tooltip">
                <strong>{source.video_title || source.source}</strong>
                <small>{part.slice(1, -1)}</small>
                <span>{source.preview}</span>
              </span>
            </span>
          );
        })}
      </Fragment>
    );
  });
}

/** A list item's continuation lines are indented under it; join them into one
 *  flowing line so the paragraph wraps naturally instead of breaking mid-thought. */
const squash = (lines: string[]) => lines.join(' ').replace(/\s+/g, ' ').trim();

export function AnswerMarkdown({ text, sources = [], onCitationClick }: {
  text: string;
  sources?: CitationSource[];
  onCitationClick?: (source: CitationSource) => void;
}) {
  const idPrefix = useId();
  let tooltipSequence = 0;
  const nextTooltipId = () => `${idPrefix}-citation-${tooltipSequence++}`;
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let items: { ordered: boolean; parts: string[] }[] = [];

  const flushPara = () => {
    if (!para.length) return;
    blocks.push(<p key={`p${blocks.length}`}>{inline(squash(para), sources, nextTooltipId, onCitationClick)}</p>);
    para = [];
  };
  const flushList = () => {
    if (!items.length) return;
    const ordered = items[0].ordered;
    const rendered = items.map((it, i) => <li key={i}>{inline(squash(it.parts), sources, nextTooltipId, onCitationClick)}</li>);
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

# YouTube transcript chunk evaluation

Date: 2026-08-26

## Source

- Video: `Lecture 02: Treatment Principles of Ayurveda`
- Channel: SWAYAM BHU
- URL: https://www.youtube.com/watch?v=vm-hu1Iew-M
- Duration: 43:18
- Transcript: Hindi, auto-generated
- Caption coverage: 2,589.56 seconds (about 99.7% of the video)
- Transcript size: 790 snippets, 5,714 words, 29,067 characters

## Method

All candidates used complete caption-snippet boundaries, a 75-second soft
target, a 120-second hard maximum, and 12 seconds of temporal overlap. Twelve
questions were generated from evidence sampled uniformly across the complete
timeline. Each candidate was embedded with `gemini-embedding-001` and queried
inside its own in-memory Chroma collection. The production `chroma_db` was not
read or written.

This is a small synthetic retrieval evaluation, not a general benchmark. Its
purpose is to select a defensible configuration for this lecture-shaped video
before collecting a broader human-authored test set.

## Results

| Target chars | Chunks | Avg seconds | Redundancy | Recall@1 | Recall@3 | MRR | Mean top-1 timestamp error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 88 | 39.46 | 25.22% | 75.00% | 100.00% | 0.8472 | 418.98 s |
| 800 | 41 | 72.77 | 10.70% | 58.33% | 91.67% | 0.7500 | 397.56 s |
| 1200 | 39 | 75.93 | 10.22% | **83.33%** | 91.67% | **0.8750** | **192.15 s** |
| 1600 | 39 | 75.93 | 10.22% | **83.33%** | 91.67% | **0.8750** | **192.15 s** |

## Decision

Use a 1,200-character soft target, 1,500-character hard maximum, 75-second
soft target, 120-second hard maximum, and 12-second overlap for long lecture
transcripts.

The 1,200 and 1,600 configurations were identical because the 75-second target
was reached before either character target. The 1,200 setting is preferable:
it produces the winning chunks while retaining a tighter safety bound for
unusually fast or malformed caption streams. The 400-character candidate had
perfect Recall@3 but more than doubled the chunk count and introduced 2.5 times
as much duplicated text as the winner.

The mean timestamp error is sensitive to a few broad introduction/conclusion
questions in this 12-question set. A larger human-authored benchmark should add
median error and evidence-window overlap before treating these values as a
cross-domain default.

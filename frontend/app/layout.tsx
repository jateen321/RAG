import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000'),
  title: 'Pustak AI — Grounded study companion',
  description: 'Ask questions across indexed Hindi and English documents with answers you can trace back to the source.',
  openGraph: {
    title: 'Pustak AI — Grounded study companion',
    description: 'Ask your books. Trace every answer.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Pustak AI — Ask your books. Trace every answer.' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Pustak AI — Grounded study companion',
    description: 'Ask your books. Trace every answer.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}

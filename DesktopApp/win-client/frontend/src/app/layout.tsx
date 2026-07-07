import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { ThemeProvider, ThemedToaster } from "@/providers/ThemeProvider";
import { I18nProvider } from "@/providers/I18nProvider";
import { THEME_STORAGE_KEY } from "@/lib/theme";
import "katex/dist/katex.min.css";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "TYQA WebUI",
  description:
    "Web UI for TYQA, a self-evolving AI scientist built on DeepAgents/LangGraph.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f9f9f9" },
    { media: "(prefers-color-scheme: dark)", color: "#212121" },
  ],
  colorScheme: "light dark",
};

// Runs before paint so the right theme class is on <html> immediately.
// Mirrors ThemeProvider's resolution; ThemeProvider takes over once React mounts.
const themeScript = `(function(){var k=${JSON.stringify(
  THEME_STORAGE_KEY
)};var t="system";try{t=localStorage.getItem(k)||"system";}catch(_){}var d=t==="dark"||(t!=="light"&&window.matchMedia("(prefers-color-scheme: dark)").matches);var e=document.documentElement;e.classList.toggle("dark",d);e.style.colorScheme=d?"dark":"light";})();`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
    >
      <body
        className={inter.className}
        suppressHydrationWarning
      >
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <NuqsAdapter>
          <ThemeProvider>
            <I18nProvider>
              {children}
              <ThemedToaster />
            </I18nProvider>
          </ThemeProvider>
        </NuqsAdapter>
      </body>
    </html>
  );
}
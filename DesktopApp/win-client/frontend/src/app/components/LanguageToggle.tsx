"use client";

import { Languages } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/providers/I18nProvider";

export function LanguageToggle() {
  const { locale, setLocale, t } = useI18n();
  const nextLocale = locale === "zh-CN" ? "en-US" : "zh-CN";
  const label =
    nextLocale === "zh-CN"
      ? t("language.switchToChinese")
      : t("language.switchToEnglish");

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      onClick={() => setLocale(nextLocale)}
      aria-label={label}
      title={label}
      className="h-8 gap-1.5 px-2 text-xs font-medium text-muted-foreground"
    >
      <Languages
        className="size-4"
        aria-hidden="true"
      />
      <span className="tabular-nums">{locale === "zh-CN" ? "ZH" : "EN"}</span>
    </Button>
  );
}
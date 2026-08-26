-- ================================================================
-- PHASE: Default footer text for the Stylish Monthly Report
-- ================================================================
-- Purpose: The Stylish Monthly Word report used to show a yellow
--          "RCA/quarterly instruction" note box on every complaint
--          page. The client removed that box and instead configured
--          the exact same text as the report footer (via Settings),
--          which renders once per page and reads better there. This
--          makes that text the real DEFAULT for a fresh/reset install
--          instead of relying on manual per-install configuration.
--
-- Strategy: Idempotent insert into dbo.APP_ReportConfig — only fires
--           if no footer_text row exists yet, so it never overwrites
--           a value any install (including this client's) already
--           configured, whether that's this same text or something
--           else entirely.
--
-- Date: 2026-08-26
-- ================================================================

USE IncidentManager;
GO

IF NOT EXISTS (SELECT 1 FROM dbo.APP_ReportConfig WHERE ConfigKey = 'footer_text')
    INSERT INTO dbo.APP_ReportConfig (ConfigKey, ConfigValue, UpdatedAt, UpdatedBy)
    VALUES (
        'footer_text',
        N'ملاحظة: 1- التقرير الشهري: الشكاوى المصنفة High يلزم ملء استمارة تحليل السبب الجذري RCA (Root Cause Analysis) إذا لم يتم ملؤها خلال المتابعة، أما المصنفة Medium أو Low فملؤها يكون تبعاً للحاجة بناءً على قرار مسؤول العملية. 2- التقرير الفصلي: ترفع استمارة تحسين تلقائياً تبعاً لـ Target الشكاوى.',
        GETDATE(),
        NULL
    );
GO

-- ================================================================
-- INSTALL: 016_seed_lookup_data_taxonomy_2026_09
-- ================================================================
-- Purpose: Adopt the domain-expert team's evolved classification
--          taxonomy (source: تصنيفات.xlsx) into APP_LOOKUP_SUBCATEGORY
--          and APP_LOOKUP_CLASSIFICATION. Domains (3) and Categories (7)
--          are unchanged -- every delta here is at the subcategory and
--          classification (leaf) level: 3 renames (UPDATE, ID unchanged),
--          8 new subcategories, ~130 new classifications, and freezing
--          the superseded 'Rights' subcategory rather than deleting it.
--
--          Idempotent by construction (UPDATE is naturally idempotent;
--          new rows are IF NOT EXISTS-guarded by explicit ID), matching
--          every other file in this folder -- safe to re-run against an
--          already-populated production database; it only adds/renames,
--          never drops or truncates anything. Seed data, like
--          008/009/010 -- not tracked in SchemaMigrationHistory (see
--          ../migrations/README.md: that table is for schema/DDL changes).
--
--          Deliberately NOT wired into the ML hierarchical predictor
--          (models_directory/.../hierarchical_predictor.py) -- the new
--          subcategories are human-selectable only for now. See the
--          2026-09 taxonomy-update plan for the full risk assessment.
--
-- Date: 2026-09-14
-- ================================================================

USE IncidentManager;
GO

-- ---------------- Renames (same SubCategoryID, name only) ----------------

-- Environment: Security -> Staffing (same bucket, renamed per domain-expert Excel)
UPDATE [dbo].[APP_LOOKUP_SUBCATEGORY] SET [SubCategoryName] = N'Staffing' WHERE [SubCategoryID] = 12 AND [SubCategoryName] <> N'Staffing';
GO

-- Listening: fix DB typo 'Dimissing Patients' -> 'Dismissing Patients'
UPDATE [dbo].[APP_LOOKUP_SUBCATEGORY] SET [SubCategoryName] = N'Dismissing Patients' WHERE [SubCategoryID] = 6 AND [SubCategoryName] <> N'Dismissing Patients';
GO

-- Safety: Clinician - Errors -> Clinician Skills (same bucket/volume, renamed)
UPDATE [dbo].[APP_LOOKUP_SUBCATEGORY] SET [SubCategoryName] = N'Clinician Skills' WHERE [SubCategoryID] = 22 AND [SubCategoryName] <> N'Clinician Skills';
GO

-- ---------------- Freeze superseded 'Rights' subcategory (kept for history) ----------------
UPDATE [dbo].[APP_LOOKUP_SUBCATEGORY] SET [IsActive] = 0 WHERE [SubCategoryID] = 8;
GO

-- ---------------- New subcategories ----------------
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SUBCATEGORY] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 29)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (29, 6, N'Making & following care plans', 1); -- excel incident count at export time: 78
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 30)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (30, 6, N'Neglect -Nourishment & Hydration', 1); -- excel incident count at export time: 6
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 31)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (31, 6, N'Outcomes & Side Effects', 1); -- excel incident count at export time: 4
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 32)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (32, 6, N'Rough Handling & Discomfort', 1); -- excel incident count at export time: 2
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 33)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (33, 3, N'Privacy & Dignity', 1); -- excel incident count at export time: 13
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 34)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (34, 3, N'Consent', 1); -- excel incident count at export time: 5
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 35)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (35, 3, N'Confidentially', 1); -- excel incident count at export time: 3
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 36)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName], [IsActive]) VALUES (36, 2, N'Token Listening', 1); -- excel incident count at export time: 1
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SUBCATEGORY] OFF;
GO

-- ---------------- New classifications (128 rows) ----------------
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CLASSIFICATION] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 156)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (156, 19, N'خلل في العناية التمريضية(الميل..)', N'Foley Problems(Nursing Care..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 157)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (157, 19, N'خلل في المتابعة والمراقبة(العلامات الحياتية,الحرارة..)', N'Clinical Evaluation/Monitoring(Vital Signs,Fever..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 158)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (158, 29, N'اعتراض حول الخطة العلاجية(عدم المتابعة بعد العملية)', N'Care Plan Problem(Post-OP..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 159)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (159, 29, N'اعتراض-قرار المغادرة', N'Care Plan Problems(Discharge Decision Maintained..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 160)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (160, 29, N'الخطة العلاجية(تعارض في التشخيص-بين الأطباء)', N'Care Plan Problem(Conflicting Diagnostic-Between Physicians..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 161)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (161, 29, N'الزيارة اليومية(المعالج,الإستشاري,الفيزيائي..)', N'Daily Doctor Visits(Attending/Consulting Physician,Physiotherapist)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 162)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (162, 29, N'العقر السريري', N'Bed Sore Problems(Nursing Care..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 163)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (163, 29, N'تأخر-إعطاء مسكن', N'Delayed Pain Management', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 164)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (164, 29, N'خلل في الخطة العلاجية(العلاج,الدواء..)', N'Care Plan(Clinical Assessment/Following Care..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 165)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (165, 29, N'خلل في المتابعة(حجز العملية..)', N'Care Plan(Failure to Book Operation..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 166)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (166, 29, N'خلل في تنفيذ واتباع الخطة العلاجية(العلاج, الدواء..)', N'Care Plan Problems(Treatment,Medication..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 167)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (167, 29, N'خلل/تأخر في المتابعة الطبية(الFolley/الDrain..)', N'Care Plan Problem/Delayed(Folley,Drain,Dressing Change..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 168)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (168, 20, N'إهمال-عام(التأخر في الإستجابة للمريض,ترك المريض دون متابعة..)', N'Neglect -General(Delayed response to patient calls,Lack of patient follow up..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 169)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (169, 21, N'خلل في العناية التمريضية(الحفاض,إجراء الحمام..)', N'Nursing Care Problems(Diapper,bath..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 170)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (170, 30, N'إهمال-تغذية(إهمال/حساسية طعام..)', N'Neglect-Nourishment(Malnutrition/Allergic Food..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 171)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (171, 30, N'إهمال-تغذية(طعام غير مناسب(النوعية))', N'Neglect-Nourishment(Wrong Food Provided..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 172)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (172, 30, N'خلل-متابعة(وجبة الطعام)', N'Neglect-Nourishment(Meal Not Requested)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 173)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (173, 31, N'نتائج/آثار جانبية(دواء,علاج..)', N'Outcomes(Treatment,Medication..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 174)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (174, 32, N'خلل في أسلوب تقديم الرعاية(أسلوب خشن..)', N'Rough Handling(Physical/Procedural Distress..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 175)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (175, 22, N'خلل في مهارة التمريض(المصل..)', N'Clinician Skills(IV Problem-Nursing Skills..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 176)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (176, 22, N'خلل في مهارة فني المختبر(سحب الدم..)', N'Technician Skills (Failed Attempts)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 177)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (177, 22, N'خلل-تمريضي(إهمال/المهارة التقنية/سوء التطبيق..)', N'Clinician Skills(Negligence,Technical Skills,Misapplication..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 178)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (178, 23, N'خطأ في التشخيص(التقييم السريري..)', N'Error- Diagnosis(Clinical Assessment/Judgement..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 179)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (179, 23, N'خطأ في التقييم/النتاج(التحليل المخبرية)', N'Error-Diagnosis( LAB Results/Interpretation)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 180)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (180, 23, N'خطأ-تشخيص(خلل في النتائج/التقارير)', N'Error – Diagnosis (Incorrect Results/Reports..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 181)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (181, 24, N'أخطاء في العمليات(حدوث حرق,جسم غريب..)', N'Surgical Safety Errors(Burn,Foreign Object..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 182)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (182, 24, N'التقاط-جرثومة(الKimal..)', N'Suspected Nosocomial Infection(Procedure-Related..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 183)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (183, 24, N'التقاط-جرثومة(المستشفى..)', N'Suspected Nosocomial Infection(Hospital Acquired Infection..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 184)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (184, 24, N'خطأ-تمريضي(شكة المصل..)', N'Safety Incidents(IV-Related Complications..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 185)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (185, 24, N'خطأ-عام(تحضير المريض-قبل العملية..)', N'Error-General(Patient Preparation Problem-Shaving..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 186)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (186, 24, N'خطأ-عام(خطأ طبي,مضاعفة جراحية/طبية..)', N'Error -General(Medical Error,Surgical/Medical Complication,Wrong Procedure..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 187)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (187, 24, N'خطأ-عام(عدم كفاية العينة/إعادة السحب..)', N'Error-General(LAB.Procedure/Repeated Testing)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 188)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (188, 25, N'خطأ دواء(خطأ في الملصق..)', N'Error-Medication(Labeling Error..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 189)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (189, 25, N'خطأ دواء(دواء خارج الصلاحية..)', N'Error – Medication (Expired Medication Used)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 190)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (190, 25, N'خطأ-دواء(خلل متابعة الوصفة الطبية..)', N'Error-Medication(Discharge Treatment Plan-Failure to Deliver)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 191)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (191, 27, N'تأخر -إستجابة(الآلية السريرية المتبعة..)', N'Failure to Respond(Clinical Process..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 192)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (192, 27, N'تأخر/خلل في الإستجابة(Emergency Care..)', N'Delayed/Failure To Respond(Emergency Care..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 193)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (193, 27, N'خلل في الرعاية(تقييم خاطئ..)', N'Nursing Care Failure(Assessment Error..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 194)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (194, 27, N'عدم الرد على الجرس(غير موصل)', N'Failure to Respond To Patient Call(Nurse call unfunctional)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 195)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (195, 26, N'خلل تنسيق(الفريق الطبي..)', N'Teamwork(Coordination Failure..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 196)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (196, 9, N'الضجة الصادرة(الكادرالتمريضي/الطبي..)', N'Accomodation(Noisy Ward Surrondings-Staff)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 197)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (197, 9, N'الضجة الصادرة(عدد الزوار,المرضى..)', N'Accomodation(Noisy Ward Surrondings-Visitors)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 198)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (198, 9, N'الضجة(أجهوة,أبواب..)', N'Accomodation(Noise-Devices,Doors..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 199)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (199, 9, N'الضجة-الورشة', N'Accomodation(Noise-Workshop..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 200)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (200, 9, N'تغذية متعدد(تغليف-نايلون..)', N'Nutritional Problem(Unsafe Packaging-Food)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 201)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (201, 9, N'جغرافية-القسم', N'Accomodation Problems(Ward Allocation, Facility Area..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 202)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (202, 9, N'خلل-متابعة الSet(الدرجة الأولى)', N'Service Issue(First-Class Amenities Not Provided)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 203)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (203, 9, N'خلل-متابعة(وجبة-مرافق)', N'Accomodation(Companion Meal..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 204)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (204, 9, N'مشاكل تتعلق بالإقامة(التكييف,التدفئة..)', N'Accomodation(Air Conditionning Problem..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 205)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (205, 9, N'مشاكل تتعلق بالإقامة(محارم,سلة النفايات..)', N'Sanitation Problems(Tissue,Basket...)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 206)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (206, 9, N'مشاكل-البيئة المحيطة(التكييف..)', N'Poor Environment(Air Conditionning Problems..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 207)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (207, 9, N'مشاكل-مستلزمات(الشراشف/وسادة/حرام...)', N'Facilities Issues(Pillows,Covers&Bedding..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 208)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (208, 9, N'مشكلة-WIFI', N'Accomodation(Amenties-No WIFI..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 209)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (209, 9, N'مشكلة-مباني(انزلاق المريض)', N'Accomodation Problems(Unsafe Environment..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 210)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (210, 11, N'أجهزة/معدات(عدم تأمين أجهزة/عملية,عطل/جهازالقلب..)', N'Equipement(Surgical Delayed- Required Device not Provided..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 211)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (211, 11, N'تأخر-عام(عطل برنامج-المعلوماتية..)', N'Delay -General(System Malfunction-IT/Informatics..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 212)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (212, 11, N'خلل-أجهزة/معدات(الجرس,الطابعة..)', N'Equipement Failure(Facility Equipement..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 213)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (213, 11, N'خلل-تأمين(أدوية ..)', N'Equipement(Medication Supply..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 214)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (214, 11, N'مشاكل تتعلق بالإقامة(أجهزة غير كافية,أسرة..)', N'Equipement Problems(Beds,Wheelchair..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 215)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (215, 11, N'مشاكل-الإقامة(خدمات إضافية/التلفاز..)', N'Equipement Problems(Amenties(TV,Refrigerator..))', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 216)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (216, 11, N'مشاكل-تجهيزات الحمام(القفل/الشطافة/كرسي الحمام..)', N'Equipement Issues(Bathroom Problems..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 217)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (217, 12, N'الإنتظار(بطء في النقل إلى الغرفة/النقل..)', N'Staffing(RoomTransfer Delayed/Transfer Delayed..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 218)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (218, 12, N'ممرضين ذكور(خلل في العناية التمريضية"الحفاض,إجراء الحمام..")', N'Staffing(Delayed Personal Care"Diapper,bath..")', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 219)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (219, 10, N'مشاكل في النظافة(الغرفة,الحمام..)', N'Hygene Problem(Poor Cleanliness..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 220)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (220, 10, N'وجود حشرات', N'Hygene Problem(Presence of Insects,..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 221)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (221, 13, N'آلية المرتجع', N'Financial Problem(Refund Process)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 222)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (222, 13, N'آلية تحديد المواعيد(عدم تحديد وقت حضور واضح..)', N'Pre-Admission Problem(ِAdmission Time Not Clarified..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 223)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (223, 13, N'إجراء إداري(ضبط الطعام/التجوّل..)', N'Policies(Security Protocol..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 224)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (224, 13, N'إجراء-إداري(كلفة الموقف..)', N'Policies(Parking-Cost)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 225)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (225, 13, N'إجراءات -مالية(الكلفة..)', N'OPD-Policy(Payement Required-Policy..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 226)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (226, 13, N'اعتراض-برتوكول الجمعيات', N'Financial Problem(Payment Policy-Association)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 227)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (227, 13, N'اعتراض-تنقّل المرضى(التنقّل..)', N'Patient Flow Issues(Referral to ..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 228)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (228, 13, N'اعتراض-حجز الهوية مقابل الدفع', N'Policies(ID/Document Pending-Payment)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 229)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (229, 13, N'الإجراءات الإدارية(خلل تنسيق بين الطبيب/الاكيب..)', N'Procedures(Failure to Coordinate Physician/Team..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 230)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (230, 13, N'المعتقدات والمبادئ(البعد الديني/الثقافي..)', N'Policies(Cultural/Religious Preference..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 231)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (231, 13, N'برتوكول-إداري(بقاء الأهل إلى جانب المريض..)', N'Policies(Continous Family Presence..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 232)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (232, 13, N'برتوكول-إداري(دخول مبكر..)', N'Policies(Early Admission Day/Pre-Procedure Timing.. )', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 233)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (233, 13, N'برتوكول-إداري(ملف طوارئ(جديد)/تحويل المريض إلى العيادات..)', N'Policies(Admission Requirement(Dyalisis)/Re-Visit Protocol(ER)/Clinic Referral(ER)..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 234)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (234, 13, N'خلل-تسجيل(عملية/تمييل..)', N'Failure to Register(Operation Procedure..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 235)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (235, 13, N'خلل-متابعة(الآلية المتبعة/المستندات/التقارير..)', N'Lack of Follow-up(Procedures/Documents/Reports..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 236)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (236, 13, N'عدم تنسيق(حالات المرضى"انزعاج..")', N'Placement Problem(Inappropriate Room Sharing..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 237)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (237, 13, N'عدم توفر اختصاص/خدمة طبية', N'Service/Specialist Availability(IVF Service,Cardiac Emergency Unit..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 238)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (238, 13, N'عدم توفر غرف درجة أولى', N'Placement Problem(First-Class Room Unavailable)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 239)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (239, 13, N'كلفة مرتفعة/زائدة', N'Financial Problem(Cost..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 240)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (240, 13, N'مشاكل-إقامة(الغرفة/القسم غير مناسب..)', N'Placement Problem(Room Discomfort..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 241)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (241, 13, N'مشاكل-الفوترة(خطأ حسابي..)', N'Billing Problem(Verify Correct Amount..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 242)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (242, 13, N'مشاكل-مكتب الدخول(ضياع ملف/كود واي-فاي..)', N'Admission Office Problem(Patient File Lost/Code Wifi Not Provided..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 243)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (243, 13, N'مشكلة-حفظ الممتلكات(فقدان أغراض,مال..)', N'Security Problem(Patient Property LOSS..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 244)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (244, 13, N'مكتب الدخول-موافقات/مستندات', N'Policies(Approval Required,Documents..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 245)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (245, 14, N'تأخر-وصول(انتظار طبيب/تقييم سريري..)', N'Delay- Access"ER"(Physician Wait/Clinical Decision..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 246)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (246, 14, N'تأخر-وصول(رفض استقبال ..)', N'Delay -Access(Refusal to Treat-Injured Patient..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 247)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (247, 14, N'تأخر-وصول(عدم الرد على الإتصالات الخارجية)', N'Delay -Access(Phone Calls Not Answered)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 248)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (248, 14, N'تأخر-وصول(عدم توفر سرير..)', N'Delay -Access(Bed Unavailable)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 249)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (249, 14, N'تأخر-وصول(مواعيد بعيدة/العيادات الخارجية,القلبية..)', N'Delay -Access(Delayed Appointment)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 250)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (250, 15, N'إنتظار-معاينة(العيادات الخارجية/القلبية..)', N'Delay- General(Waiting for Consultation..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 251)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (251, 15, N'تأخر إنجاز ملف المغادرة', N'Discharge Problem(Delayed Discharge File..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 252)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (252, 15, N'تأخر-عام(عدم جهوزية الملف..)', N'Delay -General(Incomplete Documents/Reports..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 253)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (253, 16, N'إنتظار-إجراء(إجراء طبي عام..)', N'Delay -Procedure(Medical Procedure Delayed...)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 254)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (254, 16, N'إنتظار-إجراء(الصور..)', N'Delay – Procedure(Waiting for Imaging..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 255)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (255, 16, N'إنتظار-إجراء(الفحوصات..)', N'Delay – Procedure(Waiting for LAB-Tests..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 256)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (256, 16, N'إنتظار-إجراء(نتائج الصور..)', N'Delay – Procedure(Delayed Imaging Results ..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 257)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (257, 16, N'تأجيل/تأخير(عملية,تمييل..)', N'Delay -Procedure(Delayed Surgical Procedure)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 258)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (258, 16, N'تأخر-إجراء(الإنتظار للمعاينة..)', N'Delay- Procedure"PRE-OP"(Waiting for Consultation)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 259)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (259, 16, N'تأخر-إجراء(عملية-عدم توفر سرير..)', N'Delay -Procedure(Surgery/Examination-Bed Unavailable)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 260)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (260, 18, N'السجلات الطبية(متطلبات,مستندات..)', N'Medical Records(Document Issues/Requirements..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 261)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (261, 18, N'خلل-توثيق(وثائق طبية,أمر المغادرة..)', N'Failure to Document (Document Issue/Discharge Order)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 262)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (262, 17, N'برتوكول الزيارة(عدم السماح بالزيارة..)', N'Visiting Policy(Visitors Restrictions-Family visit Refused..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 263)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (263, 1, N'عدم-شرح(التعليمات الخاصة بما قبل الدخول..)', N'Communication(Pre-Admission Instructions Problem..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 264)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (264, 1, N'عدم-شرح(الخطة العلاجية..)', N'Communication(Care Plan Explanation..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 265)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (265, 1, N'عدم-شرح(خطة العلاج-مريضICU)', N'Communication(Failure to Inform Family"ICU-Patient")', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 266)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (266, 1, N'عدم-شرح(عدم إبلاغ الأهل بالمعلومات(تربيط المريضة,NPO..))', N'Communication(Failure to Inform Family(Restraint,NPO,PEG..))', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 267)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (267, 1, N'عدم-شرح(عدم إبلاغ بنقل المريض..)', N'Communication(Failure To inform Family-Transition..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 268)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (268, 1, N'عدم-شرح(مستلزمات إضافية)', N'Communication(Additional Supplies Used/Cost Gap..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 269)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (269, 2, N'تأخر-شرح(طمأنة الأهل بعد العملية/وضع المريض..)', N'Delayed Communication(Family Delayed Informed..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 270)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (270, 3, N'تضارب-معلومات(تعليمات/تشخيصات..)', N'Conflicting Information(Conflicting Instructions/Diagnosis..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 271)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (271, 3, N'خلل-معلومات(إفصاح/كشف معلومة/أخبار سيئة..)', N'Incorrect Communication(Breaking Bad News/Disclosure of Diagnosis..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 272)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (272, 6, N'استخفاف/استبعاد(أعراض تعب-النتائج)', N'Dismissive Listening(Reported Fatigue/Dyspena Ignored..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 273)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (273, 6, N'عدم-تقبّل الأهل(إيقاف علاج)', N'Family Disagree(Care Expectations..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 274)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (274, 5, N'رفض/تجاهل(عدم الإصغاء..)', N'Ignoring Patients(Distress Ignored..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 275)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (275, 36, N'معاينة شكلية/سطحية', N'Ignored Mild Patinet Pain', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 276)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (276, 35, N'سريّة المعلومات(عدم خصوصية المعلومات..)', N'Confidentially(Lack of Patient Information..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 277)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (277, 34, N'عدم الموافقة(خلل في الإبلاغ عن تبديل طبيب..)', N'Consent(Failure to Inform Staffing/Clinician change..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 278)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (278, 7, N'أسلوب جاف/حاد/تنفّر', N'Rude Behavior(Sharp, Cold Manner..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 279)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (279, 7, N'الإهتمام/التعاطف/البسمة/سلوك غير مهني', N'Condescending Manner/Unprofessional Conduct', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 280)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (280, 7, N'التعاطي بفوقية/ازدراء', N'Discrimination/Humilation', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 281)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (281, 7, N'سلوك فادح (تضارب..)', N'Gross Patient Humiliation', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 282)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (282, 33, N'خلل-المعتقدات الدينية(ذكور-إناث)', N'Privacy(Patient''s Religious(Male/Female)..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 283)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (283, 33, N'خلل-خصوصية(احترام الخصوصية..)', N'Privacy(Lack of Patient Privacy..)', 1);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CLASSIFICATION] OFF;
GO

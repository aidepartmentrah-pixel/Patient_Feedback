-- Universal lookup/reference seed data. No business or patient data.

-- dbo.APP_LOOKUP_BUILDING (2 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_BUILDING] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_BUILDING] WHERE [BuildingID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_BUILDING] ([BuildingID], [BuildingCode], [BuildingName]) VALUES (1, N'RAH', N'RAH');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_BUILDING] WHERE [BuildingID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_BUILDING] ([BuildingID], [BuildingCode], [BuildingName]) VALUES (2, N'BCI', N'BCI');
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_BUILDING] OFF;
GO

-- dbo.APP_LOOKUP_CASE_STAGE (6 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CASE_STAGE] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STAGE] WHERE [StageID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STAGE] ([StageID], [StageName], [StageOrder]) VALUES (1, N'Examination & Diagnosis', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STAGE] WHERE [StageID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STAGE] ([StageID], [StageName], [StageOrder]) VALUES (2, N'Admission', 2);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STAGE] WHERE [StageID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STAGE] ([StageID], [StageName], [StageOrder]) VALUES (3, N'Care on the Ward', 3);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STAGE] WHERE [StageID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STAGE] ([StageID], [StageName], [StageOrder]) VALUES (4, N'Operation / Procedure', 4);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STAGE] WHERE [StageID] = 5)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STAGE] ([StageID], [StageName], [StageOrder]) VALUES (5, N'Discharge / Transfer', 5);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STAGE] WHERE [StageID] = 6)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STAGE] ([StageID], [StageName], [StageOrder]) VALUES (6, N'Unspecified', 99);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CASE_STAGE] OFF;
GO

-- dbo.APP_LOOKUP_CASE_STATUS (5 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CASE_STATUS] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STATUS] WHERE [CaseStatusID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STATUS] ([CaseStatusID], [Code], [Name], [IsFinal], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (1, N'OPEN', N'Open', 0, 1, 1, CONVERT(datetime2, '2025-12-23 13:00:20.113000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STATUS] WHERE [CaseStatusID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STATUS] ([CaseStatusID], [Code], [Name], [IsFinal], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (2, N'IN_PROGRESS', N'In Progress', 0, 1, 2, CONVERT(datetime2, '2025-12-23 13:00:20.113000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STATUS] WHERE [CaseStatusID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STATUS] ([CaseStatusID], [Code], [Name], [IsFinal], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (3, N'CLOSED', N'Closed', 1, 1, 3, CONVERT(datetime2, '2025-12-23 13:00:20.113000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STATUS] WHERE [CaseStatusID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STATUS] ([CaseStatusID], [Code], [Name], [IsFinal], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (4, N'DRAFT', N'Draft', 0, 1, 0, CONVERT(datetime2, '2026-05-08 10:30:17.480000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CASE_STATUS] WHERE [CaseStatusID] = 5)
    INSERT INTO [dbo].[APP_LOOKUP_CASE_STATUS] ([CaseStatusID], [Code], [Name], [IsFinal], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (5, N'READY_TO_SEND', N'Ready to Send', 0, 1, 0, CONVERT(datetime2, '2026-05-08 10:30:17.480000', 121));
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CASE_STATUS] OFF;
GO

-- dbo.APP_LOOKUP_CATEGORY (7 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CATEGORY] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (1, 3, N'Communication', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (2, 3, N'Listening', 2);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (3, 3, N'Respect & Patient Rights', 3);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (4, 2, N'Environment', 4);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 5)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (5, 2, N'Institutional Processes', 5);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 6)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (6, 1, N'Quality of Care', 6);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CATEGORY] WHERE [CategoryID] = 7)
    INSERT INTO [dbo].[APP_LOOKUP_CATEGORY] ([CategoryID], [DomainID], [CategoryName], [CategoryOrder]) VALUES (7, 1, N'Safety', 7);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CATEGORY] OFF;
GO

-- dbo.APP_LOOKUP_CLASSIFICATION (78 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CLASSIFICATION] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 78)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (78, 1, N'التواصل الغائب', N'Absent Communication', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 79)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (79, 2, N'التواصل المتأخر', N'Delayed Communication', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 80)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (80, 3, N'التواصل غير الصحيح', N'Incorrect Communication', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 81)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (81, 4, N'عدم تقبّل حضور الطبيب المساعد', N'Failure to Provide (Assistant Visit Issue..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 82)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (82, 4, N'خلل في التواصل(إعطاء المعلومات..)', N'Failure to Provide (Information, Treatment..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 83)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (83, 5, N'تجاهل المريض', N'Ignoring Patients', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 84)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (84, 6, N'رفض الإستماع المريض/المرافق', N'Dismissing Patients', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 85)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (85, 7, N'عدم الإحترام', N'Disrespect', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 86)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (86, 8, N'المعتقدات والمبادئ', N'Respect for Beliefs', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 87)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (87, 9, N'مشاكل تتعلق بالإقامة(التكييف,التدفئة..)', N'Accommodation Problems(Air Conditioning..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 88)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (88, 9, N'مشاكل تتعلق بالإقامة(جغرافية المكان..)', N'Accommodation Problems (Area Problem..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 89)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (89, 9, N'مشاكل تتعلق بالإقامة(أجهزة غير كافية,أسرة,تهوئة..)', N'Accommodation Problems (Devices, Beds..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 90)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (90, 9, N'مشاكل تتعلق بالإقامة(محارم,سلة النفايات..)', N'Accommodation Problems (Tissue, Basket..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 91)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (91, 9, N'طلبات الدرجة الأولى', N'First Class Services', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 92)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (92, 9, N'الضجة', N'Noise', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 93)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (93, 9, N'الضجة الصادرة(عدد الزوار,المرضى..)', N'Noise (Accompagnant, Patients..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 94)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (94, 9, N'الضجة(أجهوة,أبواب..)', N'Noise (Devices, Doors..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 95)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (95, 9, N'ضجة الموظفين', N'Noise (Employee..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 96)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (96, 9, N'الضجة من الورشة', N'Noise (Workshop..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 97)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (97, 9, N'ضجة الورشة', N'Noise (Workshop..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 98)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (98, 9, N'تغذية متعدد(الأكل بارد,غير كاف..)', N'Nutritional Problem (Cold Food, Insufficient..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 99)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (99, 9, N'تنسيق حالات المرضى(عدم القدرة على الراحة والنوم..)', N'Patient Case Coordination', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 100)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (100, 10, N'بطء تنظيف طارىء', N'Delay Cleaning', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 101)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (101, 10, N'تجاهل المريض', N'Ignoring Patients', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 102)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (102, 10, N'مشاكل في النظافة(الغرفة,الحمام..)', N'Hygiene Problem (Room..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 103)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (103, 11, N'أجهزة ولوازم (قسم العمليات,قسم الإيكو..)', N'Equipment & Supplies Problems (OR, Echo..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 104)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (104, 11, N'مشاكل في الأجهزة(المعلوماتية..)', N'IT Problems', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 105)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (105, 12, N'مشكلة في الأمن(حدوث سرقة..)', N'Security Problem (Lost..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 106)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (106, 13, N'إجراءات معقّدة/موافقات/تكاليف', N'Complex Procedures / Approvals / Costs', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 107)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (107, 13, N'خلل تنسيق إداري(مع الأقسام الأخرى..)', N'Coordination Failure (Team, Other Departments..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 108)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (108, 13, N'خلل تنسيق مع الأقسام الأخرى', N'Coordination Problem (Team, Other Departments..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 109)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (109, 13, N'تحويل المريض من الطوارئ إلى العيادات', N'Disagreement Protocol(ER..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 110)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (110, 13, N'برتوكول طبي', N'Medical Protocol', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 111)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (111, 13, N'آلية نقل العيّنة إلى مختبر خارجي', N'Disagreement Protocol(Lab..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 112)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (112, 13, N'عدم تنسيق حالات المرضى', N'Patient Cases Not Organized', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 113)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (113, 13, N'مشاكل في المستلزمات (الشراشف/وسادة/حرام...)', N'Problems in the Facilities (Pillows, Covers..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 114)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (114, 13, N'خلل في تحديد المواعيد (الصور..)', N'Scheduling Error', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 115)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (115, 14, N'عدم توفر سرير(عادي,عناية..)', N'Bed Unavailability', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 116)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (116, 14, N'مواعيد بعيدة(العيادات الخارجية,القلبية..)', N'Delay Access (Clinic Appointment)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 117)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (117, 14, N'إعطاء مواعيد صور متأخرة', N'Delay Access (Imaging Appointment)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 118)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (118, 14, N'الإنتظار للمعاينة', N'Delay Access (Waiting for Consultation..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 119)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (119, 14, N'انتظار الأدوار', N'Delay Access(Waiting for Consultation..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 120)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (120, 14, N'عدم الرد على الإتصالات الخارجية', N'Phone Calls Not Answered', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 121)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (121, 15, N'تأخير عام', N'Delay - General', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 122)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (122, 15, N'تأخر النقل (من وإلى..)', N'Delay Transfer (Room..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 123)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (123, 15, N'تأخر إنجاز ملف المغادرة', N'Discharge Delay Problem', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 124)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (124, 16, N'تأخر حضور الطبيب', N'Delay Procedure (Medical Attendance..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 125)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (125, 16, N'إنتظار الإجراءات الطبية(صور,فحوصات..)', N'Delay Procedure (Waiting for Imaging, LAB Tests..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 126)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (126, 16, N'تأخر في الرد على الNurse call', N'Delayed Nurse Call', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 127)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (127, 16, N'تأخر تقارير الصور', N'Delayed Procedure (Imaging Reports..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 128)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (128, 16, N'تأخر نتائج الفحوصات', N'Delayed Test Results', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 129)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (129, 16, N'تأجيل/تأخير(عملية,تمييل..)', N'Surgical Procedures Delayed', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 130)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (130, 17, N'اعتراض حول آلية الزيارة', N'Visiting Process', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 131)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (131, 18, N'عدم التوثيق(اللوازم الطبية,أمر المغادرة..)', N'Documentation Problem (Devices, Discharge..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 132)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (132, 19, N'الزيارة اليومية للطبيب', N'Daily Doctor Visits (Attending / Consulting Physician)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 133)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (133, 19, N'تأخر إجراء طبي', N'Delay Medical Procedure', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 134)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (134, 19, N'خلل في متابعة حالة المريض', N'Error in Monitoring', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 135)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (135, 19, N'عدم الموافقة(الخطة العلاجية,قرار المغادرة..)', N'Failure to Agree (Treatment Plan, Discharge Decision..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 136)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (136, 19, N'نقص في مهارة التمريض(المصل..)', N'IV Problem (Nursing Skills..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 137)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (137, 19, N'نقص مهارة فنيّ المختبر(نتائج,سحب الدم..)', N'Technician Skills Deficiency (Tests..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 138)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (138, 21, N'خلل في العناية التمريضية(الحفاض,إجراء الحمام..)', N'Nursing Care Problems (Diaper, Bath..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 139)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (139, 21, N'خلل في العناية التمريضية(الميل..)', N'Problem Procedure(Foley..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 140)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (140, 20, N'خطأ في إجراء الصورة', N'Imaging Procedure Error', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 141)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (141, 20, N'إهمال عام(الرعاية الشخصية,الرعاية الصحية,بيئة آمنة,الدعم النفسي..)', N'Neglect - General (Basic Care, Medical Care, Safe Environment..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 142)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (142, 22, N'خطأ تمريضي(شكة المصل..)', N'IV Problem (IV Insertion Error..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 143)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (143, 22, N'خطأ تمريضي(يهدد سلامة المريض)', N'Technical Skills of Staff (That Compromise Safety)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 144)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (144, 23, N'خطأ في التشخيص', N'Error - Diagnosis', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 145)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (145, 24, N'العقر السريري', N'Bed Sore Problems', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 146)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (146, 24, N'مضاعفات(اختلاط جراحي..)', N'Complications (Surgical Complication)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 147)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (147, 24, N'تقصير في متابعة حالة المريض', N'Error in Monitoring', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 148)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (148, 24, N'خطأ فنيّين(المختبر,الأشعة..)', N'Error Procedure (Lab, X-Ray..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 149)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (149, 24, N'خلل في إجراءات حماية المريض(التقاط جرثومة..)', N'Nasocomial Infection Problem', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 150)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (150, 24, N'بيئة غير آمنة', N'Unsafe Environment', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 151)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (151, 24, N'تغذية(تغليف غير آمن)', N'Unsafe Packaging for Food', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 152)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (152, 25, N'خطأ دواء', N'Error - Medication', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 153)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (153, 26, N'خلل تنسيق طبي(الأطباء,التمريض..)', N'Teamwork Problem (Doctors, Nursing..)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 154)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (154, 27, N'عدم الرد على الجرس(غير موصل)', N'Failure to Respond (Nurse Call Unfunctional)', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLASSIFICATION] WHERE [ClassificationID] = 155)
    INSERT INTO [dbo].[APP_LOOKUP_CLASSIFICATION] ([ClassificationID], [SubCategoryID], [Classification_AR], [Classification_EN], [IsActive]) VALUES (155, 19, N'تجربة_دخان_معدل', N'Smoke Test EN Updated', 1);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CLASSIFICATION] OFF;
GO

-- dbo.APP_LOOKUP_CLINICAL_RISK_TYPE (3 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] WHERE [ClinicalRiskTypeID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] ([ClinicalRiskTypeID], [Code], [Name], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (1, N'ORDINARY', N'Ordinary', 1, 1, CONVERT(datetime2, '2025-12-23 13:00:04.893000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] WHERE [ClinicalRiskTypeID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] ([ClinicalRiskTypeID], [Code], [Name], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (2, N'RED_FLAG', N'Red Flag', 1, 2, CONVERT(datetime2, '2025-12-23 13:00:04.893000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] WHERE [ClinicalRiskTypeID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] ([ClinicalRiskTypeID], [Code], [Name], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (3, N'NEVER_EVENT', N'Never Event', 1, 3, CONVERT(datetime2, '2025-12-23 13:00:04.893000', 121));
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] OFF;
GO

-- dbo.APP_LOOKUP_DOMAIN (3 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_DOMAIN] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_DOMAIN] WHERE [DomainID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_DOMAIN] ([DomainID], [DomainCode], [DomainName], [DomainOrder]) VALUES (1, N'CLINICAL', N'Clinical', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_DOMAIN] WHERE [DomainID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_DOMAIN] ([DomainID], [DomainCode], [DomainName], [DomainOrder]) VALUES (2, N'MANAGEMENT', N'Management', 2);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_DOMAIN] WHERE [DomainID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_DOMAIN] ([DomainID], [DomainCode], [DomainName], [DomainOrder]) VALUES (3, N'RELATIONAL', N'Relational', 3);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_DOMAIN] OFF;
GO

-- dbo.APP_LOOKUP_EXPLANATION_STATUS (4 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_EXPLANATION_STATUS] WHERE [StatusID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ([StatusID], [StatusName]) VALUES (3, N'Forcibly Closed');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_EXPLANATION_STATUS] WHERE [StatusID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ([StatusID], [StatusName]) VALUES (4, N'No Explanation Needed');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_EXPLANATION_STATUS] WHERE [StatusID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ([StatusID], [StatusName]) VALUES (2, N'Responded');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_EXPLANATION_STATUS] WHERE [StatusID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ([StatusID], [StatusName]) VALUES (1, N'Waiting');
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_EXPLANATION_STATUS] OFF;
GO

-- dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE (2 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] WHERE [FeedbackIntentTypeID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] ([FeedbackIntentTypeID], [Code], [NameAr], [NameEn], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (1, N'IMPROVEMENT_OPPORTUNITY', N'فرصة تحسين', N'Improvement Opportunity', 1, 1, CONVERT(datetime2, '2025-12-23 12:59:39.250000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] WHERE [FeedbackIntentTypeID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] ([FeedbackIntentTypeID], [Code], [NameAr], [NameEn], [IsActive], [DisplayOrder], [CreatedAt]) VALUES (2, N'NOTICE', N'تنويه', N'Notice', 1, 2, CONVERT(datetime2, '2025-12-23 12:59:39.250000', 121));
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] OFF;
GO

-- dbo.APP_LOOKUP_HARM_LEVEL (5 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_HARM_LEVEL] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_HARM_LEVEL] WHERE [HarmID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_HARM_LEVEL] ([HarmID], [HarmLevel], [SeverityOrder]) VALUES (1, N'No Harm', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_HARM_LEVEL] WHERE [HarmID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_HARM_LEVEL] ([HarmID], [HarmLevel], [SeverityOrder]) VALUES (2, N'Minor', 2);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_HARM_LEVEL] WHERE [HarmID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_HARM_LEVEL] ([HarmID], [HarmLevel], [SeverityOrder]) VALUES (3, N'Moderate', 3);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_HARM_LEVEL] WHERE [HarmID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_HARM_LEVEL] ([HarmID], [HarmLevel], [SeverityOrder]) VALUES (4, N'Severe', 4);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_HARM_LEVEL] WHERE [HarmID] = 5)
    INSERT INTO [dbo].[APP_LOOKUP_HARM_LEVEL] ([HarmID], [HarmLevel], [SeverityOrder]) VALUES (5, N'Death', 5);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_HARM_LEVEL] OFF;
GO

-- dbo.APP_LOOKUP_RECORD_TYPE (2 rows)
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_RECORD_TYPE] WHERE [RecordTypeID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_RECORD_TYPE] ([RecordTypeID], [TypeName]) VALUES (1, N'Complaint');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_RECORD_TYPE] WHERE [RecordTypeID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_RECORD_TYPE] ([RecordTypeID], [TypeName]) VALUES (2, N'Notice');
GO

-- dbo.APP_Lookup_SatisfactionStatus (3 rows)
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SatisfactionStatus] WHERE [SatisfactionStatusID] = 1)
    INSERT INTO [dbo].[APP_Lookup_SatisfactionStatus] ([SatisfactionStatusID], [StatusNameEn], [StatusNameAr], [IsActive], [CreatedAt]) VALUES (1, N'Not Present', N'غير موجود', 1, CONVERT(datetime2, '2026-02-19 14:11:06.290000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SatisfactionStatus] WHERE [SatisfactionStatusID] = 2)
    INSERT INTO [dbo].[APP_Lookup_SatisfactionStatus] ([SatisfactionStatusID], [StatusNameEn], [StatusNameAr], [IsActive], [CreatedAt]) VALUES (2, N'Satisfied', N'راض', 1, CONVERT(datetime2, '2026-02-19 14:11:06.290000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SatisfactionStatus] WHERE [SatisfactionStatusID] = 3)
    INSERT INTO [dbo].[APP_Lookup_SatisfactionStatus] ([SatisfactionStatusID], [StatusNameEn], [StatusNameAr], [IsActive], [CreatedAt]) VALUES (3, N'Not Satisfied', N'غير راض', 1, CONVERT(datetime2, '2026-02-19 14:11:06.290000', 121));
GO

-- dbo.APP_LOOKUP_SEVERITY (3 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SEVERITY] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SEVERITY] WHERE [SeverityID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_SEVERITY] ([SeverityID], [SeverityCode], [SeverityName], [SeverityOrder], [IsActive], [CreatedAt], [CreatedBy], [UpdatedAt], [UpdatedBy]) VALUES (1, N'LOW', N'Low', 1, 1, CONVERT(datetime2, '2025-12-26 11:47:02.770000', 121), NULL, NULL, NULL);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SEVERITY] WHERE [SeverityID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_SEVERITY] ([SeverityID], [SeverityCode], [SeverityName], [SeverityOrder], [IsActive], [CreatedAt], [CreatedBy], [UpdatedAt], [UpdatedBy]) VALUES (2, N'MEDIUM', N'Medium', 2, 1, CONVERT(datetime2, '2025-12-26 11:47:02.770000', 121), NULL, NULL, NULL);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SEVERITY] WHERE [SeverityID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_SEVERITY] ([SeverityID], [SeverityCode], [SeverityName], [SeverityOrder], [IsActive], [CreatedAt], [CreatedBy], [UpdatedAt], [UpdatedBy]) VALUES (3, N'HIGH', N'High', 3, 1, CONVERT(datetime2, '2025-12-26 11:47:02.770000', 121), NULL, NULL, NULL);
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SEVERITY] OFF;
GO

-- dbo.APP_LOOKUP_SOURCE (8 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SOURCE] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (1, N'Tours', N'جولات', 1, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (2, N'Attendance', N'حضور', 2, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (3, N'Hotline', N'خط ساخن', 3, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (4, N'Box', N'صندوق', 4, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 5)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (5, N'Supervisor', N'مشرف', 5, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 6)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (6, N'Employee', N'موظف', 6, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 7)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (7, N'Office WhatsApp', N'واتساب مكتب', 7, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SOURCE] WHERE [SourceID] = 8)
    INSERT INTO [dbo].[APP_LOOKUP_SOURCE] ([SourceID], [SourceName], [SourceNameAr], [DisplayOrder], [IsActive], [CreatedAt], [UpdatedAt]) VALUES (8, N'Social Media', N'وسائل التواصل', 8, 1, CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121), CONVERT(datetime2, '2025-12-26 14:02:55.977000', 121));
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SOURCE] OFF;
GO

-- dbo.APP_Lookup_SubcaseActionItemStatus (10 rows)
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'ADMIN_APPROVED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'ADMIN_APPROVED', N'Administration Approved', N'Ù…ÙˆØ§ÙÙ‚ Ù…Ù† Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 6, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'ADMIN_REJECTED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'ADMIN_REJECTED', N'Administration Rejected', N'Ù…Ø±ÙÙˆØ¶ Ù…Ù† Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 5, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'CANCELLED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'CANCELLED', N'Cancelled', N'Ù…Ù„ØºÙŠ', 10, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'DEPT_REJECTED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'DEPT_REJECTED', N'Department Rejected', N'Ù…Ø±ÙÙˆØ¶ Ù…Ù† Ø§Ù„Ø¥Ø¯Ø§Ø±Ø©', 3, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'DONE')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'DONE', N'Done', N'Ù…Ù†Ø¬Ø²', 8, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'DRAFT')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'DRAFT', N'Draft', N'Ù…Ø³ÙˆØ¯Ø©', 1, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'IN_PROGRESS')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'IN_PROGRESS', N'In Progress', N'Ù‚ÙŠØ¯ Ø§Ù„ØªÙ†ÙÙŠØ°', 7, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'SUBMITTED_TO_ADMIN')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'SUBMITTED_TO_ADMIN', N'Submitted to Administration', N'Ù…ÙØ±Ø³Ù„ Ø¥Ù„Ù‰ Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 4, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'SUBMITTED_TO_DEPT')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'SUBMITTED_TO_DEPT', N'Submitted to Department', N'Ù…ÙØ±Ø³Ù„ Ø¥Ù„Ù‰ Ø§Ù„Ø¥Ø¯Ø§Ø±Ø©', 2, 1, 0);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseActionItemStatus] WHERE [StatusCode] = N'VERIFIED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseActionItemStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsActive], [IsFinal]) VALUES (N'VERIFIED', N'Verified', N'Ù…ÙˆØ«Ù‚', 9, 1, 1);
GO

-- dbo.APP_Lookup_SubcaseStatus (16 rows)
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'ADMIN_APPROVED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'ADMIN_APPROVED', N'Administration Approved', N'Ù…ÙˆØ§ÙÙ‚ Ù…Ù† Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 7, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'ADMIN_REJECTED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'ADMIN_REJECTED', N'Administration Rejected', N'Ù…Ø±ÙÙˆØ¶ Ù…Ù† Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 6, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'CLOSED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'CLOSED', N'Closed', N'Ù…ØºÙ„Ù‚', 8, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'DEPT_ACCEPTED_PENDING_ADMIN')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'DEPT_ACCEPTED_PENDING_ADMIN', N'Department Accepted - Pending Administration', N'Ù…ÙˆØ§ÙÙ‚Ø© Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© - ÙÙŠ Ø§Ù†ØªØ¸Ø§Ø± Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 5, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'DEPT_REJECTED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'DEPT_REJECTED', N'Department Rejected', N'Ù…Ø±ÙÙˆØ¶ Ù…Ù† Ø§Ù„Ø¥Ø¯Ø§Ø±Ø©', 4, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'FORCE_CLOSED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'FORCE_CLOSED', N'Force Closed', N'Ø¥ØºÙ„Ø§Ù‚ Ø¥Ø¬Ø¨Ø§Ø±ÙŠ', 9, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'FORCE_CLOSED_AT_ADMINISTRATION')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'FORCE_CLOSED_AT_ADMINISTRATION', N'Force Closed at Administration', N'ØªÙ… Ø§Ù„Ø¥ØºÙ„Ø§Ù‚ Ø§Ù„Ù‚Ø³Ø±ÙŠ Ø¹Ù„Ù‰ Ù…Ø³ØªÙˆÙ‰ Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ø¹Ù„ÙŠØ§', 16, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'FORCE_CLOSED_AT_DEPARTMENT')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'FORCE_CLOSED_AT_DEPARTMENT', N'Force Closed at Department', N'ØªÙ… Ø§Ù„Ø¥ØºÙ„Ø§Ù‚ Ø§Ù„Ù‚Ø³Ø±ÙŠ Ø¹Ù„Ù‰ Ù…Ø³ØªÙˆÙ‰ Ø§Ù„Ø¥Ø¯Ø§Ø±Ø© Ø§Ù„Ù…Ø¹Ù†ÙŠØ©', 15, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'FORCE_CLOSED_AT_SECTION')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'FORCE_CLOSED_AT_SECTION', N'Force Closed at Section', N'ØªÙ… Ø§Ù„Ø¥ØºÙ„Ø§Ù‚ Ø§Ù„Ù‚Ø³Ø±ÙŠ Ø¹Ù„Ù‰ Ù…Ø³ØªÙˆÙ‰ Ø§Ù„Ù‚Ø³Ù…', 14, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'FORCE_CLOSED_COMPLETE')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'FORCE_CLOSED_COMPLETE', N'Force Closed - Complete', N'Ø¥ØºÙ„Ø§Ù‚ Ø¥Ø¬Ø¨Ø§Ø±ÙŠ - Ù…ÙƒØªÙ…Ù„', 11, 1, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'FORCE_CLOSED_DRAFT')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'FORCE_CLOSED_DRAFT', N'Force Closed - Draft', N'Ø¥ØºÙ„Ø§Ù‚ Ø¥Ø¬Ø¨Ø§Ø±ÙŠ - Ù…Ø³ÙˆØ¯Ø©', 10, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'PATIENT_SERVICES_DECISION_COMPLETED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'PATIENT_SERVICES_DECISION_COMPLETED', N'Patient Services Decision Completed', N'طھظ… ط¥ط¯ط®ط§ظ„ ظ‚ط±ط§ط± ط®ط¯ظ…ط§طھ ط§ظ„ظ…ط±ط¶ظ‰ ط¨ط­ط³ط¨ ط§ظ„ظ…ط±ط§ط¬ط¹ ط§ظ„ط¹ظ„ظ…ظٹظ‘ط©', 13, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'SECTION_ACCEPTED_PENDING_DEPT')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'SECTION_ACCEPTED_PENDING_DEPT', N'Section Accepted - Pending Department', N'Ù…ÙˆØ§ÙÙ‚Ø© Ø§Ù„Ù‚Ø³Ù… - ÙÙŠ Ø§Ù†ØªØ¸Ø§Ø± Ø§Ù„Ø¥Ø¯Ø§Ø±Ø©', 3, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'SECTION_DENIED')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'SECTION_DENIED', N'Section Denied', N'Ù…Ø±ÙÙˆØ¶ Ù…Ù† Ø§Ù„Ù‚Ø³Ù…', 2, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'SUBMITTED_TO_SECTION')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'SUBMITTED_TO_SECTION', N'Submitted to Section', N'Ù…ÙØ±Ø³Ù„ Ø¥Ù„Ù‰ Ø§Ù„Ù‚Ø³Ù…', 1, 0, 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseStatus] WHERE [StatusCode] = N'WAITING_PATIENT_SERVICES_DECISION')
    INSERT INTO [dbo].[APP_Lookup_SubcaseStatus] ([StatusCode], [StatusNameEn], [StatusNameAr], [DisplayOrder], [IsFinal], [IsActive]) VALUES (N'WAITING_PATIENT_SERVICES_DECISION', N'Waiting Patient Services Decision', N'ط¨ط§ظ†طھط¸ط§ط± ظ‚ط±ط§ط± ط®ط¯ظ…ط§طھ ط§ظ„ظ…ط±ط¶ظ‰ ط¨ط­ط³ط¨ ط§ظ„ظ…ط±ط§ط¬ط¹ ط§ظ„ط¹ظ„ظ…ظٹظ‘ط©', 12, 0, 1);
GO

-- dbo.APP_Lookup_SubcaseType (2 rows)
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseType] WHERE [CaseTypeCode] = N'INCIDENT_RESPONSE')
    INSERT INTO [dbo].[APP_Lookup_SubcaseType] ([CaseTypeCode], [CaseTypeNameEn], [CaseTypeNameAr], [IsActive]) VALUES (N'INCIDENT_RESPONSE', N'Incident Response', N'Ù…Ø¹Ø§Ù„Ø¬Ø© Ø´ÙƒÙˆÙ‰ Ø­Ø§Ø¯Ø«Ø©', 1);
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_Lookup_SubcaseType] WHERE [CaseTypeCode] = N'SEASONAL_REPORT_RESPONSE')
    INSERT INTO [dbo].[APP_Lookup_SubcaseType] ([CaseTypeCode], [CaseTypeNameEn], [CaseTypeNameAr], [IsActive]) VALUES (N'SEASONAL_REPORT_RESPONSE', N'Seasonal Report Response', N'Ù…Ø¹Ø§Ù„Ø¬Ø© ØªÙ‚Ø±ÙŠØ± Ù…ÙˆØ³Ù…ÙŠ', 1);
GO

-- dbo.APP_LOOKUP_SUBCATEGORY (27 rows)
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SUBCATEGORY] ON;
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 1)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (1, 1, N'Absent Communication');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 2)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (2, 1, N'Delayed Communication');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 4)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (4, 1, N'Failure to Provide');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 3)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (3, 1, N'Incorrect Communication');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 6)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (6, 2, N'Dimissing Patients');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 5)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (5, 2, N'Ignoring Patients');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 7)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (7, 3, N'Disrespect');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 8)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (8, 3, N'Rights');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 9)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (9, 4, N'Accommodation');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 11)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (11, 4, N'Equipment');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 12)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (12, 4, N'Security');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 10)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (10, 4, N'Ward Cleanliness');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 13)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (13, 5, N'Bureaucracy');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 14)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (14, 5, N'Delay - Access');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 15)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (15, 5, N'Delay - General');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 16)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (16, 5, N'Delay - Procedure');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 18)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (18, 5, N'Documentation');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 17)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (17, 5, N'Visiting');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 19)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (19, 6, N'Examination & Monitoring');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 20)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (20, 6, N'Neglect -General');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 21)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (21, 6, N'Neglect -Hygiene & Personal Care');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 22)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (22, 7, N'Clinician - Errors');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 23)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (23, 7, N'Error - Diagnosis');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 24)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (24, 7, N'Error - General');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 25)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (25, 7, N'Error - Medication');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 27)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (27, 7, N'Failure to Respond');
IF NOT EXISTS (SELECT 1 FROM [dbo].[APP_LOOKUP_SUBCATEGORY] WHERE [SubCategoryID] = 26)
    INSERT INTO [dbo].[APP_LOOKUP_SUBCATEGORY] ([SubCategoryID], [CategoryID], [SubCategoryName]) VALUES (26, 7, N'Teamwork');
SET IDENTITY_INSERT [dbo].[APP_LOOKUP_SUBCATEGORY] OFF;
GO

-- ml.EmbeddingModelVersion (1 rows)
SET IDENTITY_INSERT [ml].[EmbeddingModelVersion] ON;
IF NOT EXISTS (SELECT 1 FROM [ml].[EmbeddingModelVersion] WHERE [EmbeddingModelVersionID] = 2)
    INSERT INTO [ml].[EmbeddingModelVersion] ([EmbeddingModelVersionID], [ModelName], [ModelPathOrIdentifier], [ModelArchitecture], [ModelChecksum], [EmbeddingDimension], [PoolingMethod], [NormalizationMethod], [TokenizerIdentifier], [ActivatedAt], [RetiredAt], [IsActive], [ConfigurationJson]) VALUES (2, N'mpnet_embeddings (local)', N'models_directory/Classification_Models/model_storage/mpnet_embeddings', N'XLMRobertaModel', NULL, 768, N'mean', NULL, N'models_directory/Classification_Models/model_storage/mpnet_embeddings', CONVERT(datetime2, '2026-07-16 15:10:34.266666', 121), NULL, 1, NULL);
SET IDENTITY_INSERT [ml].[EmbeddingModelVersion] OFF;
GO

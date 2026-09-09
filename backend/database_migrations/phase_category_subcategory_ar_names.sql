/*
================================================================================
CLASS MANAGEMENT EXPANSION — Category & Sub-Category Arabic Names + Active Flag
================================================================================
Purpose: Extend the Settings > Class Management tab to manage Category and
         Sub-Category the same way Classification is already managed (add,
         rename AR/EN, freeze/unfreeze). APP_LOOKUP_CATEGORY and
         APP_LOOKUP_SUBCATEGORY currently only store an English name and have
         no active/inactive flag -- both are needed for that same pattern.

Author: System
Date: 2026-09-09
Version: 1.0

SAFE TO RUN MULTIPLE TIMES (idempotent checks included)

Changes:
- APP_LOOKUP_CATEGORY:    add CategoryNameAr NVARCHAR(100) NULL
- APP_LOOKUP_CATEGORY:    add IsActive BIT NOT NULL DEFAULT 1
- APP_LOOKUP_SUBCATEGORY: add SubCategoryNameAr NVARCHAR(150) NULL
- APP_LOOKUP_SUBCATEGORY: add IsActive BIT NOT NULL DEFAULT 1

Backfill: New Arabic-name columns start NULL on existing rows -- there is no
          safe automatic English->Arabic translation, so existing categories
          and subcategories will show a blank Arabic name until someone fills
          it in through the new Settings screen. IsActive defaults every
          existing row to active (1), matching current behavior (nothing was
          "frozen" before this column existed).
================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Class Management Expansion: Category & Sub-Category columns';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- APP_LOOKUP_CATEGORY.CategoryNameAr
    -- ========================================================================
    IF NOT EXISTS (
        SELECT * FROM sys.columns
        WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_CATEGORY')
        AND name = 'CategoryNameAr'
    )
    BEGIN
        PRINT 'Adding column: APP_LOOKUP_CATEGORY.CategoryNameAr...';
        ALTER TABLE dbo.APP_LOOKUP_CATEGORY ADD CategoryNameAr NVARCHAR(100) NULL;
        PRINT 'CategoryNameAr column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'CategoryNameAr column already exists';
        PRINT '';
    END

    -- ========================================================================
    -- APP_LOOKUP_CATEGORY.IsActive
    -- ========================================================================
    IF NOT EXISTS (
        SELECT * FROM sys.columns
        WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_CATEGORY')
        AND name = 'IsActive'
    )
    BEGIN
        PRINT 'Adding column: APP_LOOKUP_CATEGORY.IsActive...';
        ALTER TABLE dbo.APP_LOOKUP_CATEGORY ADD IsActive BIT NOT NULL DEFAULT 1;
        PRINT 'IsActive column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'IsActive column already exists on APP_LOOKUP_CATEGORY';
        PRINT '';
    END

    -- ========================================================================
    -- APP_LOOKUP_SUBCATEGORY.SubCategoryNameAr
    -- ========================================================================
    IF NOT EXISTS (
        SELECT * FROM sys.columns
        WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_SUBCATEGORY')
        AND name = 'SubCategoryNameAr'
    )
    BEGIN
        PRINT 'Adding column: APP_LOOKUP_SUBCATEGORY.SubCategoryNameAr...';
        ALTER TABLE dbo.APP_LOOKUP_SUBCATEGORY ADD SubCategoryNameAr NVARCHAR(150) NULL;
        PRINT 'SubCategoryNameAr column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'SubCategoryNameAr column already exists';
        PRINT '';
    END

    -- ========================================================================
    -- APP_LOOKUP_SUBCATEGORY.IsActive
    -- ========================================================================
    IF NOT EXISTS (
        SELECT * FROM sys.columns
        WHERE object_id = OBJECT_ID('dbo.APP_LOOKUP_SUBCATEGORY')
        AND name = 'IsActive'
    )
    BEGIN
        PRINT 'Adding column: APP_LOOKUP_SUBCATEGORY.IsActive...';
        ALTER TABLE dbo.APP_LOOKUP_SUBCATEGORY ADD IsActive BIT NOT NULL DEFAULT 1;
        PRINT 'IsActive column added successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT 'IsActive column already exists on APP_LOOKUP_SUBCATEGORY';
        PRINT '';
    END

    COMMIT TRANSACTION;

    PRINT '';
    PRINT '========================================';
    PRINT 'Category & Sub-Category column migration completed successfully';
    PRINT '========================================';
    PRINT '';

END TRY
BEGIN CATCH

    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;

    PRINT '';
    PRINT '========================================';
    PRINT 'ERROR: Category & Sub-Category column migration failed';
    PRINT '========================================';
    PRINT '';
    PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS NVARCHAR(10));
    PRINT 'Error Message: ' + ERROR_MESSAGE();
    PRINT 'Error Line: ' + CAST(ERROR_LINE() AS NVARCHAR(10));
    PRINT '';

END CATCH

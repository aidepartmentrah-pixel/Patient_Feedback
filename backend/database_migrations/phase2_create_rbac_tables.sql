/*
================================================================================
Phase 2: RBAC Core - Database Schema Creation
================================================================================
Purpose: Create authentication and role-based access control tables
Author: System
Date: 2026-01-27
Version: 1.0

IMPORTANT: Safe to run multiple times (uses IF NOT EXISTS checks)

Tables Created:
1. APP_Users - System users with credentials
2. APP_Roles - Role definitions
3. APP_UserRoleScope - User-Role-OrgUnit mappings

Test Users:
- software_admin / admin123
- worker / worker123
- complaint_supervisor / sup123
- section_admin / section123
- department_admin / dept123
- administration_admin / adminis123

================================================================================
*/

BEGIN TRANSACTION;

BEGIN TRY

    PRINT '========================================';
    PRINT 'Phase 2: Creating RBAC Tables';
    PRINT '========================================';
    PRINT '';

    -- ========================================================================
    -- 1) APP_Users Table
    -- ========================================================================
    -- Stores system users with authentication credentials
    
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_Users' AND schema_id = SCHEMA_ID('dbo'))
    BEGIN
        PRINT 'Creating table: APP_Users...';
        
        CREATE TABLE dbo.APP_Users (
            UserID INT IDENTITY(1,1) PRIMARY KEY,
            Username NVARCHAR(100) NOT NULL UNIQUE,
            PasswordHash NVARCHAR(255) NOT NULL,
            IsActive BIT NOT NULL DEFAULT 1,
            CreatedAt DATETIME NOT NULL DEFAULT GETDATE(),
            
            -- Indexes for performance
            INDEX IX_APP_Users_Username NONCLUSTERED (Username),
            INDEX IX_APP_Users_IsActive NONCLUSTERED (IsActive)
        );
        
        PRINT '✓ APP_Users table created successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ APP_Users table already exists';
        PRINT '';
    END

    -- ========================================================================
    -- 2) APP_Roles Table
    -- ========================================================================
    -- Lookup table for system roles
    
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_Roles' AND schema_id = SCHEMA_ID('dbo'))
    BEGIN
        PRINT 'Creating table: APP_Roles...';
        
        CREATE TABLE dbo.APP_Roles (
            RoleID INT IDENTITY(1,1) PRIMARY KEY,
            RoleCode NVARCHAR(50) NOT NULL UNIQUE,
            RoleNameEn NVARCHAR(100) NOT NULL,
            RoleNameAr NVARCHAR(100) NOT NULL,
            
            -- Index for performance
            INDEX IX_APP_Roles_RoleCode NONCLUSTERED (RoleCode)
        );
        
        PRINT '✓ APP_Roles table created successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ APP_Roles table already exists';
        PRINT '';
    END

    -- ========================================================================
    -- 3) APP_UserRoleScope Table
    -- ========================================================================
    -- Maps users to roles with organizational unit scope
    
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'APP_UserRoleScope' AND schema_id = SCHEMA_ID('dbo'))
    BEGIN
        PRINT 'Creating table: APP_UserRoleScope...';
        
        CREATE TABLE dbo.APP_UserRoleScope (
            UserRoleScopeID INT IDENTITY(1,1) PRIMARY KEY,
            UserID INT NOT NULL,
            RoleID INT NOT NULL,
            OrgUnitID INT NOT NULL,
            OrgUnitType NVARCHAR(50) NOT NULL,
            
            -- Foreign Keys
            CONSTRAINT FK_UserRoleScope_User 
                FOREIGN KEY (UserID) REFERENCES dbo.APP_Users(UserID) 
                ON DELETE CASCADE,
            
            CONSTRAINT FK_UserRoleScope_Role 
                FOREIGN KEY (RoleID) REFERENCES dbo.APP_Roles(RoleID) 
                ON DELETE CASCADE,
            
            -- Unique Constraint: Prevent duplicate role assignments
            CONSTRAINT UQ_UserRoleScope 
                UNIQUE (UserID, RoleID, OrgUnitID, OrgUnitType),
            
            -- Indexes for performance
            INDEX IX_UserRoleScope_UserID NONCLUSTERED (UserID),
            INDEX IX_UserRoleScope_RoleID NONCLUSTERED (RoleID),
            INDEX IX_UserRoleScope_OrgUnit NONCLUSTERED (OrgUnitID, OrgUnitType)
        );
        
        PRINT '✓ APP_UserRoleScope table created successfully';
        PRINT '';
    END
    ELSE
    BEGIN
        PRINT '✓ APP_UserRoleScope table already exists';
        PRINT '';
    END

    -- ========================================================================
    -- SEED DATA: Roles
    -- ========================================================================
    
    PRINT 'Inserting role definitions...';
    
    -- Insert roles if they don't exist
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'SOFTWARE_ADMIN')
        INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
        VALUES ('SOFTWARE_ADMIN', 'Software Administrator', 'مسؤول النظام');
    
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'WORKER')
        INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
        VALUES ('WORKER', 'Worker', 'موظف');
    
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'COMPLAINT_SUPERVISOR')
        INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
        VALUES ('COMPLAINT_SUPERVISOR', 'Complaint Supervisor', 'مشرف الشكاوى');
    
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'SECTION_ADMIN')
        INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
        VALUES ('SECTION_ADMIN', 'Section Administrator', 'مسؤول القسم');
    
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'DEPARTMENT_ADMIN')
        INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
        VALUES ('DEPARTMENT_ADMIN', 'Department Administrator', 'مسؤول الإدارة');
    
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Roles WHERE RoleCode = 'ADMINISTRATION_ADMIN')
        INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
        VALUES ('ADMINISTRATION_ADMIN', 'Administration Administrator', 'مسؤول الإدارة العامة');
    
    PRINT '✓ Roles inserted successfully';
    PRINT '';

    -- ========================================================================
    -- SEED DATA: Test Users
    -- ========================================================================
    
    PRINT 'Creating test users...';
    
    -- Note: PasswordHash values are temporary placeholders
    -- They will be replaced with proper bcrypt hashes in the service layer
    
    -- 1. software_admin
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Users WHERE Username = 'software_admin')
    BEGIN
        INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
        VALUES ('software_admin', 'TEMP_HASH_admin123', 1);
        PRINT '  ✓ Created user: software_admin';
    END
    ELSE
        PRINT '  ✓ User already exists: software_admin';
    
    -- 2. worker
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Users WHERE Username = 'worker')
    BEGIN
        INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
        VALUES ('worker', 'TEMP_HASH_worker123', 1);
        PRINT '  ✓ Created user: worker';
    END
    ELSE
        PRINT '  ✓ User already exists: worker';
    
    -- 3. complaint_supervisor
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Users WHERE Username = 'complaint_supervisor')
    BEGIN
        INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
        VALUES ('complaint_supervisor', 'TEMP_HASH_sup123', 1);
        PRINT '  ✓ Created user: complaint_supervisor';
    END
    ELSE
        PRINT '  ✓ User already exists: complaint_supervisor';
    
    -- 4. section_admin
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Users WHERE Username = 'section_admin')
    BEGIN
        INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
        VALUES ('section_admin', 'TEMP_HASH_section123', 1);
        PRINT '  ✓ Created user: section_admin';
    END
    ELSE
        PRINT '  ✓ User already exists: section_admin';
    
    -- 5. department_admin
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Users WHERE Username = 'department_admin')
    BEGIN
        INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
        VALUES ('department_admin', 'TEMP_HASH_dept123', 1);
        PRINT '  ✓ Created user: department_admin';
    END
    ELSE
        PRINT '  ✓ User already exists: department_admin';
    
    -- 6. administration_admin
    IF NOT EXISTS (SELECT 1 FROM dbo.APP_Users WHERE Username = 'administration_admin')
    BEGIN
        INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive)
        VALUES ('administration_admin', 'TEMP_HASH_adminis123', 1);
        PRINT '  ✓ Created user: administration_admin';
    END
    ELSE
        PRINT '  ✓ User already exists: administration_admin';
    
    PRINT '';

    -- ========================================================================
    -- SEED DATA: User Role Scopes
    -- ========================================================================
    
    PRINT 'Assigning role scopes to users...';
    
    -- Helper variables for role IDs
    DECLARE @RoleID_SOFTWARE_ADMIN INT = (SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'SOFTWARE_ADMIN');
    DECLARE @RoleID_WORKER INT = (SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'WORKER');
    DECLARE @RoleID_COMPLAINT_SUPERVISOR INT = (SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'COMPLAINT_SUPERVISOR');
    DECLARE @RoleID_SECTION_ADMIN INT = (SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'SECTION_ADMIN');
    DECLARE @RoleID_DEPARTMENT_ADMIN INT = (SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'DEPARTMENT_ADMIN');
    DECLARE @RoleID_ADMINISTRATION_ADMIN INT = (SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = 'ADMINISTRATION_ADMIN');
    
    -- Helper variables for user IDs
    DECLARE @UserID_software_admin INT = (SELECT UserID FROM dbo.APP_Users WHERE Username = 'software_admin');
    DECLARE @UserID_worker INT = (SELECT UserID FROM dbo.APP_Users WHERE Username = 'worker');
    DECLARE @UserID_complaint_supervisor INT = (SELECT UserID FROM dbo.APP_Users WHERE Username = 'complaint_supervisor');
    DECLARE @UserID_section_admin INT = (SELECT UserID FROM dbo.APP_Users WHERE Username = 'section_admin');
    DECLARE @UserID_department_admin INT = (SELECT UserID FROM dbo.APP_Users WHERE Username = 'department_admin');
    DECLARE @UserID_administration_admin INT = (SELECT UserID FROM dbo.APP_Users WHERE Username = 'administration_admin');
    
    -- 1. software_admin → SOFTWARE_ADMIN → OrgUnitID=0 → ADMINISTRATION
    IF NOT EXISTS (
        SELECT 1 FROM dbo.APP_UserRoleScope 
        WHERE UserID = @UserID_software_admin 
        AND RoleID = @RoleID_SOFTWARE_ADMIN 
        AND OrgUnitID = 0 
        AND OrgUnitType = 'ADMINISTRATION'
    )
    BEGIN
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (@UserID_software_admin, @RoleID_SOFTWARE_ADMIN, 0, 'ADMINISTRATION');
        PRINT '  ✓ Assigned: software_admin → SOFTWARE_ADMIN → ADMINISTRATION(0)';
    END
    ELSE
        PRINT '  ✓ Already assigned: software_admin → SOFTWARE_ADMIN → ADMINISTRATION(0)';
    
    -- 2. worker → WORKER → OrgUnitID=10 → COMPLAINT
    IF NOT EXISTS (
        SELECT 1 FROM dbo.APP_UserRoleScope 
        WHERE UserID = @UserID_worker 
        AND RoleID = @RoleID_WORKER 
        AND OrgUnitID = 10 
        AND OrgUnitType = 'COMPLAINT'
    )
    BEGIN
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (@UserID_worker, @RoleID_WORKER, 10, 'COMPLAINT');
        PRINT '  ✓ Assigned: worker → WORKER → COMPLAINT(10)';
    END
    ELSE
        PRINT '  ✓ Already assigned: worker → WORKER → COMPLAINT(10)';
    
    -- 3. complaint_supervisor → COMPLAINT_SUPERVISOR → OrgUnitID=10 → COMPLAINT
    IF NOT EXISTS (
        SELECT 1 FROM dbo.APP_UserRoleScope 
        WHERE UserID = @UserID_complaint_supervisor 
        AND RoleID = @RoleID_COMPLAINT_SUPERVISOR 
        AND OrgUnitID = 10 
        AND OrgUnitType = 'COMPLAINT'
    )
    BEGIN
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (@UserID_complaint_supervisor, @RoleID_COMPLAINT_SUPERVISOR, 10, 'COMPLAINT');
        PRINT '  ✓ Assigned: complaint_supervisor → COMPLAINT_SUPERVISOR → COMPLAINT(10)';
    END
    ELSE
        PRINT '  ✓ Already assigned: complaint_supervisor → COMPLAINT_SUPERVISOR → COMPLAINT(10)';
    
    -- 4. section_admin → SECTION_ADMIN → OrgUnitID=10 → SECTION
    IF NOT EXISTS (
        SELECT 1 FROM dbo.APP_UserRoleScope 
        WHERE UserID = @UserID_section_admin 
        AND RoleID = @RoleID_SECTION_ADMIN 
        AND OrgUnitID = 10 
        AND OrgUnitType = 'SECTION'
    )
    BEGIN
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (@UserID_section_admin, @RoleID_SECTION_ADMIN, 10, 'SECTION');
        PRINT '  ✓ Assigned: section_admin → SECTION_ADMIN → SECTION(10)';
    END
    ELSE
        PRINT '  ✓ Already assigned: section_admin → SECTION_ADMIN → SECTION(10)';
    
    -- 5. department_admin → DEPARTMENT_ADMIN → OrgUnitID=5 → DEPARTMENT
    IF NOT EXISTS (
        SELECT 1 FROM dbo.APP_UserRoleScope 
        WHERE UserID = @UserID_department_admin 
        AND RoleID = @RoleID_DEPARTMENT_ADMIN 
        AND OrgUnitID = 5 
        AND OrgUnitType = 'DEPARTMENT'
    )
    BEGIN
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (@UserID_department_admin, @RoleID_DEPARTMENT_ADMIN, 5, 'DEPARTMENT');
        PRINT '  ✓ Assigned: department_admin → DEPARTMENT_ADMIN → DEPARTMENT(5)';
    END
    ELSE
        PRINT '  ✓ Already assigned: department_admin → DEPARTMENT_ADMIN → DEPARTMENT(5)';
    
    -- 6. administration_admin → ADMINISTRATION_ADMIN → OrgUnitID=1 → ADMINISTRATION
    IF NOT EXISTS (
        SELECT 1 FROM dbo.APP_UserRoleScope 
        WHERE UserID = @UserID_administration_admin 
        AND RoleID = @RoleID_ADMINISTRATION_ADMIN 
        AND OrgUnitID = 1 
        AND OrgUnitType = 'ADMINISTRATION'
    )
    BEGIN
        INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
        VALUES (@UserID_administration_admin, @RoleID_ADMINISTRATION_ADMIN, 1, 'ADMINISTRATION');
        PRINT '  ✓ Assigned: administration_admin → ADMINISTRATION_ADMIN → ADMINISTRATION(1)';
    END
    ELSE
        PRINT '  ✓ Already assigned: administration_admin → ADMINISTRATION_ADMIN → ADMINISTRATION(1)';
    
    PRINT '';

    -- ========================================================================
    -- VERIFICATION: Display Created Data
    -- ========================================================================
    
    PRINT '========================================';
    PRINT 'Verification: Tables and Data';
    PRINT '========================================';
    PRINT '';
    
    PRINT '--- APP_Users ---';
    SELECT 
        UserID,
        Username,
        PasswordHash,
        IsActive,
        CreatedAt
    FROM dbo.APP_Users
    ORDER BY UserID;
    PRINT '';
    
    PRINT '--- APP_Roles ---';
    SELECT 
        RoleID,
        RoleCode,
        RoleNameEn,
        RoleNameAr
    FROM dbo.APP_Roles
    ORDER BY RoleID;
    PRINT '';
    
    PRINT '--- APP_UserRoleScope ---';
    SELECT 
        urs.UserRoleScopeID,
        u.Username,
        r.RoleCode,
        urs.OrgUnitID,
        urs.OrgUnitType
    FROM dbo.APP_UserRoleScope urs
    INNER JOIN dbo.APP_Users u ON urs.UserID = u.UserID
    INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
    ORDER BY u.Username, r.RoleCode;
    PRINT '';
    
    PRINT '========================================';
    PRINT 'Phase 2: RBAC Tables Created Successfully!';
    PRINT '========================================';
    PRINT '';
    PRINT 'Summary:';
    PRINT '- 3 tables created (APP_Users, APP_Roles, APP_UserRoleScope)';
    PRINT '- 6 roles defined';
    PRINT '- 6 test users created';
    PRINT '- 6 role scope assignments configured';
    PRINT '';
    PRINT 'IMPORTANT: Password hashes are temporary placeholders.';
    PRINT 'They will be replaced with bcrypt hashes in the auth service layer.';
    PRINT '';
    
    COMMIT TRANSACTION;
    PRINT '✓ Transaction committed successfully';

END TRY
BEGIN CATCH
    ROLLBACK TRANSACTION;
    
    PRINT '';
    PRINT '========================================';
    PRINT 'ERROR: Transaction rolled back!';
    PRINT '========================================';
    PRINT 'Error Number: ' + CAST(ERROR_NUMBER() AS NVARCHAR);
    PRINT 'Error Message: ' + ERROR_MESSAGE();
    PRINT 'Error Line: ' + CAST(ERROR_LINE() AS NVARCHAR);
    
    -- Re-throw the error
    THROW;
END CATCH;

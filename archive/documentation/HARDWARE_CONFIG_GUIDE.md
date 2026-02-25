# Hardware Configuration Guide
## Understanding the Settings Page

This document explains what each setting means in simple terms.

---

## Tab 1: Database Connection 🗄️

**What is this?**
These settings tell the system WHERE your database is and HOW to connect to it.

| Setting | What It Means | Example |
|---------|---------------|---------|
| **DB_SERVER** | The name or IP address of the computer that has SQL Server | `SOCIALMEDIA` (your dev PC) or `192.168.1.50` (hospital server) |
| **DB_DATABASE** | The name of the database inside SQL Server | `IncidentManager` |
| **DB_DRIVER** | The software used to talk to SQL Server | `ODBC Driver 17 for SQL Server` (don't change this) |
| **USE_WINDOWS_AUTH** | Use your Windows login instead of a username/password | `True` = Use your Windows login, `False` = Use SQL username/password |
| **DB_USERNAME** | SQL Server username (only needed if Windows Auth is OFF) | `sa` or `app_user` |
| **DB_PASSWORD** | SQL Server password (only needed if Windows Auth is OFF) | `****` (hidden for security) |
| **TRUST_SERVER_CERTIFICATE** | Skip certificate checking (needed for self-signed certs) | Usually `True` |

**Simple Explanation:**
Think of this like entering your WiFi settings - you need to tell the system which "network" (server) to connect to and which "WiFi password" (credentials) to use.

---

## Tab 2: External Views 📋

**What is this?**
These are the names of the hospital's existing data tables/views that our system reads from. The hospital already has systems for HR and patients - we just read their data.

| Setting | What It Means | Where the Data Comes From |
|---------|---------------|---------------------------|
| **HR_EMPLOYEES_VIEW** | Name of the HR employee table | Human Resources system (employee names, jobs, departments) |
| **PATIENT_ADMISSION_VIEW** | Name of the patient admissions table | Hospital Information System (patient MRN, names, admission dates) |
| **DOCTORS_VIEW** | Name of the doctors table | Hospital Information System (doctor names, specialties) |

**Simple Explanation:**
The hospital already has databases for employees, patients and doctors. These settings tell our system "what are those tables called?" so we can read information from them.

**Example:**
- Your HR department has a table called `VW_HrEmployeeProfileView`
- When someone files a complaint, we look up employee names from that table
- If the table name changes, you update it here

---

## Tab 3: Network / API 🌐

**What is this?**
These settings control how computers on the network find and connect to this system.

| Setting | What It Means | Example |
|---------|---------------|---------|
| **BACKEND_API_URL** | The full address where the backend server runs | `http://localhost:8000` (your PC) or `http://192.168.1.50:8000` (network) |
| **BACKEND_PORT** | Which "door" (port number) the server listens on | `8000` |
| **BACKEND_HOST** | Who can connect to the server | `127.0.0.1` = Only this computer, `0.0.0.0` = Anyone on the network |
| **CORS_ORIGINS** | Which websites are allowed to talk to the backend | `http://localhost:3000` (the frontend) |

**Simple Explanation:**
Think of this like giving someone your address:
- **BACKEND_HOST** = "Can visitors come to my house?" (`0.0.0.0` = yes, `127.0.0.1` = no, only family)
- **BACKEND_PORT** = "Which door do they knock on?" (8000)
- **BACKEND_API_URL** = "What's the full address?" (http://192.168.1.50:8000)
- **CORS_ORIGINS** = "Who is on my approved visitor list?"

**When do I change this?**
- Development: `127.0.0.1` and `localhost:8000`
- Production: `0.0.0.0` and your server's IP address

---

## Tab 4: Email / SMTP 📧

**What is this?**
These settings tell the system how to send email notifications (like when a complaint is assigned to someone).

| Setting | What It Means | Example |
|---------|---------------|---------|
| **NOTIFICATION_MODE** | Should the system actually send emails? | `mock` = No, just log them, `smtp` = Yes, send real emails |
| **SMTP_HOST** | The email server address | `smtp.hospital.local` or `172.16.0.10` |
| **SMTP_PORT** | Which port the email server uses | `25` (common), `587` (TLS), `465` (SSL) |
| **SMTP_USE_TLS** | Use secure connection (TLS) | `True` for Office 365, `False` for internal servers |
| **SMTP_USE_SSL** | Use SSL encryption | Usually `False` unless using port 465 |
| **SMTP_USERNAME** | Email account to send from (if authentication required) | `notifications@hospital.org` |
| **SMTP_PASSWORD** | Password for that email account | `****` (hidden) |
| **SENDER_EMAIL** | The "From" address on emails | `complaint-system@hospital.org` |
| **SENDER_NAME** | The name that appears on emails | `Hospital Complaint System` |

**Simple Explanation:**
To send emails, the system needs to know which email server to use, like setting up Outlook on a new computer.

**Common Scenarios:**
1. **Internal Exchange Server (no authentication):**
   - SMTP_HOST = `mail.hospital.local`
   - SMTP_PORT = `25`
   - SMTP_USERNAME = empty
   
2. **Office 365:**
   - SMTP_HOST = `smtp.office365.com`
   - SMTP_PORT = `587`
   - SMTP_USE_TLS = `True`
   - SMTP_USERNAME = your email

---

## Tab 5: System ⚙️

**What is this?**
General system settings, including the deployment mode.

| Setting | What It Means | Options |
|---------|---------------|---------|
| **DEPLOYMENT_MODE** | Is this a test system or the real one? | `development` = Testing/development, `production` = Real hospital use |

**Simple Explanation:**

Think of it like this:
- **Development Mode** = You're building/testing the system at your desk. The database is on your computer. You can break things without affecting real users.
- **Production Mode** = The system is LIVE. Real employees are using it. Real complaints are being filed. Be careful with changes!

**Why does this matter?**
- In development: You might use fake data, skip email sending, use localhost
- In production: Everything is real - real database, real emails, real network

---

## Quick Reference: What to Change When Moving to Production

When you move from your development computer to the hospital network:

1. **Database Tab:**
   - Change `DB_SERVER` from `SOCIALMEDIA` to the hospital server IP
   
2. **Network Tab:**
   - Change `BACKEND_HOST` from `127.0.0.1` to `0.0.0.0`
   - Change `BACKEND_API_URL` to use the server's IP
   - Add the production frontend URL to `CORS_ORIGINS`

3. **Email Tab:**
   - Change `NOTIFICATION_MODE` from `mock` to `smtp`
   - Set the real `SMTP_HOST`

4. **System Tab:**
   - Change `DEPLOYMENT_MODE` from `development` to `production`

---

## Testing Connections

Each tab has a "Test Connection" button:

- **Database Test**: Tries to connect to SQL Server and run a simple query
- **SMTP Test**: Tries to connect to the email server

| Result | What It Means |
|--------|---------------|
| ✅ **Connected** | Everything is working! |
| ❌ **Failed** | Something is wrong - check the error message |

**Common Problems:**
- "Connection refused" = Wrong server name/IP or server is off
- "Login failed" = Wrong username/password
- "Network error" = Firewall blocking the connection

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR COMPLAINT SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │   FRONTEND   │ ◄─── Runs in the browser (localhost:3000)    │
│  │   (React)    │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         │ Network Tab settings                                  │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │   BACKEND    │ ◄─── Runs on server (localhost:8000)         │
│  │   (Python)   │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         │ Database Tab settings                                 │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  SQL SERVER  │ ◄─── Your database (SOCIALMEDIA)             │
│  │  (Database)  │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         │ External Views Tab settings                           │
│         ▼                                                       │
│  ┌──────────────────────────────────────┐                       │
│  │  HOSPITAL SYSTEMS (HR, HIS)          │                       │
│  │  - VW_HrEmployeeProfileView          │                       │
│  │  - VW_PatientAdmission               │                       │
│  │  - VW_Doctors                        │                       │
│  └──────────────────────────────────────┘                       │
│                                                                 │
│  Email Tab: How to send notification emails                     │
│  System Tab: Is this dev or production?                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Need Help?

If you're not sure what to put in a field:
1. Ask your IT department for server names and IPs
2. Use the "Test Connection" button to verify your settings
3. Start in `development` mode and test everything before switching to `production`

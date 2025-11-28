from datetime import datetime, timedelta
from random import randint, choice
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Patient Feedback Analysis", layout="wide")

# ============================================================
# TOP HORIZONTAL BAR
# ============================================================
st.markdown("""
<style>
    .top-bar {
        display: flex;
        gap: 20px;
        padding: 10px 0;
        border-bottom: 1px solid #ccc;
        margin-bottom: 20px;
    }
    .top-btn {
        background-color: #F5F5F5;
        padding: 8px 16px;
        border-radius: 6px;
        border: 1px solid #ddd;
        cursor: pointer;
        font-weight: 500;
    }
    .top-btn:hover {
        background-color: #e9e9e9;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-bar">
    <div class="top-btn">Insert</div>
    <div class="top-btn">Delete</div>
    <div class="top-btn">Export</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LEFT SIDEBAR MENU
# ============================================================
with st.sidebar:
    st.header("Admin Panel")
    st.write("Logged in as: **Admin**")

    page = st.radio(
        "Navigation",
        ["Table View", "Insert Record", "Dashboard", "Reporting"]
    )

# ============================================================
# REQUIRED FIELDS (from your list)
# ============================================================
RECORD_FIELDS = [
    "feedback_received_date",
    "record_id",
    "patient_full_name",
    "issuing_department",
    "target_department",
    "source_1",
    "feedback_type",
    "domain",
    "category",
    "sub_category",
    "classification_ar",
    "complaint_text",
    "immediate_action",
    "taken_action",
    "severity_level",
    "stage",
    "harm_level",
    "status",
    "improvement_opportunity_type"
]
issuing_departments = [
    "cardiac 1",
    "cardiac 2",
    "cardiac 3",
    "CCU",
    "CSU",
    "ICN",
    "ICU",
    "ICU ارضي-تمريضي",
    "Lab-قسم التحاليل المخبرية",
    "New Cardiac3",
    "Post CSU",
    "ارضي-إستشفاء",
    "الأطفال",
    "الادارة التمريضية",
    "الادارة الطبية",
    "الرابع الغربي",
    "الرابع جديد ext",
    "الرابع شرقي",
    "الرابع شمالي",
    "الطابق الثاني",
    "العيادات الخارجية-OPD",
    "ثالث جديد",
    "ثالث شرقي",
    "ثالث غربي",
    "جراحة الأطفال",
    "دائرة الطوارئ التمريضية",
    "عيادات القلب المفتوح",
    "عيادات-طبية",
    "عيادة قلب - طبية",
    "غسيل الكلى",
    "قسم الأعمال غير التداخلية - تمريضي",
    "قسم التشخيص",
    "قسم التصوير الطبي",
    "قسم التوليد والجراحة النسائية-تمريضية",
    "قسم بنك الدم",
    "قسم عناية جراحة الكبد وزرع الأعضاء-ITU",
    "مكتب الدخول",
    "وحدة الإستعلامات والسنترال",
    "وحدة عيادات القلب- تمريض",
    "وحدة ما قبل الدخول"
]
target_departments = [
    "Call Center",
    "cardiac 1",
    "cardiac 2",
    "cardiac 3",
    "CCU",
    "CSU",
    "ICN",
    "ICU",
    "ICU -طبي",
    "Lab-قسم التحاليل المخبرية",
    "New Cardiac3",
    "Post CSU",
    "أقسام أمراض القلب-طبي",
    "ارضي-إستشفاء",
    "الأطفال",
    "الادارة الطبية",
    "التخدير - BCI",
    "الجراحة القلبية",
    "الخدمات البيئية",
    "الرابع الغربي",
    "الرابع جديد ext",
    "الرابع شرقي",
    "الرابع شمالي",
    "العيادات الخارجية-OPD",
    "المباني",
    "المطبخ-التغذية",
    "الهندسة الطبية",
    "ثالث جديد",
    "ثالث شرقي",
    "ثالث غربي",
    "جراحة الأطفال",
    "دائرة الطوارئ التمريضية",
    "دائرة الطوارئ الطبية",
    "دائرة المواد",
    "صيانة المعلوماتية",
    "عيادات القلب المفتوح",
    "قسم  الميكانيك",
    "قسم أمراض الكلى والضغط- طبية",
    "قسم أمراض قلب الأطفال",
    "قسم أمراض كهرباء القلب",
    "قسم الأعمال غير التداخلية - تمريضي",
    "قسم الاعمال التداخلية -تمريض",
    "قسم الامراض الجرثومية-طبي",
    "قسم الامراض الصدرية - طبية",
    "قسم الامن",
    "قسم التخدير والإنعاش",
    "قسم التشخيص",
    "قسم التصوير الطبي",
    "قسم التوليد والجراحة النسائية-تمريضية",
    "قسم الجراحة العامة -طبي",
    "قسم العمليات العامة",
    "قسم الفوترة",
    "قسم الكهرباء",
    "قسم امراض الأعصاب -الطبية",
    "قسم امراض الجهاز الهضمي -طبية",
    "قسم امراض العظم -طبي",
    "قسم امراض العيون-طبي",
    "قسم امراض المسالك البولية-طبية",
    "قسم بنك الدم",
    "قسم جراحة الاعصاب والدماغ -طبي",
    "قسم جراحة الشرايين والصدر-طبي",
    "مكتب الدخول",
    "مكتب الوافدين",
    "وحدة الأوردرلي",
    "وحدة الإستعلامات والسنترال",
    "وحدة جراحة الاطفال -طبي",
    "وحدة عيادات القلب- تمريض",
    "وحدة ما قبل الدخول"
]
category = [
    "Communication",
    "Environement",
    "Institutional Processes",
    "Listening",
    "Quality of Care",
    "Respect & Patient Rights",
    "Safety"
]
sub_categories = [
    "Neglect -General",
    "Absent Communication",
    "Accomodation",
    "Bureaucracy",
    "Clinician -Errors",
    "Delay -Access",
    "Delay -General",
    "Delay -Procedure",
    "Delayed Communication",
    "Dimissing Patients",
    "Disrespect",
    "Documentation",
    "Equipement",
    "Error - Diagnosis",
    "Error -General",
    "Error -Medication",
    "Examination & Monitoring",
    "Failure to Provide",
    "Failure to Respond",
    "Ignoring Patients",
    "Incorrect Communication",
    "Neglect -Hygiene & Personal Care",
    "Rights",
    "Security",
    "Teamwork",
    "Visiting",
    "Ward Cleanliness"
]
severity = [
    "HIGH",
    "MEDIUM",
    "LOW",
    "Moderate"
]
stage = [
    "Examination &Diagnosis",
    "Admissions",
    "Care on the Ward",
    "Discharge/Transfer",
    "Operation/Procedure",
    "Unspecified"
]
harm_level = [
    "Severe Harm",
    "Death",
    "High Severe",
    "Moderate Harm",
    "Minor Harm",
    "No Harm"
]
status = [
    "Closed",
    "In Progress",
    "Red Flag"
]


issuing_departments_all = ["All"] + issuing_departments
target_departments_all = ["All"] + target_departments
feedback_categories_all = ["All"] + category
sub_categories_all = ["All"] + sub_categories
severity_all = ["All"] + severity
stage_all = ["All"] + stage
harm_level_all = ["All"] + harm_level
status_all = ["All"] + status
domain_all = ["All", "Clinical", "Management", "Relational"]
category_all = ["All", "Clinical", "Management", "Relational"]

if page == "Table View":
    st.title("Incident Records")

    st.subheader("Search & Filters")

    # ----------- First Row -----------
    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns([2, 2, 2, 2])

    with row1_c1:
        search_name = st.text_input("Search by Name / ID")

    with row1_c2:
        selected_issuing = st.selectbox("Filter by Issuing Department", ["All"] + issuing_departments)

    with row1_c3:
        selected_target = st.selectbox("Filter by Target Department", ["All"] + target_departments)

    with row1_c4:
        selected_feedback_category = st.selectbox("Filter by Source", ["All"] + category)

    st.markdown("")  # spacing

    # ---------------- Second Row ----------------

    with st.expander("Show Advanced Filters"):
        row2_c1, row2_c2, row2_c3, row2_c4 = st.columns([2, 2, 2, 2])

        with row2_c1:
            selected_severity = st.selectbox("Filter by Severity", ["All"] + severity)
            selected_stage = st.selectbox("Filter by Stage", ["All"] + stage)
            selected_harm = st.selectbox("Filter by Harm", ["All"] + harm_level)

        with row2_c2:
            selected_domain = st.selectbox("Filter by Domain", ["All", "Clinical", "Management", "Relational"])
            selected_category = st.selectbox("Filter by Category", ["All", "Clinical", "Management", "Relational"])
            selected_subcategory = st.selectbox("Filter by Sub-Category", ["All"] + sub_categories)

        with row2_c3:
            selected_status = st.selectbox("Filter by Status", ["All"] + status)

        with row2_c4:
            start_date = st.date_input("Start Date", value=None, key="start_date")
            end_date = st.date_input("End Date", value=None, key="end_date")


    st.subheader("Records Table")


    # EMPTY TABLE WITH YOUR EXACT COLUMNS
    # Your exact columns
    RECORD_FIELDS = [
        "feedback_received_date", "record_id", "patient_full_name", "issuing_department",
        "target_department", "source_1", "feedback_type", "domain", "category", "sub_category",
        "classification_ar", "complaint_text", "immediate_action", "taken_action",
        "severity_level", "stage", "harm_level", "status", "improvement_opportunity_type"
    ]

    # Generate 10 example records
    example_data = []
    domains = ["CLINICAL", "MANAGEMENT", "RELATIONAL"]
    categories = ["Safety", "Quality", "Environment"]
    stage = ["Admission", "Care", "Discharge"]
    statuses = ["Open", "Closed", "In Progress"]
    harm_level = ["Low", "Medium", "High"]
    improvement_opportunity = ["Yes", "No"]
    departments = ["ER", "Ward 1", "Radiology", "Cardiology", "Admin"]

    for i in range(1, 11):
        example_data.append({
            "feedback_received_date":(datetime.today() - timedelta(days=randint(0, 30))).strftime("%Y-%m-%d"),
            "record_id": f"INC{i:04d}",
            "patient_full_name": f"Patient {i}",
            "issuing_department": choice(departments),
            "target_department": choice(departments),
            "source_1": f"Source {randint(1, 3)}",
            "feedback_type": choice(["Complaint", "Suggestion"]),
            "domain": choice(domains),
            "category": choice(categories),
            "sub_category": f"Sub-{randint(1, 5)}",
            "classification_ar": f"تصنيف {i}",
            "complaint_text": f"This is a sample complaint text for record {i}.",
            "immediate_action": f"Action {i}",
            "taken_action": f"Taken Action {i}",
            "severity_level": choice(["Low", "Medium", "High"]),
            "stage": choice(stage),
            "harm_level": choice(harm_level),
            "status": choice(statuses),
            "improvement_opportunity_type": choice(improvement_opportunity)
        })

    # Convert to DataFrame

    df = pd.DataFrame(example_data)
    st.dataframe(df, use_container_width=True)


# ============================================================
# PAGE 2: INSERT RECORD
# ============================================================
elif page == "Insert Record":

    st.title("Insert New Incident Record")

    # --------------------- TEXT INPUT BLOCKS ---------------------
    st.subheader("1. Main Text Inputs")
    text_col1, text_col2 = st.columns(2)

    with text_col1:
        st.text_area("Complaint Text (Raw)", height=200, key="complaint_text")

    with text_col2:
        st.text_area("Additional Notes", height=200, key="notes_text")

    st.markdown("---")

    # --------------------- NER BLOCK ---------------------
    st.subheader("2. Named Entity Recognition (NER Outputs)")
    n1, n2 = st.columns(2)
    with n1:
        st.text_input("Patient Name (NER)", key="ner_patient")
    with n2:
        st.text_input("Doctor Name (NER)", key="ner_doctor")

    st.markdown("---")

    # --------------------- CLASSIFICATION BLOCK ---------------------
    st.subheader("3. Classification Fields (Editable)")

    for i in range(1, 9):
        st.text_input(f"Classification {i}", key=f"class_{i}")

    st.markdown("---")

    # --------------------- METADATA BLOCK ---------------------
    st.subheader("4. Record Metadata")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.date_input("Feedback Received Date")
    with m2:
        st.text_input("Record ID")
    with m3:
        st.text_input("Patient Full Name")

    m4, m5, m6 = st.columns(3)
    with m4:
        st.selectbox("Issuing Department", ["", "ER", "ICU", "OPD", "Surgery"])
    with m5:
        st.selectbox("Target Department", ["", "ER", "ICU", "OPD", "Surgery"])
    with m6:
        st.selectbox("Source 1", ["", "Phone", "Walk-in", "Email"])

    m7, m8, m9 = st.columns(3)
    with m7:
        st.selectbox("Feedback Type", ["", "Complaint", "Suggestion", "Request"])
    with m8:
        st.selectbox("Domain", ["", "Clinical", "Management", "Relational"])
    with m9:
        st.text_input("Arabic Classification")

    m10, m11, m12 = st.columns(3)
    with m10:
        st.text_input("Category")
    with m11:
        st.text_input("Sub-Category")
    with m12:
        st.selectbox("Severity Level", ["", "High", "Medium", "Low"])

    m13, m14, m15 = st.columns(3)
    with m13:
        st.text_input("Stage")
    with m14:
        st.text_input("Harm Level")
    with m15:
        st.selectbox("Status", ["", "Open", "Closed", "Pending"])

    st.selectbox("Improvement Opportunity Type", ["", "Yes", "No"])

    st.markdown("---")

    # --------------------- ACTION BUTTONS ---------------------
    b1, b2 = st.columns(2)
    with b1:
        st.button("Extract Data (NER + Classification)", type="primary")
    with b2:
        st.button("Add Record", type="primary")


# ============================================================
# PAGE 3: DASHBOARD
# ============================================================
elif page == "Dashboard":
    st.title("Dashboard")
    st.write("📊 Summary charts will go here later.")

# ============================================================
# PAGE 4: REPORTING
# ============================================================
elif page == "Reporting":
    st.title("Reporting")
    st.write("📄 Downloadable reports will appear here.")
